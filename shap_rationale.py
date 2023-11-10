# !/usr/bin/env python
# coding: utf-8

import re
import os
import json
import time
import shap #v0.41.0
import torch
import datetime
import numpy as np #v0.41.0
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from matplotlib.pyplot import figure
from transformers import AutoTokenizer, BertForSequenceClassification, TextClassificationPipeline


CN_tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-macbert-large")
MODEL_PATH = './macbert_large_with_EF'
MODEL = BertForSequenceClassification.from_pretrained(MODEL_PATH)
# PIPE = TextClassificationPipeline(model=MODEL, tokenizer=CN_tokenizer, return_all_scores=True)
PIPE = TextClassificationPipeline(model=MODEL, tokenizer=CN_tokenizer, top_k=None)

if torch.cuda.is_available():    
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def get_score(text):
    prediction = PIPE([text])
    return prediction


LCQMC_testB_dic = {'id':[], 'query':[], 'title':[], 'text_q_seg':[],'text_t_seg':[] } 
with open(r"./data/sim_interpretation_B.txt", 'r') as f:
    for line in f:
        line_dic = json.loads(line)
        for k in line_dic.keys():
            LCQMC_testB_dic[k].append(line_dic[k]) 

LCQMC_testB = pd.DataFrame.from_dict(LCQMC_testB_dic)
LCQMC_testB['sentence'] = LCQMC_testB['query'] +"[SEP]" + LCQMC_testB['title']
LCQMC_testB


def shap_obtain_output_dic(LCQMC_test, col='sentence', algorithm = 'auto', per_n = 3000):
    '''
    col: 'sentence', 'sentence_with_space'
    algorithm: permutation , auto
    per_n : >= 10
    '''
    output = {"id": [], "label": [], "rationale": [], "rationale_score": []}
    number = 0 
    
    SHAP_explainer = shap.Explainer(PIPE, algorithm=algorithm)  
    t0 = time.time()
    
    for ID in LCQMC_test['id']:
        output["id"].append(ID)
        LCQMC_test_Row  = LCQMC_test[LCQMC_test['id'] == ID]
        text = LCQMC_test_Row[col].item()
        model_logits_dic =  get_score(text)
        pred_label = np.argmax(np.array([model_logits_dic[0][0]['score'], model_logits_dic[0][1]['score']]))
        output["label"].append(pred_label)
    
        shap_values_total = 0 
        for k10 in range(int(per_n/10)):
            raw_shap_values = SHAP_explainer([text])
            assert raw_shap_values.data[0][0] == ''  
            assert raw_shap_values.data[0][-1] == '' 
            
            shap_values = raw_shap_values.values[0][1:-1,:]
            shap_data = raw_shap_values.data[0][1:-1]
            data_list = LCQMC_test_Row['text_q_seg'].item() + ['[SEP]']  + LCQMC_test_Row['text_t_seg'].item()
            if len(shap_data) > len(data_list): #for multiple characters in one token
                print(f'multiple characters in one token: ID = {ID}')
                i = 0 
                shap_data_len = len(shap_data)
                shap_data = shap_data.tolist()
                while i < shap_data_len: 
                    if shap_data[i] not in data_list :
                        shap_data[i+1] = shap_data[i] + shap_data[i+1]
                        shap_data[i] = 'na'
                        shap_values[i+1] = (shap_values[i] + shap_values[i+1]) / 2 
                        shap_values[i] = 0.0
                        i+=1
                    else:
                        i+=1
                shap_data = np.array(shap_data)
                SEP_index = np.where(shap_data == 'na')[0]
                shap_data = np.delete(shap_data, SEP_index)
                shap_values = np.delete(shap_values, SEP_index, axis = 0)
            elif len(shap_data) < len(data_list): #for the case shap combine multiple token into one characters
                print(f'less characters (combine tokens) in sharp: ID = {ID}')
                len_shap_data = len(shap_data) 
                i = 0 
                while i < len_shap_data: 
                    if shap_data[i] not in data_list:
                        length_i = len(shap_data[i])
                        insert_ele = list(shap_data[i])
                        for k in range(length_i):
                            shap_data = np.insert(shap_data, i+k+1, insert_ele[k], axis=0)
                            shap_values = np.insert(shap_values, i+k+1, shap_values[i], axis=0)
                            i+=1
                        shap_data = np.delete(shap_data, i-k)
                        shap_values = np.delete(shap_values, i-k, axis=0)       
                            
                    else:
                        i+=1
            else:
                pass
            
            shap_values_total += shap_values 
        if col == 'sentence_with_space':
            assert len(np.where(shap_data == '[SEP] ')[0]) == 1  #make sure only one [SEP]
            SEP_index = np.where(shap_data == '[SEP] ')[0].item()
        if col == 'sentence':
            assert len(np.where(shap_data == '[SEP]')[0]) == 1  #make sure only one [SEP]
            SEP_index = np.where(shap_data == '[SEP]')[0].item()

        all_imp_index = np.array([i for i in range(len(data_list))])
        q_imp_index = all_imp_index[all_imp_index<SEP_index]
        t_imp_index = all_imp_index[all_imp_index>SEP_index] - SEP_index -1
        output["rationale"].append([list(q_imp_index),list(t_imp_index)])
        
        output["rationale_score"].append([list(shap_values_total[:,0][q_imp_index]),list(shap_values_total[:,0][t_imp_index+SEP_index +1])])
        number += 1      
        print(f'{number}: Elapsed time: {format_time(time.time() - t0)}')
    
    if not os.path.exists('shap_output'):
        os.mkdir('shap_output')
    np.save('shap_output/output.npy', output) 
        
    return (output)
    
shap_output = shap_obtain_output_dic(LCQMC_testB, col='sentence', algorithm = 'auto', per_n = 100)

