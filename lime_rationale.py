# !/usr/bin/env python
# coding: utf-8

import os
import json 
import time
import datetime
import numpy as np
import pandas as pd
from lime.lime_text import LimeTextExplainer
from transformers import AutoTokenizer, BertForSequenceClassification

CN_tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-macbert-large")
test_data_path = "./data/sim_interpretation_B.txt"
pred_model_output_dir = './macbert_large_with_EF'

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


LCQMC_testB_dic = {'id':[], 'query':[], 'title':[], 'text_q_seg':[],'text_t_seg':[] } 
with open(test_data_path, 'r') as f:
    for line in f:
        line_dic = json.loads(line)
        for k in line_dic.keys():
            LCQMC_testB_dic[k].append(line_dic[k]) 

LCQMC_testB = pd.DataFrame.from_dict(LCQMC_testB_dic)
LCQMC_testB['sentence'] = LCQMC_testB['query'] +"[SEP]" + LCQMC_testB['title']
LCQMC_testB

sentence_with_space = []
for i in range(len(LCQMC_testB)):
    space_sentence = ' '.join(LCQMC_testB['text_q_seg'][i]+['[SEP]']+LCQMC_testB['text_t_seg'][i])
    space_sentence = space_sentence.replace('   ', ' [PAD] ') #using [PAD] to replace ' ' token in space_sentence
    sentence_with_space.append(space_sentence)
LCQMC_testB['sentence_with_space'] = sentence_with_space
LCQMC_testB.head(3)


model_interpret = BertForSequenceClassification.from_pretrained(pred_model_output_dir)
def predictor(texts):
    model_input = CN_tokenizer(texts, return_tensors="pt", add_special_tokens = True, padding=True)
    model_output = model_interpret(**model_input)
    tensor_logits =model_output[0] 
    probas = tensor_logits.detach().numpy()
    return probas 

def lime_obtain_output_dic(LCQMC_test, bow = False):
    t0 = time.time() 
    output = {"id": [], "label": [], "rationale": [], "rationale_score": []}
    number = 0 
    for ID in LCQMC_test['id']:
        output["id"].append(ID)

        LCQMC_test_Row  = LCQMC_test[LCQMC_test['id'] == ID]

        text = LCQMC_test_Row['sentence_with_space'].item()
        proba = predictor(text)
        pred_label = np.argmax(proba ,axis=1).item()
        output['label'].append(pred_label)
        
        LIME_exp = LimeTextExplainer(class_names=['0','1'],  bow=bow, split_expression=' |\[SEP\]' , 
                                     mask_string = '[UNK]')
        exp = LIME_exp.explain_instance(text, predictor,  num_features=100, num_samples=3000)
        #show plot:
        #exp.show_in_notebook(text=text)
        
        assert exp.available_labels()[0] == 1
        _list = np.array(LCQMC_test_Row['text_q_seg'].item() + ['[SEP]'] + LCQMC_test_Row['text_t_seg'].item())
        
        assert len(np.where(_list == '[SEP]')[0]) == 1  #make sure only one [SEP]
        SEP_index = np.where(_list == '[SEP]')[0].item()

        if not bow:  #do not use bag of words
            assert len(exp.as_map()[1]) == len(_list)-1   #make sure # score = # token
            sorted_token1 = sorted(exp.as_map()[1], key=lambda t: t[1], reverse=True)    
            sorted_token0 = sorted(exp.as_map()[1], key=lambda t: t[1], reverse=False)  
        else: #use bag of words
            try:
                QT_list = np.array(LCQMC_test_Row['text_q_seg'].item() +  LCQMC_test_Row['text_t_seg'].item())
                sorted_token1 = sorted([(i, dict(exp.as_list())[QT_list[i]]) for i in range(len(QT_list))], key=lambda t: t[1], reverse=True)    
                sorted_token0 = sorted([(i, dict(exp.as_list())[QT_list[i]]) for i in range(len(QT_list))], key=lambda t: t[1], reverse=False)    
            except KeyError: #lime use [PAD] to repalce space 
                QT_list = np.array([i.replace(' ','[PAD]') for i in LCQMC_test_Row['text_q_seg'].item()] +  
                                   [i.replace(' ','[PAD]') for i in LCQMC_test_Row['text_t_seg'].item()])
                sorted_token1 = sorted([(i, dict(exp.as_list())[QT_list[i]]) for i in range(len(QT_list))], key=lambda t: t[1], reverse=True)    
                sorted_token0 = sorted([(i, dict(exp.as_list())[QT_list[i]]) for i in range(len(QT_list))], key=lambda t: t[1], reverse=False)    

        if pred_label == 1:
            imp_ids_q = [item[0] for item in sorted_token1 if item[0] < SEP_index]
            imp_ids_t = [item[0]-SEP_index for item in sorted_token1 if item[0] > SEP_index-1]
            imp_scr_q = [item[1] for item in sorted_token1 if item[0] < SEP_index]
            imp_scr_t = [item[1] for item in sorted_token1 if item[0] > SEP_index-1]
            
        if pred_label == 0:
            imp_ids_q = [item[0] for item in sorted_token0 if item[0] < SEP_index]
            imp_ids_t = [item[0]-SEP_index for item in sorted_token0 if item[0] > SEP_index-1]
            imp_scr_q = [item[1] for item in sorted_token0 if item[0] < SEP_index]
            imp_scr_t = [item[1] for item in sorted_token0 if item[0] > SEP_index-1]
            
        rationale = [imp_ids_q, imp_ids_t]
        rationale_score = [imp_scr_q, imp_scr_t]
        output['rationale'].append(rationale)
        output['rationale_score'].append(rationale_score)
        
        print(f'{number}, Elapsed time: {format_time(time.time() - t0)}')
        number += 1 

    if not os.path.exists('lime_scores'):
        os.mkdir('lime_scores')
    np.save('lime_scores/output.npy', output) 
    
    return (output)


# Obtain LIME scores
lime_output = lime_obtain_output_dic(LCQMC_testB, ,bow = False)

# Load LIME scores
lime_output = np.load('./lime_scores/output.npy',allow_pickle=True).item()