# !/usr/bin/env python
# coding: utf-8

from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from captum.attr import visualization as viz
from captum.attr import IntegratedGradients, LayerConductance, LayerIntegratedGradients
from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer
import torch
import json
import os
import numpy as np
import pandas as pd
TEST_DATA_PATH = "./data/sim_interpretation_B.txt"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load model
TRAINED_MODEL_PATH =  './macbert_large_with_EF'
model = BertForSequenceClassification.from_pretrained(TRAINED_MODEL_PATH)
model.to(device)
model.eval()
model.zero_grad()

# load tokenizer
tokenizer = BertTokenizer.from_pretrained(TRAINED_MODEL_PATH)

def predict(inputs):
    return model(inputs)[0]


ref_token_id = tokenizer.pad_token_id # A token used for generating token reference
sep_token_id = tokenizer.sep_token_id # A token used as a separator between question and text and it is also added to the end of the text.
cls_token_id = tokenizer.cls_token_id # A token used for prepending to the concatenated question-text word sequence


def construct_input_ref_pair(text, ref_token_id, sep_token_id, cls_token_id):

    text_ids = tokenizer.encode(text, add_special_tokens=False)
    # construct input token ids
    input_ids = [cls_token_id] + text_ids + [sep_token_id]
    # construct reference token ids 
    ref_input_ids = [cls_token_id] + [ref_token_id] * len(text_ids) + [sep_token_id]

    return torch.tensor([input_ids], device=device), torch.tensor([ref_input_ids], device=device), len(text_ids)

def construct_input_ref_token_type_pair(input_ids, sep_ind=0):
    seq_len = input_ids.size(1)
    token_type_ids = torch.tensor([[0 if i <= sep_ind else 1 for i in range(seq_len)]], device=device)
    ref_token_type_ids = torch.zeros_like(token_type_ids, device=device)# * -1
    return token_type_ids, ref_token_type_ids

def construct_input_ref_pos_id_pair(input_ids):
    seq_length = input_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
    # we could potentially also use random permutation with `torch.randperm(seq_length, device=device)`
    ref_position_ids = torch.zeros(seq_length, dtype=torch.long, device=device)

    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
    ref_position_ids = ref_position_ids.unsqueeze(0).expand_as(input_ids)
    return position_ids, ref_position_ids
    
def construct_attention_mask(input_ids):
    return torch.ones_like(input_ids)

def custom_forward(inputs):
    preds = predict(inputs)
    return torch.softmax(preds, dim = 1)[0][0].unsqueeze(-1)


lig = LayerIntegratedGradients(custom_forward, model.bert.embeddings)

LCQMC_testB_dic = {'id':[], 'query':[], 'title':[], 'text_q_seg':[],'text_t_seg':[] } 
with open(TEST_DATA_PATH, 'r') as f:
    for line in f:
        line_dic = json.loads(line)
        for k in line_dic.keys():
            LCQMC_testB_dic[k].append(line_dic[k])
LCQMC_testB = pd.DataFrame.from_dict(LCQMC_testB_dic)
LCQMC_testB['sentence'] = LCQMC_testB['query'] +"[SEP]" + LCQMC_testB['title']


def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions

ig_dic = {'id':[], 'label':[], 'rationale':[], 'rationale_score':[]}
for i in range(len(LCQMC_testB)): #

    LCQMC_testB_line = LCQMC_testB.iloc[i]
    question_pair_text = LCQMC_testB_line['query'] + '[SEP]' + LCQMC_testB_line['title']
    q_len, t_len = len(LCQMC_testB_line['text_q_seg']),  len(LCQMC_testB_line['text_t_seg'])
    sep_loc =  len(LCQMC_testB_line['text_q_seg'])

    input_ids, ref_input_ids, sep_id = construct_input_ref_pair(question_pair_text, ref_token_id, sep_token_id, cls_token_id)
    token_type_ids, ref_token_type_ids = construct_input_ref_token_type_pair(input_ids, sep_id)
    position_ids, ref_position_ids = construct_input_ref_pos_id_pair(input_ids)
    attention_mask = construct_attention_mask(input_ids)
    indices = input_ids[0].detach().tolist()
    all_tokens = tokenizer.convert_ids_to_tokens(indices)
    score = predict(input_ids) 
    label = torch.argmax(score).item()

    attributions, delta = lig.attribute(inputs=input_ids,
                                        baselines=ref_input_ids,
                                        return_convergence_delta=True)
    attributions_sum = summarize_attributions(attributions).tolist()[1:-1]
    q_attributions, t_attributions = attributions_sum[:sep_loc],  attributions_sum[sep_loc+1:]
    assert q_len == len(q_attributions)
    assert t_len == len(t_attributions)
    
    ig_dic['id'].append(i)
    ig_dic['label'].append(label)
    ig_dic['rationale'].append([  [i for i in range(q_len)], [j for j in range(t_len)] ])
    ig_dic['rationale_score'].append([q_attributions, t_attributions])


if not os.path.exists('ig_scores'):
    os.mkdir('ig_scores')
np.save('./ig_scores/output.npy', ig_dic) 
