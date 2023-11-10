# !/usr/bin/env python
# coding: utf-8
import os
import json
import torch
import string
import random
import time
import datetime
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from zhon.hanzi import punctuation as CHN_punctuation

from transformers import AutoTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

CN_tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-macbert-large")
ENG_punctuation =  string.punctuation
P_LIST = list(ENG_punctuation) + list(CHN_punctuation) 

seed_val = 2023
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:    
    device = torch.device("cpu") 


#function for calculating accuracy
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

#function for timing
def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

#function for calculating Levenshtein ratio 
import Levenshtein
def Levenshtein_similarity(string1, string2):
    '''
    output: scalar, Levenshtein similarity of string1 & string2
    '''
    Levenshtein_ratio = Levenshtein.ratio(string1, string2)
    return(Levenshtein_ratio)


# prepare training data
max_length = 74 #define the longest sentnece
LCQMC_train = pd.read_csv('./data/lcqmc/train.tsv', sep='\t', header=None)
LCQMC_train.rename(columns = {0:'query', 1:'title', 2:'label'}, inplace = True)
LCQMC_train['query'] = [q if q[-1] not in P_LIST else q[:-1] for q in LCQMC_train['query']] #remove punctuation
LCQMC_train['title'] = [q if q[-1] not in P_LIST else q[:-1] for q in LCQMC_train['title']] #remove punctuation
LCQMC_train['sentence'] = LCQMC_train['query'] +"[SEP]" + LCQMC_train['title']

#calculating Levenshtein similarity of each (query title) pair
LCQMC_train['LS'] = [Levenshtein_similarity(q,t) for q, t in zip(LCQMC_train['query'], LCQMC_train['title'])]
LCQMC_train['weightLS1'] = LCQMC_train['label'] * (1-LCQMC_train['LS']) + (1-LCQMC_train['label'])*LCQMC_train['LS'] 
minWeightLS = min([i for i in LCQMC_train['weightLS1'] if i!=0])
LCQMC_train['weightLS1'] = np.array([i if i!=0 else minWeightLS for i in LCQMC_train['weightLS1'] ])

# remove long training sentence:
longSen_index = []
for i in range(len(LCQMC_train)):
    if len(LCQMC_train['query'][i]) + len(LCQMC_train['title'][i]) + 1 > max_length-2:
        longSen_index.append(i)
LCQMC_train = LCQMC_train.drop(longSen_index)

LCQMC_train = LCQMC_train.reset_index(drop=True)
LCQMC_train['weightLS'] = LCQMC_train['weightLS1']/np.mean(LCQMC_train['weightLS1'])
LCQMC_train



# prepare dev data
LCQMC_dev = pd.read_csv('./data/lcqmc/dev.tsv', sep='\t', header=None)
LCQMC_dev.rename(columns = {0:'query', 1:'title', 2:'label'}, inplace = True)
LCQMC_dev['query'] = [q if q[-1] not in P_LIST else q[:-1] for q in LCQMC_dev['query']] #remove punctuation
LCQMC_dev['title'] = [q if q[-1] not in P_LIST else q[:-1] for q in LCQMC_dev['title']] #remove punctuation
LCQMC_dev['sentence'] = LCQMC_dev['query'] +"[SEP]" + LCQMC_dev['title']
print('{:>5,} dev rows'.format(len(LCQMC_dev)))
LCQMC_dev



# prepare test data
LCQMC_testB_dic = {'id':[], 'query':[], 'title':[], 'text_q_seg':[],'text_t_seg':[] } 
with open(r"./data/sim_interpretation_B.txt", 'r') as f:
    for line in f:
        line_dic = json.loads(line)
        for k in line_dic.keys():
            LCQMC_testB_dic[k].append(line_dic[k]) 

LCQMC_testB = pd.DataFrame.from_dict(LCQMC_testB_dic)
LCQMC_testB['sentence'] = LCQMC_testB['query'] +"[SEP]" + LCQMC_testB['title']
LCQMC_testB


# prepare input (input_ids, attention_masks, and labels)
#training data 
train_input_ids = []
train_attention_masks = []
for sent in LCQMC_train['sentence']:
    encoded_dict = CN_tokenizer.encode_plus(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        truncation=True,
                        padding='max_length',
                        max_length = max_length,           # Pad & truncate all sentences.
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',)     # Return pytorch tensors.
    
    train_input_ids.append(encoded_dict['input_ids'])
    train_attention_masks.append(encoded_dict['attention_mask'])
# Convert the lists into tensors.
train_input_ids = torch.cat(train_input_ids, dim=0)
train_attention_masks = torch.cat(train_attention_masks, dim=0)
train_labels = torch.tensor(LCQMC_train['label'])
train_weight = torch.tensor(LCQMC_train['weightLS'], dtype =torch.float)   


#dev data 
dev_input_ids = []
dev_attention_masks = []
for sent in LCQMC_dev['sentence']:
    encoded_dict = CN_tokenizer.encode_plus(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        truncation=True,
                        padding='max_length',
                        max_length = max_length,           # Pad & truncate all sentences.
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',)     # Return pytorch tensors.
    
    dev_input_ids.append(encoded_dict['input_ids'])
    dev_attention_masks.append(encoded_dict['attention_mask'])
# Convert the lists into tensors.
dev_input_ids = torch.cat(dev_input_ids, dim=0)
dev_attention_masks = torch.cat(dev_attention_masks, dim=0)
dev_labels = torch.tensor(LCQMC_dev['label'])

train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels, train_weight)
dev_dataset = TensorDataset(dev_input_ids, dev_attention_masks, dev_labels)
print('{:>5,} training samples'.format(len(train_dataset)))
print('{:>5,} validation samples'.format(len(dev_dataset)))

# Create an iterator for dataset to save memory during training
batch_size = 64   #use batch size 64

## train_dataloader:
train_dataloader = DataLoader(
            train_dataset,
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size)

## validation_dataloader 
validation_dataloader = DataLoader(
            dev_dataset,    
            sampler = SequentialSampler(dev_dataset), # Pull out batches sequentially.
            batch_size = batch_size)



# Training one model, epoch = 2.
methods_list = ['with_EF', 'without_EF']
epochs = 2 
for method in methods_list:
    
    model = BertForSequenceClassification.from_pretrained(
        "hfl/chinese-macbert-large",
        num_labels = 2, #binary classification
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
    )

    # run this model on the GPU.
    model.to(device)
    optimizer = AdamW(model.parameters(),lr = 2e-5, eps = 1e-8)

    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

    training_stats = []
    total_t0 = time.time()

    for epoch_i in range(0, epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        t0 = time.time()
        total_train_loss = 0
        model.train()
        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            b_weight = batch[3].to(device)

            model.zero_grad()        

            result = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, 
                           labels=b_labels, return_dict=True)
            logits = result.logits
            oriloss = result.loss 

            full_loss = F.cross_entropy(logits, b_labels, reduction = 'none')
            
            if method == 'with_EF':
                loss = torch.mean(full_loss * b_weight)
            else:
                loss = torch.mean(full_loss)

            total_train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)                
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.5f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))

        ########### Validation ##########
        print("")
        print("Running Validation Whole...")
        t0 = time.time()
        model.eval()

        total_eval_accuracy , total_eval_loss, nb_eval_steps = 0, 0, 0
        for batch in validation_dataloader:        
            b_input_ids, b_input_mask, b_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)

            with torch.no_grad():        
                result = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, 
                               labels=b_labels, return_dict=True)

            loss = result.loss
            logits = result.logits            
            total_eval_loss += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            total_eval_accuracy += flat_accuracy(logits, label_ids)

        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        print("Whole Validation Accuracy: {0:.5f}".format(avg_val_accuracy))

        avg_val_loss = total_eval_loss / len(validation_dataloader)

        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)

        print("Whole Validation Loss: {0:.5f}".format(avg_val_loss))
        print("Whole Validation took: {:}".format(validation_time))

    #save models
    output_dir = f'./macbert_large_{method}/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("Saving model to %s" % output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model  
    model_to_save.save_pretrained(output_dir)
    CN_tokenizer.save_pretrained(output_dir)

    print("")
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))



test_input_ids = []
test_attention_masks = []
for sent in LCQMC_testB['sentence']:
    encoded_dict = CN_tokenizer.encode_plus(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        truncation=True,
                        padding='max_length',
                        max_length = 74,           # Pad & truncate all sentences.
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',)     # Return pytorch tensors.
    
    test_input_ids.append(encoded_dict['input_ids'])
    test_attention_masks.append(encoded_dict['attention_mask'])
# Convert the lists into tensors.
test_input_ids = torch.cat(test_input_ids, dim=0)
test_attention_masks = torch.cat(test_attention_masks, dim=0)
test_dataset = TensorDataset(test_input_ids, test_attention_masks)
print('{:>5,} test dataset samples'.format(len(test_dataset)))
## test_dataloader 
test_dataloader = DataLoader(
            test_dataset, 
            sampler = SequentialSampler(test_dataset), # Pull out batches sequentially.
            batch_size = 64)


methods_list = ['with_EF', 'without_EF']
for method in methods_list:
    output_dir = f'./macbert_large_{method}/'

    model = BertForSequenceClassification.from_pretrained(output_dir)
    model.to(device)
    print("")
    print("Running test ...")
    t0 = time.time()
    model.eval()
    logits_B_test  = []
    
    for batch in test_dataloader:        
        b_input_ids, b_input_mask = batch[0].to(device), batch[1].to(device)
        with torch.no_grad():        
            result = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, return_dict=True)

        loss = result.loss
        logits = result.logits            

        logits = logits.detach().cpu().numpy()
        logits_B_test.append(logits)
        validation_time = format_time(time.time() - t0)
        print(validation_time)
        
    if not os.path.exists(output_dir+'predictionB'):
        os.mkdir(output_dir+'predictionB')
    np.save(output_dir+'predictionB/logits_B_test.npy', np.vstack(logits_B_test)) 



