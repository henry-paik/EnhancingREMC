import ast
import linecache
from torch.utils.data import Dataset, DataLoader
import subprocess
import torch
from transformers import RobertaModel, RobertaTokenizer, AdamW, get_linear_schedule_with_warmup, RobertaForMaskedLM
from torch.distributions import Categorical
import pickle 
import torch.nn.functional as F
import logging
import time
import torch.nn as nn
from itertools import islice, chain
import re
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import json
import random
import traceback
import pandas as pd
import sklearn
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from bert_data_util.meta_data import LABEL_TO_ID, MINOR_LABELS, MINOR_LABEL_IDS

def set_deterministic(seed=27):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def get_ref_inputids(tokenizer, ref_sent: list):
    input_ids_ls = [tokenizer.encode(sent) for sent in ref_sent]
    max_len = max([len(ls) for ls in input_ids_ls])
    #input_ids = [ls + [1.0]*(max_len-len(ls)) for ls in input_ids_ls]
    input_ids = [ls + [0]*(max_len-len(ls)) for ls in input_ids_ls]
    input_mask = [ [1.0]*len(ls) + [0]*(max_len-len(ls)) for ls in input_ids_ls]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    return input_ids, input_mask

def get_logger(name):
    # init
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter\
        ('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter(
        fmt='%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s', datefmt='%Y-%m-%d %H:%M:%S'
    )

    # handler: file
    timestr = time.strftime('%Y%m%d_%H:%M:%S')
    file_name = 'log/{}_{}.log'.format(name, timestr)

    file_handler = logging.FileHandler(file_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # wrap
    logger.handlers.clear()
    logger.addHandler(file_handler)
    return logger

def soft_label_loss(pred, gold, eps=0.3, pad_mask=-1):
    n_class = pred.size(1)
    one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
    one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
    log_prb = F.log_softmax(pred, dim=1)
    non_pad_mask = gold.ne(pad_mask)
    loss = -(one_hot * log_prb).sum(dim=1)
    loss = loss.masked_select(non_pad_mask).sum()  # average later
    return loss/len(pred)

def soft_label_loss_mc(pred, gold, mc_idx, eps=0.3, pad_mask=-1, w=30):
    w = w/(w+1)
    n_class = pred.size(1)
    one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
    one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
    log_prb = F.log_softmax(pred, dim=1)
    # non_pad_mask = gold.ne(pad_mask)
    loss = -(one_hot * log_prb).sum(dim=1) # (1, bs)
    loss = loss.masked_select(mc_idx.to('cuda')).sum()*w + loss.masked_select(~mc_idx.to('cuda')).sum()*(1-w)  # average later
    return loss/len(pred)

def ref_encoding(tokenizer, token_ls, max_len=340):
    new_tok_space = [' '+e if e not in ent_ls else e for e in token_ls]
    new_list = list(chain(*tokenizer(new_tok_space, add_special_tokens=False)['input_ids']))
    new_list.insert(0, tokenizer.bos_token_id)
    new_list.append(tokenizer.eos_token_id)

    input_attn = torch.zeros(max_len)
    input_attn[:len(new_list)] = 1

    input_ids = torch.LongTensor(new_list + [tokenizer.pad_token_id]*(max_len-len(new_list)))
    input_attn = input_attn.long()
    return input_ids, input_attn

def softmax(pred):
    return np.exp(pred) / np.sum(np.exp(pred), axis=1,keepdims=True)

def get_metric_ori(pred, ori_label):
    
    data_dict = {'p':0, 'r':0, 'f1':0, 'f1mac':0, '6ma':0, '6mi':0}
    labels = list(range(25))
    labels.pop(0)
    prec, rec, f1, _ = precision_recall_fscore_support(ori_label, pred, labels=labels, average='macro')
    data_dict['f1mac'] = f1
    pure_prec, pure_rec, pure_f1, _ = precision_recall_fscore_support(ori_label, pred, labels=labels, average='micro')
    data_dict['p'], data_dict['r'], data_dict['f1'] = pure_prec, pure_rec, pure_f1
    
    # report
    labels = list(range(25))
    report = classification_report(ori_label, pred, labels=labels)
    report_dict = classification_report(ori_label, pred, labels=labels, output_dict=True)
    for i in MINOR_LABEL_IDS:
        data_dict[i] = report_dict[str(i)]['f1-score'] 
    
    for i, k in enumerate(['6ma']):
        prec, rec, f1, _ = precision_recall_fscore_support(ori_label, pred, labels=MINOR_LABEL_IDS[:i+5], average='macro')
        data_dict[k] = f1

    for i, k in enumerate(['6mi']):
        prec, rec, f1, _ = precision_recall_fscore_support(ori_label, pred, labels=MINOR_LABEL_IDS[:i+5], average='micro')
        data_dict[k] = f1
        
    df = pd.Series(data_dict)
    
    return df, report, (pure_prec, pure_rec, pure_f1)

def get_auc(pred_score, ans, threshold=None):
    
    if len(ans)<11086:
        fn = './data/nyt10m_test_annotated_ori_data.pkl'
        fn_ori = './data/nyt-10m-ori-method-bert-base-maxlen256-test.pkl'
    else:
        fn = './data/nyt10m_dev_annotated_ori_data.pkl'
        fn_ori = './data/nyt-10m-ori-method-bert-base-maxlen256-val.pkl'
        
    with open(fn, 'rb') as f:
        anno_test = pickle.load(f)
    with open(fn_ori, 'rb') as f:
        test_ori = pickle.load(f)
        test_ori = [i['ori_idx'] for i in test_ori]
    

    ans_to_sentid = {j: i for i, j in enumerate(test_ori)} # mapping idx and origianl line idx
    # Calculate AUC
    NA_ID = 0
    pred_score = softmax(pred_score)
    assert len(ans) == len(pred_score)
    print(pred_score.shape)
    # pred_score = [l[an].item() for an, l in zip(ans, pred_score)]
    # pred_score = np.array(pred_score)
    sorted_result = []
    
    total = 0
    for j, sent_id in enumerate(test_ori):
        for rel in range(len(LABEL_TO_ID)):
            if rel != NA_ID:
                sorted_result.append({'sent_id': sent_id, 'relation': rel, 'score': pred_score[j][rel]})
                if 'anno_list' in anno_test[sent_id]: 
                    if rel in anno_test[sent_id]['anno_list']:
                        total += 1
                else:
                    if rel == ans[j]:
                        total += 1
    
    sorted_result.sort(key=lambda x: x['score'], reverse=True)
    prec = []; rec = []; correct = 0
    for i, item in enumerate(sorted_result):
        if 'anno_list' in anno_test[item['sent_id']]: 
            if item['relation'] in anno_test[item['sent_id']]['anno_list']:
                correct += 1 
        else: 
            if item['relation'] == ans[ans_to_sentid[item['sent_id']]]:
                correct += 1 
        prec.append(float(correct) / float(i + 1))
        rec.append(float(correct) / float(total))
    auc = sklearn.metrics.auc(x=rec, y=prec)
    np_prec = np.array(prec)
    np_rec = np.array(rec) 
    max_micro_f1 = (2 * np_prec * np_rec / (np_prec + np_rec + 1e-20)).max()
    max_micro_f1_threshold = sorted_result[(2 * np_prec * np_rec / (np_prec + np_rec + 1e-20)).argmax()]['score']

    # Calculate F1
    # threshold = 0.19
    pred_result_vec = np.zeros((len(ans), 25), dtype=np.int)
    # pred_result_vec[pred >= 0.5] = 1
    if threshold is None:
        pred_result_vec[pred_score >= max_micro_f1_threshold] = 1
    else:
        pred_result_vec[pred_score >= threshold] = 1
    
    label_vec = []
    for sent_idx, j in zip(test_ori, ans):
        one_hot = np.zeros(25, dtype=np.int)
        if 'anno_list' in anno_test[sent_idx]:
            for _v in anno_test[sent_idx]['anno_list']:
                one_hot[_v] = 1
            label_vec.append(one_hot)
        else:
            one_hot[j] = 1
            label_vec.append(one_hot)

    label_vec = np.stack(label_vec, 0) 
    assert label_vec.shape == pred_result_vec.shape
    
    micro_p = sklearn.metrics.precision_score(label_vec, pred_result_vec, labels=list(range(1, len(LABEL_TO_ID))), average='micro')
    micro_r = sklearn.metrics.recall_score(label_vec, pred_result_vec, labels=list(range(1, len(LABEL_TO_ID))), average='micro')
    micro_f1 = sklearn.metrics.f1_score(label_vec, pred_result_vec, labels=list(range(1, len(LABEL_TO_ID))), average='micro')

    macro_p = sklearn.metrics.precision_score(label_vec, pred_result_vec, labels=list(range(1, len(LABEL_TO_ID))), average='macro')
    macro_r = sklearn.metrics.recall_score(label_vec, pred_result_vec, labels=list(range(1, len(LABEL_TO_ID))), average='macro')
    macro_f1 = sklearn.metrics.f1_score(label_vec, pred_result_vec, labels=list(range(1, len(LABEL_TO_ID))), average='macro')
    
    mc6f1 = sklearn.metrics.f1_score(label_vec, pred_result_vec, labels=MINOR_LABEL_IDS, average='micro')

    acc = (label_vec == pred_result_vec).mean()

    result = {'acc': acc, 'micro_p': micro_p, 'micro_r': micro_r, 'micro_f1': micro_f1, 'macro_p': macro_p, 'macro_r': macro_r, 'macro_f1': macro_f1, 'np_prec': np_prec[:10], 'np_rec': np_rec[:10], 'max_micro_f1': max_micro_f1, 'max_micro_f1_threshold': max_micro_f1_threshold, 'auc': auc, 'p@100': np_prec[99], 'p@200': np_prec[199], 'p@300': np_prec[299], '6mi': mc6f1}
    return result, threshold

def calc_aug_data_num(num_instance={19: 6, 20: 90, 23: 694, 24: 21}, factor=4, selected_r=0.5):
    """
    args
    - num_instance: dict for MC and # training instances
    - factor: how many times generate data for each MC
    - ratio of reliable instances
    """
    samples = np.array([num_instance[k] for k in sorted(num_instance.keys())])
    target = factor*sum(samples)
    samples = (samples * selected_r).astype(int)
    samples[samples<1] = 1 # secure minimum number of reliable instance

    target_num_dict = {k: v for k, v in zip(sorted(num_instance.keys()), samples)}
    aug_per_sent = np.round((target/len(num_instance)/samples)).astype(int)
    aug_num_dict = {k: v for k, v in zip(sorted(num_instance.keys()), aug_per_sent)}
    return aug_num_dict, target_num_dict

def calc_aug_data_num_permc(selected_r, num_instance={17: 60, 19: 6, 20: 90, 21: 16, 22: 3, 24: 21}, factor=4):
    """
    args
    - num_instance: dict for MC and # training instances
    - factor: how many times generate data for each MC
    - ratio of reliable instances
    """
    
    target = factor*sum(v for k, v in num_instance.items())
    # samples = (samples * selected_r).astype(int)
    samples = np.array(selected_r)

    target_num_dict = {k: v for k, v in zip(sorted(num_instance.keys()), samples)}
    aug_per_sent = np.round((target/len(num_instance)/samples)).astype(int)
    aug_num_dict = {k: v for k, v in zip(sorted(num_instance.keys()), aug_per_sent)}
    return aug_num_dict, target_num_dict
