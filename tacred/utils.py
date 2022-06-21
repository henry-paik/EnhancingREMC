import ast
import linecache
from torch.utils import data
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
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from data_util.meta_data import LABEL_TO_ID
import numpy as np

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

def get_metric_cre(pred: np.array, challenge_set_path: str, logger):
    """
    - CRE paper: https://aclanthology.org/2020.emnlp-main.302.pdf
    - original code from: https://github.com/shacharosn/CRE
    """
    logger.info("===============CRE===============")
    
    NO_RELATION = "no_relation"
    label_map_reverse = {v: k for k, v in LABEL_TO_ID.items()}
    pred = list(map(lambda x: label_map_reverse[x], pred))

    ###### utils for CRE
    def compute_f1(preds, labels):
        """Compute precision, recall and f1 as a row data """

        n_gold = n_pred = n_correct = 0
        for p_, label in zip(preds, labels):
            if p_ != NO_RELATION:
                n_pred += 1
            if label != NO_RELATION:
                n_gold += 1
            if (p_ != NO_RELATION) and (label != NO_RELATION) and (p_ == label):
                n_correct += 1
        if n_correct == 0:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        else:
            prec = n_correct * 1.0 / n_pred
            recall = n_correct * 1.0 / n_gold
            if prec + recall > 0:
                f1 = 2.0 * prec * recall / (prec + recall)
            else:
                f1 = 0.0
            return {'precision': prec, 'recall': recall, 'f1': f1}
    ###### end of utils for CRE
    
    with open(challenge_set_path, "r") as tacred_test_file:
        gold_data = json.load(tacred_test_file)
        
    ids = [l['id'] for l in gold_data]    
    pred_data = {}
    for p, i_ in zip(pred, ids):
        pred_data[i_] = p

    true_y = []
    pred_y = []

    for row in gold_data:

        true_y.append(row['gold_relation'])

        if pred_data[row['id']] == row['id_relation']:

            pred_y.append(row['id_relation'])
        else:
            pred_y.append(NO_RELATION)


    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    for row, pp, tt in zip(gold_data, pred_y, true_y):

        curr_id_relation = row["id_relation"]

        true_positive += 1  if pp == curr_id_relation and tt == curr_id_relation else 0
        false_positive += 1 if pp == curr_id_relation and tt != curr_id_relation else 0
        true_negative += 1  if pp != curr_id_relation and tt != curr_id_relation else 0
        false_negative += 1 if pp != curr_id_relation and tt == curr_id_relation else 0

    acc = accuracy_score(true_y, pred_y)
    logger.info("ACCURACY:   {:.2%}".format(acc))

    total_accuracy_score_positive = true_positive / (true_positive + false_negative)
    total_accuracy_score_negative = true_negative / (false_positive + true_negative)
    logger.info("POSITIVE ACCURACY:   {:.2%}".format(total_accuracy_score_positive))
    logger.info("NEGATIVE ACCURACY:   {:.2%}".format(total_accuracy_score_negative))

    number_of_examples = len(true_y)
    logger.info("TRUE POSITIVE:   {:.3f} \t\t (NUMBER:   {})".format(true_positive / number_of_examples, true_positive))
    logger.info("FALSE POSITIVE:   {:.3f} \t\t (NUMBER:   {})".format(false_positive / number_of_examples, false_positive))
    logger.info("TRUE NEGATIVE:   {:.3f} \t\t (NUMBER:   {})".format(true_negative / number_of_examples, true_negative))
    logger.info("FALSE NEGATIVE:   {:.3f} \t\t (NUMBER:   {})".format(false_negative / number_of_examples, false_negative))

    f1 = compute_f1(pred_y, true_y)

    logger.info("Precision: {:.2%}\t Recall: {:.2%}\t  F1: {:.2%}\n".format(f1["precision"], f1["recall"], f1["f1"]))

def get_metric_rev(pred: np.array, rev_label: np.array, sample_weight: np.array, logger, mcs=[12, 27, 29, 41]):
    """
    Calculate test scores on TACRED-revisited
    """
    logger.info("===============TACRED-revisited===============")
    data_dict = {'p':0, 'r':0, 'f1':0, 'f1mac':0, '4ma':0, '4mi':0, 'wf1mac': None, 'wf1':None, 'wf1mc': None}
    # conf_res_df = cal_ft_fn(pred, rev_label)

    s = sample_weight
    s2 = np.isin(rev_label, mcs)

    # overall f1
    labels = list(range(42))
    labels.pop(1)
    report = classification_report(rev_label, pred, labels=labels)
    logger.info(report)
    
    prec, rec, f1, _ = precision_recall_fscore_support(rev_label, pred, labels=labels, average='macro')
    data_dict['f1mac'] = f1
    pure_prec, pure_rec, pure_f1, _ = precision_recall_fscore_support(rev_label, pred, labels=labels, average='micro')
    data_dict['p'], data_dict['r'], data_dict['f1'] = pure_prec, pure_rec, pure_f1

    # weighted sample
    if s is not None:
        prec, rec, f1, _ = precision_recall_fscore_support(rev_label, pred, labels=labels, average='macro', sample_weight=s)
        data_dict['wf1mac'] = f1
        prec, rec, f1, _ = precision_recall_fscore_support(rev_label, pred, labels=labels, average='micro', sample_weight=s)
        data_dict['wf1'] = f1
        
    if s2 is not None:
        prec, rec, f1, _ = precision_recall_fscore_support(rev_label, pred, labels=labels, average='micro', sample_weight=s2)
        data_dict['wf1mc'] = f1

    # MC f1
    prec, rec, f1, _ = precision_recall_fscore_support(rev_label, pred, labels=mcs, average='macro')
    data_dict['4ma'] = f1
    prec, rec, f1, _ = precision_recall_fscore_support(rev_label, pred, labels=mcs, average='micro')
    data_dict['4mi'] = f1
    logger.info(data_dict)

def get_metric_ori(pred: np.array, labels: np.array, ref_labels_id, logger, class_labels=[0]+list(range(2, 42))):
    
    logger.info("===============original TACRED===============")
    report = classification_report(labels, pred)
    logger.info(report)

    p_, r_, f1_, _ = precision_recall_fscore_support(labels, pred, labels=class_labels, average='micro')
    logger.info('precision: {} - recall: {} - f1: {}'.format(p_, r_, f1_))
    
    p_, r_, f1_, _ = precision_recall_fscore_support(labels, pred, labels=ref_labels_id, average='micro')
    logger.info('MC precision: {} - MC recall: {} - MC f1: {}'.format(p_, r_, f1_))
    
def calc_aug_data_num(num_instance={12:122, 27:23, 29:76, 41:6}, factor=4, selected_r=0.5):
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

def calc_aug_data_num_permc(selected_r, num_instance={12:122, 27:23, 29:76, 41:6}, factor=4):
    """
    args
    - num_instance: dict for MC and # training instances
    - factor: how many times generate data for each MC
    - ratio of reliable instances
    """
    samples = np.array([num_instance[k] for k in sorted(num_instance.keys())])
    target = factor*sum(samples)
    # samples = (samples * selected_r).astype(int)
    samples = (samples*np.array(selected_r)).astype(int)
    samples[samples<1] = 1 # secure minimum number of reliable instance

    target_num_dict = {k: v for k, v in zip(sorted(num_instance.keys()), samples)}
    aug_per_sent = np.round((target/len(num_instance)/samples)).astype(int)
    aug_num_dict = {k: v for k, v in zip(sorted(num_instance.keys()), aug_per_sent)}
    return aug_num_dict, target_num_dict


def calc_aug_data_num_permc_n(selected_n, num_instance={12:122, 27:23, 29:76, 41:6}, factor=4):
    """
    args
    - num_instance: dict for MC and # training instances
    - factor: how many times generate data for each MC
    - ratio of reliable instances
    """
    # samples = np.array([num_instance[k] for k in sorted(num_instance.keys())])
    # target = factor*sum(samples)
    # # samples = (samples * selected_r).astype(int)
    # samples = (samples*np.array(selected_r)).astype(int)
    target = factor*sum([num_instance[k] for k in sorted(num_instance.keys())])
    samples = np.array(selected_n)
    samples[samples<1] = 1 # secure minimum number of reliable instance

    target_num_dict = {k: v for k, v in zip(sorted(num_instance.keys()), samples)}
    aug_per_sent = np.round((target/len(num_instance)/samples)).astype(int)
    aug_num_dict = {k: v for k, v in zip(sorted(num_instance.keys()), aug_per_sent)}
    return aug_num_dict, target_num_dict


def cal_ft_fn(pred, ans, labels=list(range(42))):

    def _cal_conf(cnf_matrix, i=0):
        # i means which class to choose to do one-vs-the-rest calculation
        # rows are actual obs whereas columns are predictions
        TP = cnf_matrix[i,i]
        FP = cnf_matrix[:,i].sum() - TP # predicted - TP: incorrectly labeled as i
        FN = cnf_matrix[i,:].sum() - TP # actual - TP: incorrectly labeled as non-i
        TN = cnf_matrix.sum().sum() - TP - FP - FN
        return {'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN}

    res = {i: None for i in labels}
    cnf_matrix = confusion_matrix(ans, pred, labels=labels)
    for i in labels:
        res[i] = _cal_conf(cnf_matrix, i)
    return pd.DataFrame(res)


def cal_ft_fn(pred, ans, labels=list(range(42))):

    def _cal_conf(cnf_matrix, i=0):
        # i means which class to choose to do one-vs-the-rest calculation
        # rows are actual obs whereas columns are predictions
        TP = cnf_matrix[i,i]
        FP = cnf_matrix[:,i].sum() - TP # predicted - TP: incorrectly labeled as i
        FN = cnf_matrix[i,:].sum() - TP # actual - TP: incorrectly labeled as non-i
        TN = cnf_matrix.sum().sum() - TP - FP - FN
        return {'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN}

    res = {i: None for i in labels}
    cnf_matrix = confusion_matrix(ans, pred, labels=labels)
    for i in labels:
        res[i] = _cal_conf(cnf_matrix, i)
    return pd.DataFrame(res)