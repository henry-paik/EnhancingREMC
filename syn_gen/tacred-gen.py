import torch
import logging
import time
import numpy as np
from transformers import RobertaTokenizer, RobertaForMaskedLM 
import csv
import json
from copy import deepcopy
import datetime
import argparse
from collections import defaultdict
from math import ceil

LABEL_TO_ID = {'org:founded_by': 0,
 'no_relation': 1,
 'per:employee_of': 2,
 'org:alternate_names': 3,
 'per:cities_of_residence': 4,
 'per:children': 5,
 'per:title': 6,
 'per:siblings': 7,
 'per:religion': 8,
 'per:age': 9,
 'org:website': 10,
 'per:stateorprovinces_of_residence': 11,
 'org:member_of': 12,
 'org:top_members/employees': 13,
 'per:countries_of_residence': 14,
 'org:city_of_headquarters': 15,
 'org:members': 16,
 'org:country_of_headquarters': 17,
 'per:spouse': 18,
 'org:stateorprovince_of_headquarters': 19,
 'org:number_of_employees/members': 20,
 'org:parents': 21,
 'org:subsidiaries': 22,
 'per:origin': 23,
 'org:political/religious_affiliation': 24,
 'per:other_family': 25,
 'per:stateorprovince_of_birth': 26,
 'org:dissolved': 27,
 'per:date_of_death': 28,
 'org:shareholders': 29,
 'per:alternate_names': 30,
 'per:parents': 31,
 'per:schools_attended': 32,
 'per:cause_of_death': 33,
 'per:city_of_death': 34,
 'per:stateorprovince_of_death': 35,
 'org:founded': 36,
 'per:country_of_birth': 37,
 'per:date_of_birth': 38,
 'per:city_of_birth': 39,
 'per:charges': 40,
 'per:country_of_death': 41}

MINOR_LABELS =  ['per:country_of_death',
  'org:member_of',  
  'org:dissolved',      
  'org:shareholders',]


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

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    
    file_handler = logging.FileHandler(file_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # wrap
    logger.handlers.clear()
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger

def prepare_mc_data(PATH):
    with open(PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    MC_data_for_gen = []
    for j, d in enumerate(data):
        if LABEL_TO_ID[d['relation']] in MINOR_LABELS:
            temp = {}
            temp['line_idx'] = j
            temp['h'] = {'pos': [d['subj_start'], d['subj_end']], 'name': ' ' .join([d['token'][i] for i in range(d['subj_start'], d['subj_end']+1)])}
            temp['t'] = {'pos': [d['obj_start'], d['obj_end']], 'name': ' ' .join([d['token'][i] for i in range(d['obj_start'], d['obj_end']+1)])}
            temp['text'] = ' '.join(d['token'])
            temp['relation'] = d['relation']
            temp['subj_type'] = d['subj_type']
            temp['obj_type'] = d['obj_type']
            MC_data_for_gen.append(temp)
    del data
    return MC_data_for_gen

def _pre_processing1(token):
    if (token.lower() == '-lrb-'):
        return '@lrb@'
    elif (token.lower() == '-rrb-'):
        return '@rrb@'
    elif (token.lower() == '-lsb-'):
        return '@lsb@'
    elif (token.lower() == '-rsb-'):
        return '@rsb@'
    elif (token.lower() == '-lcb-'):
        return '@lcb@'
    elif (token.lower() == '-rcb-'):
        return '@rcb@'
    return token

def _pre_processing2(token):
    if (token.lower() == '@lrb@'):
        return '('
    elif (token.lower() == '@rrb@'):
        return ')'
    elif (token.lower() == '@lsb@'):
        return '['
    elif (token.lower() == '@rsb@'):
        return ']'
    elif (token.lower() == '@lcb@'):
        return '{'
    elif (token.lower() == '@rcb@'):
        return '}'
    return token

def _post_processing(jsonl, head, tail):
    """
    input:
        {'id': 300, 'h': [47], 't': [45], 'token': ['Social', 'engineering', 'on', 'television', 'is', 'a', ..., ], 'relation': ...}
    return:
        {"id": "e779865fb938ff85a23e", 
        "relation": "no_relation", 
        "token": ["Spain", "'s", "government", "asked", "Venezuela", "to", "extradite", "a", "Basque", "separatist", "suspected", "of", "belonging", "to", "the", "violent", "group", "ETA", ",", "the", "deputy", "prime", "minister", "said", "Friday", "."], 
        "subj_start": 17, 
        "subj_end": 17, 
        "obj_start": 24, 
        "obj_end": 24, 
        "subj_type": "ORGANIZATION", 
        "obj_type": "DATE"}
    """
    subj_start, subj_end = jsonl['h'][0], jsonl['h'][-1]
    obj_start, obj_end = jsonl['t'][0], jsonl['t'][-1]

    jsonl['subj_start'] = subj_start
    jsonl['subj_end'] = subj_end
    jsonl['obj_start'] = obj_start
    jsonl['obj_end'] = obj_end
   
    reconstruct_h = ' '.join([jsonl['token'][i] for i in range(subj_start, subj_end+1)])
    reconstruct_t = ' '.join([jsonl['token'][i] for i in range(obj_start, obj_end+1)])

    # check head
    try:
        print(jsonl)
        print(reconstruct_h, head)
        assert reconstruct_h == head
    except Exception as e:
        logger.error(e)
        logger.error(jsonl['id'])
        if reconstruct_h.find(head) < 0:
            raise
            
    # check tail
    try:
        assert reconstruct_t == tail
    except:
        logger.error(e)
        logger.error(jsonl['id'])
        if reconstruct_t.find(tail) < 0:
            raise
    return jsonl
    
def update_jsonl(data_dict, idx):
    """
    - input datadict
    {'line_idx': 101, 'h': {'pos': [25, 27], 'name': 'Noordhoff Craniofacial Foundation'}, 't': {'pos': [5, 5], 'name': 'TOAID'}, 'text': 'The ... Corps .', 'relation': 'org:member_of', 'subj_type': 'ORGANIZATION', 'obj_type': 'ORGANIZATION'}
    - right before return
    {'id': 83208, 'h': [21], 't': [11, 12], 
    'relation': '/business/company/advisors', 
    'token': [..., 'year', 'of', 'his', 'birth', '.']}

    """
    hs, he = data_dict['h']['pos']
    head = data_dict['h']['name']
    ts, te = data_dict['t']['pos']
    tail = data_dict['t']['name']
    
    temp = {'id': idx, 'h': data_dict['h']['pos'], 't':data_dict['t']['pos'], 'relation':data_dict['relation']}
    
    data_dict['text'] = ' '.join([_pre_processing1(t) for t in data_dict['text'].split()])
    temp['token'] = [_pre_processing2(t) for t in data_dict['text'].split()]
    return _post_processing(temp, head, tail)

def _get_rep_id(masked_stack, mask_idx, model, ori_ids, top_cand_n=10, tau=1.5, is_must=False, device='cuda', batch_size=20):
    global filter_list
    with torch.no_grad():
        token_logits = model(masked_stack.to(device))[0] # (batch_size, seq_len, vocab_size)
        mask_token_logits = token_logits[torch.arange(batch_size)[:, None], mask_idx, :] # (batch_size, 1, vocab_size)
        mask_token_logits = torch.softmax(mask_token_logits.squeeze(), dim=1)
        
        top_n = torch.topk(mask_token_logits, top_cand_n, dim=1) # top_n.values or indices : (batch_size, top_cand_n)
        val = np.array(top_n.values.cpu())
        cand = top_n.indices.cpu()
        p = val**(1/tau) / np.sum((val)**(1/tau), axis=1)[:, None]
        selected = list()
        for i in range(batch_size):
            while True:
                sel_elem = np.random.choice(cand[i], p=p[i])
                if is_must:
                    if (sel_elem not in filter_list) and (sel_elem != ori_ids[i]):
                        break
                    if (sel_elem not in filter_list) and (np.max(p[i])==1):
                        break
                    
                else:
                    if sel_elem not in filter_list:
                        break
            selected.append(sel_elem)
    return torch.tensor(selected)

def gen_new_inputids(new_list, pass_idx, model, batch_size, n_sub, tau=1.5, device='cuda'):
    """
    generate 1 new sentence
    - pass_idx = obj, sub idx to skip
    Return
    ---
    input_ids: numpy
    """
    global filter_list, mask_token_id
    track_replace = torch.zeros((batch_size,1), dtype=int)
    stacked_ids = torch.tensor(new_list).repeat(batch_size,1)

    candidate = [idx for idx, t in enumerate(new_list) if (t not in filter_list) and (idx not in pass_idx)]
    temp_copy = deepcopy(new_list)
    p_ls = []
    for mask_idx in candidate:
        ori_ids = temp_copy[mask_idx]
        temp_copy[mask_idx] = mask_token_id
        temp_in = torch.tensor(temp_copy).unsqueeze(0)
        with torch.no_grad():
            token_logits = model(temp_in.to(device))[0] # (batch_size, seq_len, vocab_size)
            
            mask_token_logits = token_logits[0, mask_idx, :] # (batch_size, 1, vocab_size)
            mask_token_logits = torch.softmax(mask_token_logits, dim=0)
            p_ls.append((torch.max(mask_token_logits) != 1).item())
        temp_copy[mask_idx] = ori_ids
    
    # cannot substitute 
    if len(candidate) == 0:
        print(0)
        return None, n_sub, len(candidate)

    if n_sub < 1:
        n_sub *= len(candidate)
        n_sub = ceil(n_sub)
    if sum(np.array(p_ls)) < n_sub:
        raise
    
    # randomly select index
    must_replace_idx = sorted(np.random.choice(np.array(candidate)[p_ls], size=n_sub, replace=False))
    for mask_idx in candidate:
        #print(mask_idx)
        is_must = mask_idx in must_replace_idx
        ori_ids = stacked_ids[:, [mask_idx]]
        mask_idx = torch.tensor(mask_idx).repeat(batch_size, 1)
        masked_stack = stacked_ids.scatter(1, mask_idx, mask_token_id)
        rep_ids = _get_rep_id(masked_stack, mask_idx, model, ori_ids, top_cand_n=5, tau=tau, is_must=is_must, device=device, batch_size=batch_size)
        rep_ids = rep_ids[:, None]
        track_replace += (ori_ids != rep_ids)
        stacked_ids = stacked_ids.scatter(1, mask_idx, rep_ids)

    if sum(track_replace>=n_sub) > 0:
        stacked_ids = torch.masked_select(stacked_ids, track_replace>=n_sub).view(sum(track_replace>=n_sub), -1)
        return stacked_ids, n_sub, len(candidate)
    else:
        return [], n_sub, len(candidate)

def tracking_ent_idx(input_token, ss, se, os, oe, ori_subj, ori_obj):
    sub_idx = list(range(ss, se+1))
    obj_idx = list(range(os, oe+1))
    new_input_ids = []
    new_subidx = []; new_objidx = []
    recon_sub = []; recon_obj = []
    j = 0 # inserting <s>,</s> when post processing
    for i, token in enumerate(input_token):
        if i != 0:
            token = ' ' + token
        for t in tokenizer(token, add_special_tokens=False)['input_ids']:
#             print(t)
            j += 1
            if i in sub_idx:
                new_subidx.append(j)
                recon_sub.append(t)
            if i in obj_idx:
                new_objidx.append(j)
                recon_obj.append(t)
            new_input_ids.append(t)
    new_input_ids.insert(0, tokenizer.bos_token_id)
    new_input_ids.append(tokenizer.eos_token_id)
    assert tokenizer.decode(recon_sub).replace(' ','') == ori_subj.replace(' ','')
    assert tokenizer.decode(recon_obj).replace(' ','') == ori_obj.replace(' ','')
    return new_input_ids, new_subidx, new_objidx

def write_csv(line_idx, new_sent_set, old_sub_idx, old_obj_idx, label_id, ori_in_ids, must_n_sub, cand_len, w_fn):    
    sub_len = []
    with open('./source/' + w_fn + '.csv', 'a+', encoding='utf-8-sig') as f:
        writer_ = csv.writer(f, delimiter='\t')
        for n_s in new_sent_set:
            _val = sum(ori_in_ids != n_s)
            sub_len.append(_val.item())
            print(must_n_sub, _val, cand_len)
            if _val < must_n_sub:
                raise
            logger.info(tokenizer.decode(n_s))
            writer_.writerow((line_idx, n_s.tolist(), old_sub_idx, old_obj_idx, label_id))

    with open('./source/' + w_fn + '_meta.csv', 'a+', encoding='utf-8-sig') as f:
        writer_ = csv.writer(f, delimiter='\t')
        writer_.writerow((line_idx, must_n_sub, cand_len, sub_len))

if __name__ == "__main__":
    datestr = datetime.datetime.strftime(datetime.datetime.now(), '%y-%m-%d')
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default='./source/', help='path for saving predictiction and answer numpy array')
    parser.add_argument("--train-data", type=str, default='./../tacred/data/train.json', help='original train data')
    parser.add_argument("--gen-per-sent", type=int, default=300, help='how many data you will generate per sentence')
    parser.add_argument("--bs", type=int, default=20, help='batch size for inference')
    parser.add_argument("--tokenizer-path", type=str, default='roberta-base', help='base tokenizer for generation')
    parser.add_argument("--n-sub", type=float, default=0.3, help='minimum number of substitution words compared to the original sentence')
    parser.add_argument("--tau", type=float, default=1.5, help='minimum number of substitution words compared to the original sentence')
    parser.add_argument("--save-name", type=str, default='tacred-gen-roberta-base', help='file name for saving result')
    parser.add_argument("--max_len", type=int, default=370, help='maximum length of source sentence')
    parser.add_argument("--file-seed", type=int, required=True, help='run subset of files')
    parser.add_argument("--split-size", type=int, default=60, help='# subset of target data')
    parser.add_argument("--device", type=str, default='cuda', help='device option')
    parser.add_argument("--max-trial", type=int, default=300, help='max trial number to meet predefined number of sentence')
    parser.add_argument("--model-path", type=str, required=True, help='mlm model path')

    args = parser.parse_args()
    
    args.save_name = args.save_name + '_sub{}_gen{}_tau{}_fileseed{}'.format(str(args.n_sub).replace('.', ''), args.gen_per_sent, str(args.tau).replace('.', ''), args.file_seed)
    logger = get_logger(args.save_name)
    logger.info(args)
    model = RobertaForMaskedLM.from_pretrained(args.model_path)
    model = model.to(args.device)
    model.cuda().half().eval()
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_path)
    special_tokens = ['madeupword0000', 'madeupword0001'] # for generation
    num_added_tokens = tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    filter_list = tokenizer.all_special_ids # ['<s>', '<pad>', '</s>', '<unk>'] [0, 2, 3, 1, 50264, 50261, 50262]    
    mask_token_id = tokenizer.mask_token_id
    
    # load MC data
    data_stat = defaultdict(int)
    data_ls = []; iter_order  = []; j = -1
    mc_data = prepare_mc_data(args.train_data)
    for line_idx, l in enumerate(mc_data):
        token_len = len(tokenizer(l['text'])['input_ids'])
        if token_len <= args.max_len:
            j += 1
            data_stat[LABEL_TO_ID[l['relation']]] += 1
            data_ls.append(l)
            iter_order.append((token_len, j))
            logger.info("valid line {}".format(line_idx))
    iter_order = sorted(iter_order, key=lambda x: x[0], reverse=True)
    assert j+1 == len(data_ls)
    assert len(iter_order) == len(data_ls)
    logger.info(LABEL_TO_ID)
    logger.info(data_stat)
    logger.info("# data {}".format(len(data_ls)))
    cnt = 0
    for _, data_idx in iter_order: # TACRED MC total 227 sent; file seed range 1~6
        cnt += 1
        if args.split_size*(args.file_seed-1) <= cnt < args.split_size*(args.file_seed):
            input_data = data_ls[data_idx]
            logger.info('job start {}'.format(input_data))
            ori_subj, ori_obj = input_data['h']['name'], input_data['t']['name']
            input_data = update_jsonl(input_data, input_data['line_idx'])
            ss, se = input_data['subj_start'], input_data['subj_end']
            os, oe = input_data['obj_start'], input_data['obj_end']
            if len(set(range(ss,se+1)) & set(range(os, oe+1))) > 0:
                logger.info('invalid MC {}'.format(input_data))
                continue
            new_input_ids, old_sub_idx, old_obj_idx = tracking_ent_idx(input_data['token'], ss, se, os, oe, ori_subj, ori_obj)
            new_ids_set = torch.LongTensor()
            n_trial = 0
            while True:
                n_trial += 1
                print('trial', n_trial)
                new_input_ids_copy = deepcopy(new_input_ids)
                
                ########################## gen new sent
                generated_input_ids, must_n_sub, cand_len = gen_new_inputids(new_input_ids_copy, pass_idx=old_sub_idx+old_obj_idx, batch_size=args.bs, model=model, n_sub=args.n_sub, device=args.device) # including <s>, </s>

                if generated_input_ids is None:
                    break

                if len(generated_input_ids) > 1:
                    new_ids_set = torch.cat([new_ids_set, generated_input_ids])
                    new_ids_set = torch.unique(new_ids_set, dim=0)

                if len(new_ids_set) >= args.gen_per_sent:
                    new_ids_set = new_ids_set[:args.gen_per_sent]
                    break
                
                if n_trial >= args.max_trial:
                    break

            logger.info("{} - {} - {}".format(len(new_ids_set), must_n_sub, cand_len))
            if generated_input_ids is None:
                continue

            write_csv(input_data['id'], new_ids_set, old_sub_idx, old_obj_idx, LABEL_TO_ID[input_data['relation']], torch.tensor(new_input_ids), must_n_sub, cand_len, w_fn=args.save_name)
            
