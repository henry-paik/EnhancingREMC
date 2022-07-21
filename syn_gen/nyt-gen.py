import torch
import logging
import time
import numpy as np
from torch.utils import data
from transformers import BertForMaskedLM
from transformers import AutoTokenizer
import csv
from copy import deepcopy
import datetime
import argparse
from math import ceil
import pickle
LABEL_TO_ID = {"/people/person/nationality": 2, "/time/event/locations": 22, "/people/person/children": 14, "/business/company/advisors": 19, "/business/location": 18, "/business/company/majorshareholders": 16, "/people/person/place_lived": 5, "NA": 0, "/business/company/place_founded": 11, "/location/neighborhood/neighborhood_of": 8, "/people/deceasedperson/place_of_death": 4, "/film/film/featured_film_locations": 21, "/location/region/capital": 23, "/business/company/founders": 6, "/people/ethnicity/geographic_distribution": 17, "/location/country/administrative_divisions": 12, "/people/deceasedperson/place_of_burial": 24, "/location/country/capital": 13, "/business/person/company": 9, "/location/location/contains": 1, "/location/administrative_division/country": 10, "/location/us_county/county_seat": 20, "/people/person/religion": 15, "/people/person/place_of_birth": 3, "/people/person/ethnicity": 7}

ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}
MINOR_LABEL_ID = [20, 17, 24, 21, 19, 22]
MINOR_LABELS =  [ID_TO_LABEL[i] for i in MINOR_LABEL_ID] # 6MCs

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

def _get_rep_id(masked_stack, mask_idx, model, ori_ids, top_cand_n=10, tau=1.5, is_must=False):
    global filter_list, batch_size
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

def gen_new_inputids(new_list, pass_idx, model, n_sub, batch_size=300, tau=2.5):
    """
    generate 1 new sentence
    - pass_idx = obj, sub idx to skip
    Return
    ---
    input_ids: numpy
    """
    global filter_list, mask_token_id
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
    
    if len(candidate) == 0:
        return None, n_sub, len(candidate)
    
    track_replace = torch.zeros((batch_size,1), dtype=int)
    stacked_ids = torch.tensor(new_list).repeat(batch_size,1)
    if n_sub < 1:
        n_sub *= len(candidate)
        n_sub = ceil(n_sub)
    if sum(np.array(p_ls)) < n_sub:
        raise
    must_replace_idx = sorted(np.random.choice(np.array(candidate)[p_ls], size=n_sub, replace=False))
    
    for mask_idx in candidate:
        is_must = mask_idx in must_replace_idx
        ori_ids = stacked_ids[:, [mask_idx]]
        mask_idx = torch.tensor(mask_idx).repeat(batch_size, 1)
        masked_stack = stacked_ids.scatter(1, mask_idx, mask_token_id)
        rep_ids = _get_rep_id(masked_stack, mask_idx, model, ori_ids, top_cand_n=5, tau=tau, is_must=is_must)
        rep_ids = rep_ids[:, None]
        track_replace += (ori_ids != rep_ids)
        stacked_ids = stacked_ids.scatter(1, mask_idx, rep_ids)

    if sum(track_replace>=n_sub) > 0:
        stacked_ids = torch.masked_select(stacked_ids, track_replace>=n_sub).view(sum(track_replace>=n_sub), -1)
        return stacked_ids, n_sub, len(candidate)
    else:
        return [], n_sub, len(candidate)

def write_csv(line_idx, new_sent_set, old_sub_idx, old_obj_idx, label_id, ori_in_ids, must_n_sub, cand_len, w_fn):    
    sub_len = []
    with open('./source/' + w_fn + '.csv', 'a+', encoding='utf-8-sig') as f:
        writer_ = csv.writer(f, delimiter='\t')
        for n_s in new_sent_set:
            _val = sum(ori_in_ids != n_s)
            sub_len.append(_val.item())
            if _val < must_n_sub:
                raise
            logger.info(tokenizer.decode(n_s))
            writer_.writerow((line_idx, n_s.tolist(), old_sub_idx, old_obj_idx, label_id))

    with open('./source/' + w_fn + '_meta.csv', 'a+', encoding='utf-8-sig') as f:
        writer_ = csv.writer(f, delimiter='\t')
        writer_.writerow((line_idx, len(ori_in_ids), must_n_sub, cand_len, sub_len))
            
if __name__ == "__main__":
    datestr = datetime.datetime.strftime(datetime.datetime.now(), '%y-%m-%d')
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default='./source/', help='path for saving predictiction and answer numpy array')
    parser.add_argument("--train-data", type=str, default='./data/nyt-10m-6MCgen.pkl', help='original train data')
    parser.add_argument("--gen-per-sent", type=int, default=300, help='how many data you will generate per sentence')
    parser.add_argument("--tokenizer-path", type=str, default='bert-base', help='base tokenizer for generation')
    parser.add_argument("--save-name", type=str, default='nyt10m-gen300-bert-base-increment-p1ok', help='file name for saving result')
    parser.add_argument("--max_len", type=int, default=256, help='maximum length of source sentence')
    parser.add_argument("--device", type=str, default='cuda', help='project name to save file')
    parser.add_argument("--file-seed", type=int, default=1, help='run subset of files')
    parser.add_argument("--n-sub", type=float, default=0.2, help='minimum number of substitution words compared to the original sentence')
    parser.add_argument("--bs", type=int, default=30, help='batch size')
    parser.add_argument("--tau", type=float, default=1.5, help='minimum number of substitution words compared to the original sentence')
    parser.add_argument("--max-trial", type=int, default=300, help='max trial number to meet predefined number of sentence')
    parser.add_argument("--model-path", type=str, required=True, help='mlm model path')
    args = parser.parse_args()
    
    args.save_name = args.save_name + 'sub{}_gen{}_tau{}_fileseed_{}'.format(str(args.n_sub).replace('.', ''), args.gen_per_sent, str(args.tau).replace('.', ''), args.file_seed)
    logger = get_logger(args.save_name)
    logger.info(args)
    device = args.device
    batch_size = args.bs
    n_sub = args.n_sub
    model = BertForMaskedLM.from_pretrained(args.model_path)
    model = model.to(device)
    
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    special_tokens = ['[unused0]','[unused1]','[unused2]','[unused3]','[unused4]','[unused5]']
    num_added_tokens = tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    filter_list = tokenizer.all_special_ids # ['<s>', '<pad>', '</s>', '<unk>'] [0, 2, 3, 1, 50264, 50261, 50262]    
    mask_token_id = tokenizer.mask_token_id

    with open(args.train_data, 'rb') as f:
        data_ls = pickle.load(f)
    logger.info(LABEL_TO_ID)
    logger.info("# data {}".format(len(data_ls)))
    for cnt, input_data in enumerate(data_ls): # max 814; seed 1~6
        if (args.file_seed-1)*100 <= cnt < args.file_seed*100:
            ss = input_data['ss']
            se = input_data['se']
            os = input_data['os']
            oe = input_data['oe']
            in_ids = input_data['input_ids'][0].tolist()
            sub_idx = list(range(ss, se))
            obj_idx = list(range(os, oe))
            logger.info('subj {}'.format(tokenizer.decode([in_ids[i] for i in sub_idx])))
            logger.info('obj {}'.format(tokenizer.decode([in_ids[i] for i in obj_idx])))

            new_ids_set = torch.LongTensor()
            n_trial = 0
            while True:
                n_trial += 1
                new_input_ids_copy = deepcopy(in_ids)
                
                ########################## gen new sent
                generated_input_ids, must_n_sub, cand_len = gen_new_inputids(new_input_ids_copy, pass_idx=sub_idx+obj_idx, batch_size=batch_size, model=model, n_sub=n_sub) # including <s>, </s>
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

            logger.info("new_ids_set {} - must {} - cand len {}".format(len(new_ids_set), must_n_sub, cand_len))
            if generated_input_ids is None:
                continue
            write_csv(input_data['ori_idx'], new_ids_set, sub_idx, obj_idx, input_data['labels'], torch.tensor(in_ids), must_n_sub, cand_len, w_fn=args.save_name)