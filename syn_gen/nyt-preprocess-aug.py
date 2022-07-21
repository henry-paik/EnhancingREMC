from transformers import AutoTokenizer
import time
import json
import datetime
import argparse
import csv
import numpy as np
import torch
import pickle

ID_TO_LABEL = {0: 'NA', 1: '/location/location/contains', 2: '/people/person/nationality', 3: '/people/person/place_of_birth', 4: '/people/deceasedperson/place_of_death', 5: '/people/person/place_lived', 6: '/business/company/founders', 7: '/people/person/ethnicity', 8: '/location/neighborhood/neighborhood_of', 9: '/business/person/company', 10: '/location/administrative_division/country', 11: '/business/company/place_founded', 12: '/location/country/administrative_divisions', 13: '/location/country/capital', 14: '/people/person/children', 15: '/people/person/religion', 16: '/business/company/majorshareholders', 17: '/people/ethnicity/geographic_distribution', 18: '/business/location', 19: '/business/company/advisors', 20: '/location/us_county/county_seat', 21: '/film/film/featured_film_locations', 22: '/time/event/locations', 23: '/location/region/capital', 24: '/people/deceasedperson/place_of_burial'}
LABEL_TO_ID = {v: k for k, v in ID_TO_LABEL.items()}
MAX_LEN = 256

class InsertTypeInfo:
    def __init__(self, tokenizer, w_fn, aug_path):
        self.features = list()
        self.tokenizer = tokenizer
        self.data = None
        self.ori_data = None
        self.w_fn = w_fn
        self.cnt = 0
        self.s_indicator1 = self.tokenizer.get_vocab()['[unused0]']
        self.s_indicator2 = self.tokenizer.get_vocab()['[unused1]']
        self.o_indicator1 = self.tokenizer.get_vocab()['[unused2]']
        self.o_indicator2 = self.tokenizer.get_vocab()['[unused3]']
        
        self._load_data(aug_path)
        
    def _load_data(self, aug_path):
        self.data = []
        with open(aug_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f, delimiter='\t')
            for _, row in enumerate(reader):
                l = [json.loads(r.encode().decode('utf-8-sig')) for r in row]
                self.data.append(l)

    def _tokenize(self, input_data):
        d_id, input_ids, subji, obji, rel = input_data
        input_ids = np.array(input_ids[1:-1])
        
        ss, se, os, oe = subji[0], subji[-1], obji[0], obji[-1]        
            
        pos_head = [ss-1, se-1]
        pos_tail = [os-1, oe-1]

        pos_min = pos_head
        pos_max = pos_tail
        if pos_head[0] > pos_tail[0]:
            pos_min = pos_tail
            pos_max = pos_head
            rev = True
        else:
            rev = False

        sent0 = input_ids[:pos_min[0]]
        ent0 = input_ids[pos_min[0]:pos_min[1]]
        sent1 = input_ids[pos_min[1]:pos_max[0]]
        ent1 = input_ids[pos_max[0]:pos_max[1]]
        sent2 = input_ids[pos_max[1]:]

        ent0 = np.concatenate([[self.s_indicator1], ent0, [self.s_indicator2]]) if not rev else np.concatenate([[self.o_indicator1], ent0, [self.o_indicator2]])
        ent1 = np.concatenate([[self.o_indicator1], ent1, [self.o_indicator2]]) if not rev else np.concatenate([[self.s_indicator1], ent1, [self.s_indicator2]])

        re_tokens = np.concatenate([[self.tokenizer.cls_token_id], sent0, ent0, sent1, ent1, sent2, [self.tokenizer.cls_token_id]])

        pos1 = 1 + len(sent0) if not rev else 1 + len(sent0) + len(ent0) + len(sent1)
        pos2 = 1 + len(sent0) + len(ent0) + len(sent1) if not rev else 1 + len(sent0)
        
        if pos1 > MAX_LEN or pos2 > MAX_LEN:
            return None
        
        avai_len = len(re_tokens)
        if avai_len > MAX_LEN:
            return None

        # Padding
        re_tokens = np.concatenate([re_tokens, [self.tokenizer.pad_token_id]*(MAX_LEN - avai_len)])
        re_tokens = torch.tensor(re_tokens).long().unsqueeze(0)  # (1, L)

        # Attention mask
        att_mask = torch.zeros(re_tokens.size()).long()  # (1, L)
        att_mask[0, :avai_len] = 1
                    
        feature = {'input_ids': re_tokens, 'input_mask': att_mask, 'ss': pos1, 'os': pos2, 'labels': rel, 'ori_idx': d_id}
        self.features.append(feature)
        
    def exec_insert(self):
        cnt = 0
        start_time = time.time()
        for d in self.data:
            self._tokenize(d)
            cnt += 1
            if cnt % 1000==0:
                # self.write_fn()
                end_time = time.time()
                print(end_time-start_time)
            
        with open(self.w_fn + '.pkl', 'wb') as f:
            pickle.dump(self.features, f)

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, required=False, default='./data/', help="saving file dir")
    parser.add_argument("--aug-path", type=str, required=True, help="augmentation target dataset")
    args = parser.parse_args()
    
    datestr = datetime.datetime.now().strftime("%Y-%m-%d")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    special_tokens = ['[unused0]','[unused1]','[unused2]','[unused3]','[unused4]','[unused5]']
    num_added_tokens = tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    
    wfn = args.aug_path.split('/')[-1].split('.')[-2] + '_aug_'+ datestr
    print(wfn)
    insertObj = InsertTypeInfo(tokenizer, args.save_dir + wfn, args.aug_path)
    insertObj.exec_insert()
