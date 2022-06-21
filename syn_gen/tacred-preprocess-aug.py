from transformers import AutoTokenizer
import time
import json
import datetime
import argparse
import csv
import numpy as np

class InsertTypeInfo:
    
    def __init__(self, tokenizer, w_fn, aug_path, ori_path):
        self.features = list()
        self.tokenizer = tokenizer
        self.data = None
        self.ori_data = None
        self.w_fn = w_fn
        self.cnt = 0
        self.s_indicator = self.tokenizer.get_vocab()['@']
        self.st_indicator = self.tokenizer.get_vocab()['*']
        self.o_indicator = self.tokenizer.get_vocab()['#']
        self.ot_indicator = self.tokenizer.get_vocab()['^']
        
        self._ori_load_data(ori_path)
        self._load_data(aug_path)
        
    def _ori_load_data(self, ori_path):
        """
        load original data for subj/obj NER
        """
        with open(ori_path, 'r') as f:
            self.ori_data = {}
            for i, l in enumerate(json.load(f)):
                self.ori_data[i] = {'st': None, 'ot': None}
                self.ori_data[i]['st'] = l['subj_type'].lower().replace('_', ' ')
                self.ori_data[i]['ot'] = l['obj_type'].lower().replace('_', ' ')
    
    def _load_data(self, aug_path):
        """
        load generated augmentation dataset
        """
        self.data = []
        with open(aug_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f, delimiter='\t')
            for i, row in enumerate(reader):
                l = [json.loads(r.encode().decode('utf-8-sig')) for r in row]
                self.data.append(l)
#         len_ls.append(len(l))
    
    def write_fn(self):
        with open(self.w_fn + '.json', 'a+') as wf: 
            for ff in self.features:
                self.cnt += 1
                ff['idx'] = self.cnt
                print(ff)
                wf.write(json.dumps(ff) + '\n')
        self.features[:] = []
        
    # def _tokenize(self, input_ids, subj_type, obj_type, ss, se, os, oe, tokenizer, max_seq_length=370):
    def _tokenize(self, input_data):
        """
        target: "<s>The 57 NGOs that support#^organization^TOAID# are Compassion China Foundation, the Chinese Social China Foundation, 
        Foreign Disaster Volunteer Institute in Thailand, the@*organization*Noordhoff Craniofacial Foundation@ and the Chinese Army for Peace Corps.</s>",

        "sub": "Noordhoff Craniofacial Foundation", "obj": "TOAID", "new_sub": "@*organization*Noordhoff Craniofacial Foundation@"
        line_idx, n_s, old_sub_idx, old_obj_idx, label_id
        83208	[0, 4688, 1566, 11, ..., 3113, 479, 2]	[24]	[12, 13, 14, 15]	19
        new_subj.extend([1039, 3226])
        new_subj.extend(subj_type_ids)
        new_subj.append(3226)
        new_subj.extend(subj_token) 
        new_subj.append(1039)
        """

        def _remove_wspace(inputids):
            inputids = self.tokenizer.decode(inputids).strip()
            inputids = self.tokenizer(inputids, add_special_tokens=False)['input_ids']
            return inputids
        
        d_id, input_ids, subji, obji, rel = input_data
        input_ids = np.array(input_ids)
        
        ss, se, os, oe = subji[0], subji[-1], obji[0], obji[-1]
        subj_type, obj_type = self.ori_data[d_id]['st'], self.ori_data[d_id]['ot']
        sub_obj_order = True if ss < os else False
        new_ss = None; new_os = None
        res = None
        
        if sub_obj_order:
            a = input_ids[:ss]
            b = input_ids[ss:se+1]
            b = _remove_wspace(b)

            c = input_ids[se+1:os]
            d = input_ids[os:oe+1]
            d = _remove_wspace(d)

            e = input_ids[oe+1:]
            
            new_ss = len(a)
            res = np.concatenate([a, [self.s_indicator, self.st_indicator], self.tokenizer(subj_type, add_special_tokens=False)['input_ids'], [self.st_indicator]])
            res = np.concatenate([res, b, [self.s_indicator], c])
            
            new_os = len(res)
            res = np.concatenate([res, [self.o_indicator, self.ot_indicator], self.tokenizer(obj_type, add_special_tokens=False)['input_ids'], [self.ot_indicator]])
            res = np.concatenate([res, d, [self.o_indicator], e])

        else:
            a = input_ids[:os]
            b = input_ids[os:oe+1]
            b = _remove_wspace(b)

            c = input_ids[oe+1:ss]
            d = input_ids[ss:se+1]
            d = _remove_wspace(d)

            e = input_ids[se+1:]
            
            new_os = len(a)
            res = np.concatenate([a, [self.o_indicator, self.ot_indicator], self.tokenizer(obj_type, add_special_tokens=False)['input_ids'], [self.ot_indicator]])
            res = np.concatenate([res, b, [self.o_indicator], c])
            
            new_ss = len(res)
            res = np.concatenate([res, [self.s_indicator, self.st_indicator], self.tokenizer(subj_type, add_special_tokens=False)['input_ids'], [self.st_indicator]])
            res = np.concatenate([res, d, [self.s_indicator], e])
        
        assert res[new_os] == self.o_indicator
        assert res[new_ss] == self.s_indicator

        feature = {
            'id': d_id,
            'sentence': self.tokenizer.decode(input_ids),
            'sub': self.tokenizer.decode([input_ids[i] for i in subji]),
            'obj': self.tokenizer.decode([input_ids[i] for i in obji]),
            'input_ids': res.tolist(),
            'labels': rel,
            'ss': int(new_ss),
            'os': int(new_os),
        }
        
        self.features.append(feature)

        
    def exec_insert(self):
        cnt = 0
        start_time = time.time()
        for d in self.data:
            self._tokenize(d)
            cnt += 1
            if cnt % 1000==0:
                self.write_fn()
                end_time = time.time()
                print(end_time-start_time)
            
        if len(self.features) > 0:
            self.write_fn()

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, required=False, default='./data/', help="saving file dir")
    parser.add_argument("--ori-path", type=str, required=False, default='./../tacred/data/train.json', help="original train dataset")
    parser.add_argument("--aug-path", type=str, required=False, default='./source/merge-tacred-gen-roberta-base-increment_sub01_gen300_tau15-2022-02-11.csv', help="augmentation dataset")
    args = parser.parse_args()
    
    datestr = datetime.datetime.now().strftime("%Y-%m-%d")
    tokenizer = AutoTokenizer.from_pretrained('roberta-large')
    special_tokens = ['madeupword0000', 'madeupword0001', 'madeupword0002'] # for generation
    num_added_tokens = tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    
    wfn = args.aug_path.split('/')[-1].split('.')[-2] + '_type_inserted_'+ datestr
    insertObj = InsertTypeInfo(tokenizer, args.save_dir + wfn, args.aug_path, args.ori_path)
    insertObj.exec_insert()
