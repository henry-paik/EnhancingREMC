import copy
import json
from itertools import islice
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", type=str, default='./../tacred/data/train.json', help='transformation data path')
    parser.add_argument("--out", type=str, default='tacred-mlm-plain-train', help='output name')
    args = parser.parse_args()
    
    data_ls = []
    with open(args.train_data, 'r', encoding='utf-8') as f:
        l = json.load(f)
    
    with open('./data/' + args.out + '.txt', 'w') as f:
        for ll in l:
            ll = ' '.join(ll['token'])
            f.write(ll + '\n')
            