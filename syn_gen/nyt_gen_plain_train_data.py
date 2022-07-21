import copy
import json
from itertools import islice
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", type=str, default='./../nyt10/data/nyt10m_train.txt', help='transformation data path')
    parser.add_argument("--out", type=str, default='nyt10-mlm-plain-train.txt', help='output name')
    args = parser.parse_args()
    
    data_ls = []
    with open(args.train_data, 'r') as f:
        for l in f.readlines():
            l = json.loads(l)
            l = l['text'].strip()
            data_ls.append(l)
    
    for l in set(data_ls):
        with open('./data/' + args.out + '.txt', 'a+') as f_w:
            f_w.write(l + '\n')