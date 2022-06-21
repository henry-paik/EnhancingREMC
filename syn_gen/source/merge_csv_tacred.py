import os 
import datetime
import argparse

if __name__ == "__main__":
    
    datestr = datetime.datetime.strftime(datetime.datetime.now(), '%y-%m-%d')
    parser = argparse.ArgumentParser()
    parser.add_argument("--part-fn", type=str, default='cuda', help='target .csv files name that was seperately generated')
    args = parser.parse_args()
        
    datetime_str = datetime.datetime.now().strftime("%Y-%m-%d")
    exclude_fn = 'meta'
    ls = [i for i in os.listdir() if (args.part_fn in i) and (exclude_fn not in i)]

    W_NAME = 'merge-{}-{}.csv'.format(args.part_fn, datetime_str)
    for fn in ls:
        with open(W_NAME, 'a+') as wf:
            with open(fn, 'r') as f:
                for l in f.readlines():
                    wf.write(l)
