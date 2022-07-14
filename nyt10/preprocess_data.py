import json
from transformers import AutoConfig, AutoTokenizer
import warnings
import pickle
import argparse
import torch
from bert_data_util.meta_data import LABEL_TO_ID
warnings.filterwarnings('ignore')

def prerpocess(item, tokenizer, max_length=256):
    """
    Args:
        item: data instance containing 'text' / 'token', 'h' and 't'
    Return:
        Name of the relation of the sentence
    """
    # Sentence -> token
    if 'text' in item:
        sentence = item['text']
        is_token = False
    else:
        sentence = item['token']
        is_token = True
    pos_head = item['h']['pos']
    pos_tail = item['t']['pos']

    pos_min = pos_head
    pos_max = pos_tail
    if pos_head[0] > pos_tail[0]:
        pos_min = pos_tail
        pos_max = pos_head
        rev = True
    else:
        rev = False

    if not is_token:
        sent0 = tokenizer.tokenize(sentence[:pos_min[0]])
        ent0 = tokenizer.tokenize(sentence[pos_min[0]:pos_min[1]])
        sent1 = tokenizer.tokenize(sentence[pos_min[1]:pos_max[0]])
        ent1 = tokenizer.tokenize(sentence[pos_max[0]:pos_max[1]])
        sent2 = tokenizer.tokenize(sentence[pos_max[1]:])
    else:
        sent0 = tokenizer.tokenize(' '.join(sentence[:pos_min[0]]))
        ent0 = tokenizer.tokenize(' '.join(sentence[pos_min[0]:pos_min[1]]))
        sent1 = tokenizer.tokenize(' '.join(sentence[pos_min[1]:pos_max[0]]))
        ent1 = tokenizer.tokenize(' '.join(sentence[pos_max[0]:pos_max[1]]))
        sent2 = tokenizer.tokenize(' '.join(sentence[pos_max[1]:]))

    ent0 = ['[unused0]'] + ent0 + ['[unused1]'] if not rev else ['[unused2]'] + ent0 + ['[unused3]']
    ent1 = ['[unused2]'] + ent1 + ['[unused3]'] if not rev else ['[unused0]'] + ent1 + ['[unused1]']


    re_tokens = ['[CLS]'] + sent0 + ent0 + sent1 + ent1 + sent2 + ['[SEP]']

    pos1 = 1 + len(sent0) if not rev else 1 + len(sent0 + ent0 + sent1)
    pos2 = 1 + len(sent0 + ent0 + sent1) if not rev else 1 + len(sent0)
    pos1 = min(max_length - 1, pos1)
    pos2 = min(max_length - 1, pos2)
        
    indexed_tokens = tokenizer.convert_tokens_to_ids(re_tokens)
    avai_len = len(indexed_tokens)

    # Padding
    while len(indexed_tokens) < max_length:
        indexed_tokens.append(0)  # 0 is id for [PAD]
        indexed_tokens = indexed_tokens[:max_length]
    indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # (1, L)

    # Attention mask
    att_mask = torch.zeros(indexed_tokens.size()).long()  # (1, L)
    att_mask[0, :avai_len] = 1

    return {'input_ids': indexed_tokens, 'input_mask': att_mask, 'ss': pos1, 'os': pos2, 'labels': LABEL_TO_ID[item['relation']], 'ori_idx': item['line_idx']}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-in", type=str, required=True, help='input .txt file')
    parser.add_argument("--fn-out", type=str, required=True, help='filename for output .pkl file')
    parser.add_argument("--max-seq-len", type=int, default=256, help='input sentence max length; sentence over max length will be skipped')
    args = parser.parse_args()
    
    data = []
    
    # sample line of data
    # {'text': 'One of the newest auditoriums in New York , built into the bedrock below Carnegie Hall , has added a dose of richness to the concert scene .',
    #   'relation': '/location/location/contains',
    #   't': {'pos': [73, 86], 'id': 'm.016p8t', 'name': 'Carnegie Hall'},
    #   'h': {'pos': [33, 41], 'id': 'm.059rby', 'name': 'New York'},
    #   'line_idx': 0}
    with open(args.file_in, 'r') as f:
        for idx, line in enumerate(f):
            line = line.rstrip()
            if len(line) > 0:
                line = eval(line)
                line['line_idx'] = idx
                data.append(line)
    total_len = len(data)
    print("original # data", total_len)
    print("original result data format \n", data[100])
    
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    special_tokens = ['[unused0]','[unused1]','[unused2]','[unused3]',]
    added_token = tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})

    re_data = []
    for cnt, _item in enumerate(data, 1):
        l = prerpocess(_item, tokenizer=tokenizer, max_length=args.max_seq_len)
        if l != None:
            re_data.append(l)
        if cnt % 1000 == 0:
            print("{} / {}".format(cnt, total_len))

    # while data:
    #     cnt += 1
    #     _item = data.pop(0)
    #     l = prerpocess(_item, tokenizer=tokenizer, max_length=args.max_seq_len)
    #     if l != None:
    #         re_data.append(l)
    #     if cnt % 1000 == 0:
    #         print("{} / {}".format(cnt, total_len))

    print("sample result data format", re_data[100])
    print("result # data", len(re_data))

    with open('./data/' + args.fn_out + '.pkl', 'wb') as f:
        pickle.dump(re_data, f)
