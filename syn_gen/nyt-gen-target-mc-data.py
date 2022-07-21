import copy
import json
from itertools import islice
import argparse
import torch
from transformers import AutoTokenizer
import pickle 

LABEL_TO_ID = {"/people/person/nationality": 2, "/time/event/locations": 22, "/people/person/children": 14, "/business/company/advisors": 19, "/business/location": 18, "/business/company/majorshareholders": 16, "/people/person/place_lived": 5, "NA": 0, "/business/company/place_founded": 11, "/location/neighborhood/neighborhood_of": 8, "/people/deceasedperson/place_of_death": 4, "/film/film/featured_film_locations": 21, "/location/region/capital": 23, "/business/company/founders": 6, "/people/ethnicity/geographic_distribution": 17, "/location/country/administrative_divisions": 12, "/people/deceasedperson/place_of_burial": 24, "/location/country/capital": 13, "/business/person/company": 9, "/location/location/contains": 1, "/location/administrative_division/country": 10, "/location/us_county/county_seat": 20, "/people/person/religion": 15, "/people/person/place_of_birth": 3, "/people/person/ethnicity": 7}

ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}
MINOR_LABEL_ID = [20, 17, 24, 21, 19, 22]
MINOR_LABELS =  [ID_TO_LABEL[i] for i in MINOR_LABEL_ID] # 6MCs

MAX_LEN = 256

def preprocess(item, tokenizer, max_length=MAX_LEN):
    """
    Args:
        item: data instance containing 'text' / 'token', 'h' and 't'
    Return:
        Name of the relation of the sentence
    """
    # Sentence -> token
    sentence = item['text']
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

    sent0 = tokenizer.tokenize(sentence[:pos_min[0]].strip())
    ent0 = tokenizer.tokenize(' '+sentence[pos_min[0]:pos_min[1]])
    sent1 = tokenizer.tokenize(' '+sentence[pos_min[1]:pos_max[0]].strip())
    ent1 = tokenizer.tokenize(' '+sentence[pos_max[0]:pos_max[1]])
    sent2 = tokenizer.tokenize(' '+sentence[pos_max[1]:].strip())

    re_tokens = [tokenizer.cls_token] + sent0 + ent0 + sent1 + ent1 + sent2 + [tokenizer.sep_token]
    pos1 = 1 + len(sent0) if not rev else 1 + len(sent0 + ent0 + sent1)
    pos12 = 1 + len(sent0 + ent0) if not rev else 1 + len(sent0 + ent0 + sent1 + ent1)
    pos2 = 1 + len(sent0 + ent0 + sent1) if not rev else 1 + len(sent0)
    pos22 = 1 + len(sent0 + ent0 + sent1 + ent1) if not rev else 1 + len(sent0 + ent0)

    if pos1 > max_length or pos2 > max_length:
        return None
    
    indexed_tokens = tokenizer.convert_tokens_to_ids(re_tokens)
    avai_len = len(indexed_tokens)
    if avai_len > max_length:
        return None

    indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # (1, L)
    return {'input_ids': indexed_tokens, 'ss': pos1, 'se':pos12,  'os': pos2, 'oe': pos22, 'labels': LABEL_TO_ID[item['relation']], 'ori_idx': item['line_idx']}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-in", type=str, required=True, help='input original file')
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    data = []
    with open(args.file_in, 'r') as f:
        for idx, line in enumerate(f):
            line = line.rstrip()
            if len(line) > 0:
                line = eval(line)
                line['line_idx'] = idx
                data.append(line)
                
    re_data = []
    for _item in data:
        if _item['relation'] in MINOR_LABELS:
            l = preprocess(_item, tokenizer=tokenizer)
            if l != None:
                re_data.append(l)

    with open('./data/nyt-10m-6MCgen.pkl', 'wb') as f:
        pickle.dump(re_data, f)
