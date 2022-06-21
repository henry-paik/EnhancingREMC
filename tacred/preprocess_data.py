import json
from transformers import AutoConfig, AutoTokenizer
import warnings
import pickle
import argparse
warnings.filterwarnings('ignore')
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

def convert_token(token, idx=None, ss=None, os=None, se=None, oe=None, for_inputid=False):
    """ Convert PTB tokens to normal tokens """
    if (token.lower() == '-lrb-'):
        return ' ('
    elif (token.lower() == '-rrb-'):
        return ' )'
    elif (token.lower() == '-lsb-'):
        return ' ['
    elif (token.lower() == '-rsb-'):
        return ' ]'
    elif (token.lower() == '-lcb-'):
        return ' {'
    elif (token.lower() == '-rcb-'):
        return ' }'
    
    if for_inputid:
        if idx == 0 or ss == idx or os == idx:
            return token
        else:
            return " " + token
    else:
        return token


def tokenize(max_seq_length, tokens, subj_type, obj_type, ss, se, os, oe):
    """
    Implement the following input formats:
        - entity_mask: [SUBJ-NER], [OBJ-NER].
        - entity_marker: [E1] subject [/E1], [E2] object [/E2].
        - entity_marker_punct: @ subject @, # object #.
        - typed_entity_marker: [SUBJ-NER] subject [/SUBJ-NER], [OBJ-NER] obj [/OBJ-NER]
        - typed_entity_marker_punct: @ * subject ner type * subject @, # ^ object ner type ^ object #
    """
    sents = []

    subj_type = tokenizer.tokenize(subj_type.replace("_", " ").lower())
    obj_type = tokenizer.tokenize(obj_type.replace("_", " ").lower())

    for i_t, token in enumerate(tokens):
        tokens_wordpiece = tokenizer.tokenize(token)
        if i_t == ss:
            new_ss = len(sents)
            tokens_wordpiece = ['@'] + ['*'] + subj_type + ['*'] + tokens_wordpiece
        if i_t == se:
            tokens_wordpiece = tokens_wordpiece + ['@']
        if i_t == os:
            new_os = len(sents)
            tokens_wordpiece = ["#"] + ['^'] + obj_type + ['^'] + tokens_wordpiece
        if i_t == oe:
            tokens_wordpiece = tokens_wordpiece + ["#"]

        sents.extend(tokens_wordpiece)
    sents = sents[:max_seq_length - 2]
    input_ids = tokenizer.convert_tokens_to_ids(sents)
    input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)
    return input_ids, new_ss + 1, new_os + 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-in", type=str, default='./data/train.json', help='input .json file')
    parser.add_argument("--fn-out", type=str, default='train', help='filename for output .pkl file')
    args = parser.parse_args()
    
    
    tokenizer = AutoTokenizer.from_pretrained('roberta-large')
    max_seq_length = 370
    features = []

    with open(args.file_in, "r") as fh:
        data = json.load(fh)

    for j, d in enumerate(data):

        ss, se = d['subj_start'], d['subj_end']
        os, oe = d['obj_start'], d['obj_end']

        _tokens = d['token']
        tokens = [convert_token(token, idx, ss, os, se, oe, for_inputid=True) for idx, token in enumerate(_tokens)] #delete special case
        _tokens = [convert_token(token) for token in _tokens] #delete special case
        ori_s = [tokens[i] for i in range(ss, se+1)]
        ori_o = [tokens[i] for i in range(os, oe+1)]

        input_ids, new_ss, new_os = tokenize(max_seq_length, tokens, d['subj_type'], d['obj_type'], ss, se, os, oe)
        rel = LABEL_TO_ID[d['relation']]

        feature = {
            'id': j,
            'sentence': ' '.join(_tokens),
            'sub': ' '.join(ori_s),
            'obj': ' '.join(ori_o),
            'input_ids': input_ids,
            'labels': rel,
            'ss': new_ss,
            'os': new_os,
        }
        features.append(feature)

    with open('./data/{}.pkl'.format(args.fn_out), 'wb') as f:
        pickle.dump(features, f)
        
