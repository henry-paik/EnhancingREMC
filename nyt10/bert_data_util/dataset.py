from torch.utils.data import Dataset, DataLoader
import torch
import pickle
import numpy as np

# training data format
# {'input_ids': tensor([[    0,  3762,     9,     5,  8946, 15279,  4031,    29,    11,  1437,
#           50261,  4030,   469, 50261,  2156,  1490,    88,     5, 35085,   874,
#            1437, 50262,  9518, 48451,  1631, 50262,  2156,    34,   355,    10,
#           12234,     9, 38857,     7,     5,  4192,  1310,   479,     2,     1,
#               1,     1,     1,     1,     1,     1,     1,     1,     1,     1, ...
#               1,     1,     1,     1,     1,     1]]),
#  'input_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...
#           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
#  'ss': 10,
#  'os': 21,
#  'labels': 1}

class CustomDataset(Dataset):
    def __init__(self, path, max_len=256, ref_labels_id=[-1]):
        self.max_len = max_len
        with open(path, 'rb') as f:
            self.data = pickle.load(f) 
            self.m_id = ref_labels_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # print(idx)
        return {'input_ids': self.data[idx]['input_ids'][0][:self.max_len], \
            'labels': self.data[idx]['labels'], \
                'ss': self.data[idx]['ss'], \
                    'os': self.data[idx]['os'], \
                        'input_mask': self.data[idx]['input_mask'][0][:self.max_len], \
                        'is_m': 1 if self.data[idx]['labels'] in self.m_id else 0}
        
def collate_fn(batch):
    input_ids = [f["input_ids"] for f in batch]
    input_mask = [f['input_mask'] for f in batch]
    labels = [f["labels"] for f in batch]
    ss = [f["ss"] for f in batch]
    os = [f["os"] for f in batch]
    attn_guide = [f["is_m"] for f in batch]

    input_ids = torch.stack(input_ids)
    input_mask = torch.stack(input_mask)
    labels = torch.tensor(labels, dtype=torch.long)
    ss = torch.tensor(ss, dtype=torch.long)
    os = torch.tensor(os, dtype=torch.long)
    attn_guide = torch.tensor(attn_guide, dtype=torch.long)

    output = (input_ids, input_mask, labels, ss, os, attn_guide)
    return output
