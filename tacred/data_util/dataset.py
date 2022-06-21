from torch.utils.data import Dataset, DataLoader
import torch
import pickle

class CustomDataset(Dataset):
    def __init__(self, path, ref_labels_id=[-1], labels=range(42)):
        with open(path, 'rb') as f:
            self.data = pickle.load(f) # json row
            self.m_id = ref_labels_id
        self.data = [l for l in self.data if l['labels'] in labels]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {'input_ids': self.data[idx]['input_ids'], \
            'labels': self.data[idx]['labels'], \
                'ss': self.data[idx]['ss'], \
                    'os': self.data[idx]['os'], \
                        'is_m': 1 if self.data[idx]['labels'] in self.m_id else 0}
    
def collate_fn(batch):
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [f["input_ids"] + [1.0] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    labels = [f["labels"] for f in batch]
    ss = [f["ss"] for f in batch]
    os = [f["os"] for f in batch]
    attn_guide = [f["is_m"] for f in batch]

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)
    ss = torch.tensor(ss, dtype=torch.long)
    os = torch.tensor(os, dtype=torch.long)
    attn_guide = torch.tensor(attn_guide, dtype=torch.long)

    output = (input_ids, input_mask, labels, ss, os, attn_guide)
    return output

