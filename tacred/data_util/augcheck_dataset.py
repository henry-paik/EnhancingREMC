from torch.utils.data import Dataset, DataLoader
import torch
import pickle
import json

class TrainAugcheckCustomDataset(Dataset):
    def __init__(self, path, ref_labels_id=[-1]):
        """
        Dataset class for selecting high score instances in train dataset 
        """
        self.data = []
        self.m_id = ref_labels_id
        self.counter = None
        
        # selecte aug candidate from train dataset
        with open(path, 'rb') as f:
            self.data = pickle.load(f) # json row
            self._data = []
            for j, i in enumerate(self.data):
                if i['labels'] in self.m_id:
                    self._data.append(i)
            self.data = self._data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        _res = {'input_ids': self.data[idx]['input_ids'], \
                'labels': self.data[idx]['labels'], \
                    'ss': self.data[idx]['ss'], \
                        'os': self.data[idx]['os'], \
                            'is_m': 1 if self.data[idx]['labels'] in self.m_id else 0,\
                                'id': self.data[idx]['id']}
        return _res
    
class AugdataAugcheckCustomDataset(Dataset):
    def __init__(self, path, ref_labels_id=[-1], selected_instance_id=None):
        self.data = []
        self.m_id = ref_labels_id
        self.selected_instance_id = selected_instance_id
        self.counter = None
        self.counter = {k: 0 for k in self.selected_instance_id}
        
        with open(path, 'r') as f:
            for line in f:
                line = json.loads(line)
                # filter non selected MCs
                if line['labels'] not in self.selected_instance_id:
                    continue
                if line['id'] not in self.selected_instance_id[line['labels']]['selected']:
                    continue
                self.counter[line['labels']] += 1
                self.data.append(line)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        _res = {'input_ids': self.data[idx]['input_ids'], \
                'labels': self.data[idx]['labels'], \
                    'ss': self.data[idx]['ss'], \
                        'os': self.data[idx]['os'], \
                            'is_m': 1 if self.data[idx]['labels'] in self.m_id else 0,\
                                'id': self.data[idx]['id'],\
                                    'aug_id': self.data[idx]['idx']}
        return _res
    
def collate_fn_select_aug(batch):
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [f["input_ids"] + [1.0] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    labels = [f["labels"] for f in batch]
    ss = [f["ss"] for f in batch]
    os = [f["os"] for f in batch]
    id_ls = [f["id"] for f in batch]
    if 'aug_id' not in batch[0]:
        aug_id_ls = id_ls
    else:
        aug_id_ls = [f["aug_id"] for f in batch]
        
    # attn_guide = [f["is_m"] for f in batch]

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)
    ss = torch.tensor(ss, dtype=torch.long)
    os = torch.tensor(os, dtype=torch.long)
    id_ls = torch.tensor(id_ls, dtype=torch.long)
    aug_id_ls = torch.tensor(aug_id_ls, dtype=torch.long)
    # attn_guide = torch.tensor(attn_guide, dtype=torch.long)
    
    output = (input_ids, input_mask, labels, ss, os, id_ls, aug_id_ls)
    return output

class AugCustomDatasetWithFactor(Dataset):
    """
    Use original(except noisy data) and selected augmentation
    - denoised original data
    - selected augmentation
    - only for selected instance
    """
    def __init__(self, train_path, aug_path, selected_instance_idx, aug_s_i_idx, ref_labels_id=[-1]):
        self.data = []
        self.aug_data = []
        self.m_id = ref_labels_id
        self.selected_instance_idx = selected_instance_idx
        self.aug_s_i_idx = aug_s_i_idx
        self.data_id = []
        self.aug_id = []
        
        # train dataset
        # filter removed instance
        with open(train_path, 'rb') as f:
            self.data = pickle.load(f)
            self._data = []
            for j, i in enumerate(self.data):
                if i['labels'] in self.selected_instance_idx:
                    if i['id'] not in self.selected_instance_idx[i['labels']]['selected']:
                        continue
                    # else:
                        self.data_id.append(j) # save reliable MC instances id
                self._data.append(i)
            self.data = self._data
                
        with open(aug_path, 'r') as f:
            for line in f:
                line = json.loads(line)
                # filter non selected MCs
                if line['labels'] not in self.aug_s_i_idx:
                    continue
                if line['idx'] not in self.aug_s_i_idx[line['labels']]['selected']:
                    continue
                self.aug_data.append(line)
                self.aug_id.append(line['idx'])
        self.data.extend(self.aug_data)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {'input_ids': self.data[idx]['input_ids'], \
            'labels': self.data[idx]['labels'], \
                'ss': self.data[idx]['ss'], \
                    'os': self.data[idx]['os'], \
                        'is_m': 1 if self.data[idx]['labels'] in self.m_id else 0}
        
class AugCustomDataset(Dataset):
    """
    Use original(except noisy data) and selected augmentation
    - denoised original data
    - selected augmentation
    """
    def __init__(self, train_path, aug_path, selected_instance_idx, aug_s_i_idx, ref_labels_id=[-1]):
        self.data = []
        self.aug_data = []
        self.m_id = ref_labels_id
        self.selected_instance_idx = selected_instance_idx
        self.aug_s_i_idx = aug_s_i_idx
        self.data_id = []
        self.aug_id = []
        
        # train dataset
        # filter removed instance
        with open(train_path, 'rb') as f:
            self.data = pickle.load(f)
            self._data = []
            for j, i in enumerate(self.data):
                if i['labels'] in self.selected_instance_idx:
                    # if i['id'] in self.selected_instance_idx[i['labels']]['removed']:
                    if i['id'] not in self.selected_instance_idx[i['labels']]['selected']:
                        continue
                    # else:
                        self.data_id.append(j) # save reliable MC instances id
                self._data.append(i)
            self.data = self._data
                
        with open(aug_path, 'r') as f:
            for line in f:
                line = json.loads(line)
                # filter non selected MCs
                if line['labels'] not in self.aug_s_i_idx:
                    continue
                if line['idx'] not in self.aug_s_i_idx[line['labels']]['selected']:
                    continue
                self.aug_data.append(line)
                self.aug_id.append(line['idx'])
        self.data.extend(self.aug_data)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {'input_ids': self.data[idx]['input_ids'], \
            'labels': self.data[idx]['labels'], \
                'ss': self.data[idx]['ss'], \
                    'os': self.data[idx]['os'], \
                        'is_m': 1 if self.data[idx]['labels'] in self.m_id else 0}

class AugCustomDatasetWOAUG(Dataset):
    """
    Use original(except noisy data) and selected augmentation
    - denoised original data
    - selected augmentation
    """
    def __init__(self, train_path, selected_instance_idx, ref_labels_id=[-1]):
        self.data = []
        self.m_id = ref_labels_id
        self.selected_instance_idx = selected_instance_idx
        
        # train dataset
        # filter removed instance
        with open(train_path, 'rb') as f:
            self.data = pickle.load(f)
            self._data = []
            for j, i in enumerate(self.data):
                if i['labels'] in self.selected_instance_idx:
                    # if i['id'] in self.selected_instance_idx[i['labels']]['removed']:
                    if i['id'] not in self.selected_instance_idx[i['labels']]['selected']:
                        continue
                self._data.append(i)
            self.data = self._data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {'input_ids': self.data[idx]['input_ids'], \
            'labels': self.data[idx]['labels'], \
                'ss': self.data[idx]['ss'], \
                    'os': self.data[idx]['os'], \
                        'is_m': 1 if self.data[idx]['labels'] in self.m_id else 0}

class AugCustomDatasetTest(Dataset):
    """
    lower removal target prob
    """
    def __init__(self, train_path, aug_path, selected_instance_idx, aug_s_i_idx, ref_labels_id=[-1]):
        self.data = []
        self.aug_data = []
        self.m_id = ref_labels_id
        self.selected_instance_idx = selected_instance_idx
        self.aug_s_i_idx = aug_s_i_idx
        self.data_id = []
        self.aug_id = []
        
        # train dataset
        # filter removed instance
        with open(train_path, 'rb') as f:
            self.data = pickle.load(f)
            self._data = []
            for j, i in enumerate(self.data):
                if i['labels'] in self.selected_instance_idx:
                    if i['id'] in self.selected_instance_idx[i['labels']]['removed']:
                        i['labels'] += 99 # tag
                        print(i)
                    # else:
                        self.data_id.append(j) # save reliable MC instances id
                self._data.append(i)
            self.data = self._data
                
        with open(aug_path, 'r') as f:
            for line in f:
                line = json.loads(line)
                # filter non selected MCs
                if line['labels'] not in self.aug_s_i_idx:
                    continue
                if line['idx'] not in self.aug_s_i_idx[line['labels']]['selected']:
                    continue
                self.aug_data.append(line)
                self.aug_id.append(line['idx'])
        self.data.extend(self.aug_data)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {'input_ids': self.data[idx]['input_ids'], \
            'labels': self.data[idx]['labels'], \
                'ss': self.data[idx]['ss'], \
                    'os': self.data[idx]['os'], \
                        'is_m': 1 if self.data[idx]['labels'] in self.m_id else 0}
        
        
class OnlydenoiseAugCustomDataset(Dataset):
    """
    Use original(except noisy data) and selected augmentation
    - denoised original data
    - selected augmentation
    """
    def __init__(self, train_path, selected_instance_idx, ref_labels_id=[-1]):
        self.data = []
        self.aug_data = []
        self.m_id = ref_labels_id
        self.selected_instance_idx = selected_instance_idx
        self.data_id = []
        self.aug_id = []
        
        # train dataset
        # filter removed instance
        with open(train_path, 'rb') as f:
            self.data = pickle.load(f)
            self._data = []
            for j, i in enumerate(self.data):
                if i['labels'] in self.selected_instance_idx:
                    if i['id'] in self.selected_instance_idx[i['labels']]['removed']:
                        continue
                self._data.append(i)
            self.data = self._data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {'input_ids': self.data[idx]['input_ids'], \
            'labels': self.data[idx]['labels'], \
                'ss': self.data[idx]['ss'], \
                    'os': self.data[idx]['os'], \
                        'is_m': 1 if self.data[idx]['labels'] in self.m_id else 0}
        
        
def collate_fn_aug(batch):
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [f["input_ids"] + [1.0] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    labels = [f["labels"] for f in batch]
    ss = [f["ss"] for f in batch]
    os = [f["os"] for f in batch]
    idx_ls = [f["idx"] for f in batch]
    # attn_guide = [f["is_m"] for f in batch]

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)
    ss = torch.tensor(ss, dtype=torch.long)
    os = torch.tensor(os, dtype=torch.long)
    idx_ls = torch.tensor(idx_ls, dtype=torch.long)
    # attn_guide = torch.tensor(attn_guide, dtype=torch.long)
    
    output = (input_ids, input_mask, labels, ss, os, idx_ls)
    return output

def collate_fn_aug_test(batch):
    """
    wehn aug training, lower removal target prob
    """
    removal_target = [j for j, f in enumerate(batch) if f["labels"]>99]
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
    removal_target = torch.tensor(removal_target, dtype=torch.long)

    output = (input_ids, input_mask, labels, ss, os, attn_guide, removal_target)
    return output