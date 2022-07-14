from torch.utils.data import Dataset, DataLoader
import torch
import pickle
import json

class TrainAugcheckCustomDataset(Dataset):
    def __init__(self, path, max_len=256, ref_labels_id=[-1]):
        self.max_len = max_len
        self.data = []
        with open(path, 'rb') as f:        
            _temp = pickle.load(f)
            for _, i in enumerate(_temp):
                if i['labels'] not in ref_labels_id:
                        continue
                self.data.append(i)
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
                        'is_m': 1 if self.data[idx]['labels'] in self.m_id else 0,\
                            'id': self.data[idx]['ori_idx']}
    
class AugdataAugcheckCustomDataset(Dataset):
    def __init__(self, path, max_len=256, ref_labels_id=[-1], selected_instance_id=None):
        self.data = []
        self.max_len = max_len
        self.m_id = ref_labels_id
        self.selected_instance_id = selected_instance_id
        self.counter = {k: 0 for k in self.selected_instance_id}
        
        with open(path, 'rb') as f:
            temp_data = pickle.load(f) 
            for aug_id, line in enumerate(temp_data):
                line['idx'] = aug_id
                # filter non selected MCs
                if line['labels'] not in self.selected_instance_id:
                    continue
                if line['ori_idx'] not in self.selected_instance_id[line['labels']]['selected']:
                    continue
                self.counter[line['labels']] += 1
                self.data.append(line)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        _res = {'input_ids': self.data[idx]['input_ids'][0][:self.max_len], \
                'labels': self.data[idx]['labels'], \
                    'ss': self.data[idx]['ss'], \
                        'os': self.data[idx]['os'], \
                            'input_mask': self.data[idx]['input_mask'][0][:self.max_len], \
                                'is_m': 1 if self.data[idx]['labels'] in self.m_id else 0,\
                                    'id': self.data[idx]['ori_idx'],\
                                        'aug_id': self.data[idx]['idx']}
        return _res
    
def collate_fn_select_aug(batch):
    input_ids = [f["input_ids"] for f in batch]
    input_mask = [f['input_mask'] for f in batch]
    labels = [f["labels"] for f in batch]
    ss = [f["ss"] for f in batch]
    os = [f["os"] for f in batch]
    id_ls = [f["id"] for f in batch]
    if 'aug_id' not in batch[0]:
        aug_id_ls = id_ls
    else:
        aug_id_ls = [f["aug_id"] for f in batch]
        
    # attn_guide = [f["is_m"] for f in batch]

    input_ids = torch.stack(input_ids)
    input_mask = torch.stack(input_mask)
    labels = torch.tensor(labels, dtype=torch.long)
    ss = torch.tensor(ss, dtype=torch.long)
    os = torch.tensor(os, dtype=torch.long)
    id_ls = torch.tensor(id_ls, dtype=torch.long)
    aug_id_ls = torch.tensor(aug_id_ls, dtype=torch.long)
    # attn_guide = torch.tensor(attn_guide, dtype=torch.long)
    
    output = (input_ids, input_mask, labels, ss, os, id_ls, aug_id_ls)
    return output

class AugCustomDataset(Dataset):
    """
    Use original(except noisy data) and selected augmentation
    - denoised original data
    - selected augmentation
    """
    def __init__(self, train_path, aug_path, selected_instance_idx, aug_s_i_idx, ref_labels_id=[-1], max_len=256):
        self.data = []
        self.aug_data = []
        self.m_id = ref_labels_id
        self.selected_instance_idx = selected_instance_idx
        self.aug_s_i_idx = aug_s_i_idx
        self.data_id = []
        self.aug_id = []
        self.max_len = max_len
        # train dataset
        # filter removed instance
        with open(train_path, 'rb') as f:
            _temp = pickle.load(f)
            for j, i in enumerate(_temp):
                if i['labels'] in self.selected_instance_idx:
                    # if i['id'] in self.selected_instance_idx[i['labels']]['removed']:
                    if i['ori_idx'] not in self.selected_instance_idx[i['labels']]['selected']:
                        continue
                    else:
                        self.data_id.append(j) # save reliable MC instances id
                self.data.append(i)

        # load aug data
        with open(aug_path, 'rb') as f:
            temp_data = pickle.load(f) 
            for aug_id, line in enumerate(temp_data):
                line['idx'] = aug_id
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
        return {'input_ids': self.data[idx]['input_ids'][0][:self.max_len], \
            'labels': self.data[idx]['labels'], \
                'ss': self.data[idx]['ss'], \
                    'os': self.data[idx]['os'], \
                        'input_mask': self.data[idx]['input_mask'][0][:self.max_len],\
                            'is_m': 1 if self.data[idx]['labels'] in self.m_id else 0}

class OnlyAugCustomDataset(Dataset):
    """
    Use original(except noisy data) and selected augmentation
    - denoised original data
    - selected augmentation
    """
    def __init__(self, train_path, aug_path, selected_instance_idx, aug_s_i_idx, ref_labels_id=[-1], max_len=256):
        self.data = []
        self.aug_data = []
        self.m_id = ref_labels_id
        self.selected_instance_idx = selected_instance_idx
        self.aug_s_i_idx = aug_s_i_idx
        self.data_id = []
        self.aug_id = []
        self.max_len = max_len
        # train dataset
        # filter removed instance
        with open(train_path, 'rb') as f:
            _temp = pickle.load(f)
            for j, i in enumerate(_temp):
                if i['labels'] in self.selected_instance_idx:
                    # if i['id'] in self.selected_instance_idx[i['labels']]['removed']:
                    if i['ori_idx'] not in self.selected_instance_idx[i['labels']]['selected']:
                        continue
                    else:
                        self.data_id.append(j) # save reliable MC instances id

        # load aug data
        with open(aug_path, 'rb') as f:
            temp_data = pickle.load(f) 
            for aug_id, line in enumerate(temp_data):
                line['idx'] = aug_id
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
        return {'input_ids': self.data[idx]['input_ids'][0][:self.max_len], \
            'labels': self.data[idx]['labels'], \
                'ss': self.data[idx]['ss'], \
                    'os': self.data[idx]['os'], \
                        'input_mask': self.data[idx]['input_mask'][0][:self.max_len],\
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
            
        # integrity check
        self.data = [i for i in self.data if len(i['input_ids']) != 370]
        self.data = [i for i in self.data if not i['ss'] == i['os']]
        self.data = [i for j, i in enumerate(self.data) if (i['ss'] < len(i['input_ids'])) & (i['os'] < len(i['input_ids']))] # no larger than inputids

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