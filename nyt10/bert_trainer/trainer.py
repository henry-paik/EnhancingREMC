
import pickle
import torch
import sys, os.path
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
import time
import pandas as pd
import sklearn
_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(_dir)
from utils import soft_label_loss, get_auc, soft_label_loss_mc

class Trainer:
    def __init__(self, config=None, logger=None, model=None, optimizer=None, scheduler=None, data_loader=None, val_loader=None, ref_labels_id=None, is_test=False, is_augcheck=False, need_attn=False):
        self.config = config
        self.scheduler = scheduler
        self.model = model
        self.ref_labels_id = ref_labels_id
        self.data_loader = data_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.logger = logger
        self.model = self.model.to(self.config.device)
        self.step_iter = 0 # for exponential addtional train
        self.start_epoch = 0 # for resume
        self.is_test = is_test
        self.is_augcheck = is_augcheck
        self.need_attn = need_attn

        # for augmentation
        self.id_ls_ls = None
        self.aug_id_ls_ls = None
        self.pred_ls_ls = None
        self.pred_ls_raw = None
        self.ans_ls_ls = None
        self.attn_score_ls_ls = None

        # continue training
        if self.config.resume is not None:
            self._resume_checkpoint(self.config.resume)

    def train(self, is_weighted=False, w=30):
        for epoch in range(self.start_epoch, self.config.total_epochs):
            if is_weighted:
                self._train_epoch_weighted(epoch, mc_w=w)
            else:
                self._train_epoch(epoch)
            self._save_checkpoint(epoch)
            self._valid_epoch(epoch)

    def additional_train(self, optimizer, is_weighted=False, w=30, is_frozen=True):
        self.start_epoch = 1
        self.optimizer = optimizer
        if is_frozen:
            for param in self.model.bone.parameters():
                param.requires_grad = False

        for epoch in range(self.start_epoch, self.config.total_epochs+1):
            if self.config.additional_train_type == 'exponential':
                if is_weighted:
                    self._train_epoch_exponential_weight(epoch, mc_w=w)
                else:
                    self._train_epoch_exponential(epoch)
            self._save_checkpoint(epoch, is_additional_train=True)
            # self._valid_epoch(epoch)
            
    def test(self):
        self._valid_epoch(0)
      

    def select_train_instance_withfactor(self, aug_num: dict):
        """
        select train instances for augmentation
        aug_num: desired number of augmentation
            {12: array([ 1.,  1.,  1.,  2., ...]),
            27: array([ 2.,  3.,  4.,  6.,  7. 31.]),
            29: array([ 1.,  1.,  2.,  2.,  3., ..., 32.]),
            41: array([ 5., 10., 14., 19., 23., 28.])}
        """
        self._valid_epoch(0)
        selected_instance_idx = self._get_train_aug_idx_withfactor(aug_num)
        return selected_instance_idx
    
    def select_aug_instance_withfactor(self, aug_num, selected_instance_id):
        """
        select train instances for augmentation
        """
        print(aug_num, selected_instance_id)
        self._valid_epoch(0)
        selected_aug_instance_idx = self._get_aug_idx_withfactor(aug_num, selected_instance_id)
        return selected_aug_instance_idx
   
    def _train_epoch_weighted(self, epoch, mc_w):
        running_loss = 0
        self.model.train()
        for batch_idx, batch in enumerate(self.data_loader, 1):                
            try:
                self.optimizer.zero_grad()
                input_ids, input_mask, labels, ss, os, attn_guide  = tuple(t.to(self.config.device) for t in batch)
                mc_idx = [elem in self.ref_labels_id for elem in labels.view(-1)]

                out, inverse_attn_out, ref_val_out, attn_score, gating = self.model(input_ids=input_ids, attention_mask=input_mask, sub_idx=ss, obj_idx=os, attn_guide=attn_guide, device=self.config.device)


                if sum(mc_idx) > 0: # if minor class exist in batch
                    loss = soft_label_loss_mc(out, labels, mc_idx=torch.tensor(mc_idx), eps=self.config.main_ep, w=mc_w) + (torch.sum(inverse_attn_out[mc_idx, labels.view(-1)[mc_idx]])/sum(mc_idx)) + soft_label_loss(ref_val_out, torch.LongTensor(sorted(self.ref_labels_id)).to(self.config.device), eps=self.config.refval_ep)                    
                else:
                    loss = soft_label_loss_mc(out, labels, mc_idx=torch.tensor(mc_idx), eps=self.config.main_ep, w=mc_w) + soft_label_loss(ref_val_out, torch.LongTensor(sorted(self.ref_labels_id)).to(self.config.device), eps=self.config.refval_ep)

                print(torch.cuda.memory_allocated()/1024**3)
                print(torch.cuda.memory_reserved()/1024**3)

                loss.backward()
                self.optimizer.step()
                current_lr = self.scheduler.get_last_lr()[0]
                self.scheduler.step()
                running_loss += loss.detach().cpu()

                if batch_idx % self.config.log_step == 0:
                    self.logger.info('Epoch: {} Step: {} Loss: {} lr: {}'.format(epoch, batch_idx, running_loss/batch_idx, current_lr))
                    
            except Exception as e:
                self.logger.error(e)

    def _train_epoch(self, epoch):
        running_loss = 0
        self.model.train()
        for batch_idx, batch in enumerate(self.data_loader, 1):
            print(1)
            try:
                self.optimizer.zero_grad()

                input_ids, input_mask, labels, ss, os, attn_guide  = tuple(t.to(self.config.device) for t in batch)
                out, inverse_attn_out, ref_val_out, attn_score, gating = self.model(input_ids=input_ids, attention_mask=input_mask, sub_idx=ss, obj_idx=os, attn_guide=attn_guide, device=self.config.device)
                
                mc_idx = [elem in self.ref_labels_id for elem in labels.view(-1)]

                if sum(mc_idx) > 0: # if minor class exist in batch
                    loss = soft_label_loss(out, labels, eps=self.config.main_ep) + (torch.sum(inverse_attn_out[mc_idx, labels.view(-1)[mc_idx]])/sum(mc_idx)) + soft_label_loss(ref_val_out, torch.LongTensor(sorted(self.ref_labels_id)).to(self.config.device), eps=self.config.refval_ep)
                else:
                    loss = soft_label_loss(out, labels, eps=self.config.main_ep) + soft_label_loss(ref_val_out, torch.LongTensor(sorted(self.ref_labels_id)).to(self.config.device), eps=self.config.refval_ep)

                print(torch.cuda.memory_allocated()/1024**3)
                print(torch.cuda.memory_reserved()/1024**3)

                loss.backward()
                self.optimizer.step()
                current_lr = self.scheduler.get_last_lr()[0]
                self.scheduler.step()
                running_loss += loss.detach().cpu()

                if batch_idx % self.config.log_step == 0:
                    self.logger.info('Epoch: {} Step: {} Loss: {} lr: {}'.format(epoch, batch_idx, running_loss/batch_idx, current_lr))
                    
            except Exception as e:
                self.logger.error(e)
                print(input_ids, input_mask, labels, ss, os, attn_guide)
                print(mc_idx)
                print(out.size(), inverse_attn_out.size(), ref_val_out.size())
                self.logger.info('{}; {}; {}; {}; {}; {};'.format(input_ids, input_mask, labels, ss, os, attn_guide))
                raise

    def _train_epoch_exponential(self, epoch):
        running_loss = 0
        self.model.train()
        T = len(self.data_loader)*(self.config.total_epochs-2)
        for batch_idx, batch in enumerate(self.data_loader, 1):
            self.step_iter +=1
            thr = np.exp((self.step_iter/T-1)*5) * (self.config.exp_max_thr - self.config.exp_min_thr) + self.config.exp_min_thr
            thr = min(thr, 0.85)
                
            try:
                self.optimizer.zero_grad()
                input_ids, input_mask, labels, ss, os, attn_guide  = tuple(t.to(self.config.device) for t in batch)
                out, inverse_attn_out, ref_val_out, attn_score, gating = self.model(input_ids=input_ids, attention_mask=input_mask, sub_idx=ss, obj_idx=os, attn_guide=attn_guide, device=self.config.device)
                
                # 1) loss_ref_val
                loss = soft_label_loss(ref_val_out, torch.LongTensor(sorted(self.ref_labels_id)).to(self.config.device), eps=self.config.refval_ep)
                non_calc_idx = torch.any(torch.softmax(out, dim=1)>=thr, dim=1).detach().cpu() # confidence exceed threshold

                mc_idx = torch.tensor([elem in self.ref_labels_id for elem in labels.view(-1)])
                mc_idx = mc_idx&~non_calc_idx # mc&calc
                # 2) inverse attn guide loss for mc: mc & calc
                if sum(mc_idx) > 0:
                    loss += (torch.sum(inverse_attn_out[mc_idx, labels.view(-1)[mc_idx]])/sum(mc_idx)) 
                # 3) main loss for calc
                if sum(~non_calc_idx) > 0: #calc
                    loss += soft_label_loss(out[~non_calc_idx], labels[~non_calc_idx], eps=self.config.main_ep)
                
                loss.backward()
                self.optimizer.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                running_loss += loss.detach().cpu()
                
                if batch_idx % self.config.log_step == 0:
                    self.logger.info('Epoch: {} Step: {} Loss: {} lr: {} trh: {}'.format(epoch, batch_idx, running_loss/batch_idx, current_lr, thr))
                    
            except Exception as e:
                self.logger.error(e)


    def _train_epoch_exponential_weight(self, epoch, mc_w):
        running_loss = 0
        self.model.train()
        T = len(self.data_loader)*(self.config.total_epochs-2)
        for batch_idx, batch in enumerate(self.data_loader, 1):
            self.step_iter +=1
            thr = np.exp((self.step_iter/T-1)*5) * (self.config.exp_max_thr - self.config.exp_min_thr) + self.config.exp_min_thr
            thr = min(thr, 0.85)
                
            try:
                self.optimizer.zero_grad()
                input_ids, input_mask, labels, ss, os, attn_guide  = tuple(t.to(self.config.device) for t in batch)
                out, inverse_attn_out, ref_val_out, attn_score, gating = self.model(input_ids=input_ids, attention_mask=input_mask, sub_idx=ss, obj_idx=os, attn_guide=attn_guide, device=self.config.device)
                
                # 1) loss_ref_val
                loss = soft_label_loss(ref_val_out, torch.LongTensor(sorted(self.ref_labels_id)).to(self.config.device), eps=self.config.refval_ep)
                non_calc_idx = torch.any(torch.softmax(out, dim=1)>=thr, dim=1).detach().cpu() # confidence exceed threshold

                mc_idx = torch.tensor([elem in self.ref_labels_id for elem in labels.view(-1)])
                mc_idx = mc_idx&~non_calc_idx # mc&calc
                # 2) inverse attn guide loss for mc: mc & calc
                if sum(mc_idx) > 0:
                    loss += (torch.sum(inverse_attn_out[mc_idx, labels.view(-1)[mc_idx]])/sum(mc_idx)) 
                # 3) main loss for calc
                if sum(~non_calc_idx) > 0: #calc
                    loss += soft_label_loss_mc(out[~non_calc_idx], labels[~non_calc_idx], mc_idx=mc_idx[~non_calc_idx], eps=self.config.main_ep, w=mc_w)
                loss.backward()
                self.optimizer.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                running_loss += loss.detach().cpu()
                
                if batch_idx % self.config.log_step == 0:
                    print(torch.cuda.memory_allocated()/1024**3)
                    print(torch.cuda.memory_reserved()/1024**3)
                    print("mc&calc")
                    print(mc_idx)
                    print("calc")
                    print(~non_calc_idx)
                    self.logger.info('Epoch: {} Step: {} Loss: {} lr: {} trh: {}'.format(epoch, batch_idx, running_loss/batch_idx, current_lr, thr))
                    
            except Exception as e:
                self.logger.error(e)
                
                
    def _valid_epoch(self, epoch, class_labels=np.arange(25).tolist()):
        pred_ls = []; answer_ls=[]; attn_score_ls=[]; id_ls=[]; aug_id_ls=[]; gating_ls = []
        class_labels.pop(0)
        loss_func = nn.CrossEntropyLoss()
        running_loss = 0
        print("===val in")
        print(torch.cuda.memory_allocated()/1024**3)
        print(torch.cuda.memory_reserved()/1024**3)
        with torch.no_grad():
            self.model.eval()
            print(len(self.val_loader))
            for batch_idx, batch in enumerate(self.val_loader, 1):
                try:
                    print('bs num', batch_idx)
                    # if batch_idx == 3:
                    #     break
                    print(torch.cuda.memory_allocated()/1024**3)
                    print(torch.cuda.memory_reserved()/1024**3)
                    if self.is_augcheck:
                        input_ids, input_mask, labels, ss, os, sent_id, aug_sent_id  = tuple(t.to(self.config.device) for t in batch)
                        id_ls.append(sent_id.detach().cpu().numpy())
                        aug_id_ls.append(aug_sent_id.detach().cpu().numpy())
                        print('id_ls',id_ls)
                        print('sug ls',aug_id_ls)

                    else:
                        input_ids, input_mask, labels, ss, os, _  = tuple(t.to(self.config.device) for t in batch)
                    out, inverse_attn_out, ref_val_out, attn_score, gating = self.model(input_ids=input_ids, attention_mask=input_mask, sub_idx=ss, obj_idx=os, attn_guide=None, device=self.config.device)
                    print(out)
                    loss = loss_func(out, labels)
                    running_loss += loss.detach().cpu()
                    
                    # for analysis
                    pred_ls.append(out.detach().cpu().numpy())
                    answer_ls.append(labels.detach().cpu().numpy())
                    attn_score_ls.append(attn_score.detach().cpu().numpy())
                    gating_ls.append(gating.detach().cpu().numpy())

                    if batch_idx % self.config.log_step == 0:
                        self.logger.info('Validation-Epoch: {} Step: {} Loss: {}'.format(epoch, batch_idx, running_loss/batch_idx))
                
                except Exception as e:
                    self.logger.error(e)
        self.pred_ls_raw = np.concatenate(pred_ls)
        self.pred_ls_ls = [np.argmax(l, axis=1) for l in pred_ls]
        self.pred_ls_ls = np.concatenate(self.pred_ls_ls)
        self.ans_ls_ls = np.concatenate(answer_ls)
        self.attn_score_ls_ls = np.concatenate(attn_score_ls)
        
        assert len(self.pred_ls_ls) == len(self.ans_ls_ls)
        
        p_, r_, f1_, _ = precision_recall_fscore_support(self.ans_ls_ls, self.pred_ls_ls, labels=class_labels, average='micro')
        self.logger.info('epoch: {} - step: {} - precision: {} - recall: {} - f1: {}'.format(epoch, batch_idx, p_, r_, f1_))
        
        p_, r_, f1_, _ = precision_recall_fscore_support(self.ans_ls_ls, self.pred_ls_ls, labels=self.ref_labels_id, average='micro')
        self.logger.info('epoch: {} - step: {} - minor precision: {} - recall: {} - f1: {}'.format(epoch, batch_idx, p_, r_, f1_))
        
        if self.is_test or self.is_augcheck or self.need_attn:
                
            with open(self.config.source + str(batch_idx) + '_' + str(running_loss)[-4:-1] + '_' +  self.logger.name + '_pred_raw.pickle', 'wb') as f_pk:
                pickle.dump(self.pred_ls_raw, f_pk)   

            with open(self.config.source +  str(batch_idx) + '_' + str(running_loss)[-4:-1] + '_' + self.logger.name + '_ans.pickle', 'wb') as f_pk:
                pickle.dump(self.ans_ls_ls, f_pk)   

            with open(self.config.source +   str(batch_idx) + '_' + str(running_loss)[-4:-1] + '_' + self.logger.name + '_pred.pickle', 'wb') as f_pk:
                pickle.dump(self.pred_ls_ls, f_pk) 
                
            with open(self.config.source +   str(batch_idx) + '_' + str(running_loss)[-4:-1] + '_' + self.logger.name + '_attnscore.pickle', 'wb') as f_pk:
                pickle.dump(self.attn_score_ls_ls, f_pk)

        
        if self.is_augcheck:
            self.mc_sent_id = {}
            self.id_ls_ls = np.concatenate(id_ls)
            self.aug_id_ls_ls = np.concatenate(aug_id_ls)
            
    def _get_aug_idx_withfactor(self, aug_num, selected_instance_id):
        """
        aug sentence with weighted average
        aug_num: # aug per sent 
            e.g. {12: 4, 27: 21, 29: 6, 41: 76}
        """
        # select label
        attn_abs = np.abs(self.attn_score_ls_ls)
        selected_instance_idx = {}
        for target in self.ref_labels_id: # MCs label num
            mc_idx = self.ans_ls_ls == target
            sent_ids = selected_instance_id[target]['selected'] # attn ascending order
            aug_target_num = aug_num[target]
            print(aug_target_num, aug_target_num)
            print(sent_ids, len(sent_ids))
            # assert len(sent_ids) == len(aug_target_num)
            self.logger.info('target {} num: {}'.format(target, aug_target_num))
            print(aug_target_num)
            print(sent_ids)
            
            if len(sent_ids) < 1:
                continue
            selected_instance_idx[target] = {"selected": []}
            
            # iter training sentence id attn ascending order
            for j, s_id in enumerate(sent_ids):
                sent_idx = self.id_ls_ls == s_id
                relative_trg_idx = self.ref_labels_id.index(target)
                candidate_instance = self.aug_id_ls_ls[sent_idx] 
                len(candidate_instance)
                
                selected = np.argsort(attn_abs[sent_idx][:, relative_trg_idx])[-aug_target_num:]
                
                selected_instance = self.aug_id_ls_ls[sent_idx][selected]
                assert np.all(np.isin(selected_instance, candidate_instance))
                
                selected_instance_idx[target]["selected"].extend(selected_instance.tolist())
            
        return selected_instance_idx
    
    def _get_train_aug_idx_withfactor(self, target_num):
        """
        - select train instances having high attn score
        - reuturn ordered sentence index (attn descending order)
        - return 
            {12: 24, 27: 2, ...}
        """
        # select label
        attn_abs = np.abs(self.attn_score_ls_ls)
        selected_instance_idx = {}

        for target in self.ref_labels_id: # MCs label num
            selected_instance_idx[target] = {"selected": None, "remains": None, "removed":None}
            relative_trg_idx = self.ref_labels_id.index(target) # [12, ..] 에서 idx
            target_idx = self.ans_ls_ls == target # target idx based on answer idx
            target_len = target_num[target]
            print(target_len)
            candidate_instance = self.id_ls_ls[target_idx] # 실제 instance에서 idx
            len(candidate_instance)

            removed = np.argsort(attn_abs[target_idx][:, relative_trg_idx])[:min(int(np.round(sum(target_idx)*self.config.denoise_r)), int(sum(target_idx)-1))]
            selected = np.argsort(attn_abs[target_idx][:, relative_trg_idx])[-target_len:]
            
            removed_instance = self.id_ls_ls[target_idx][removed]
            selected_instance = self.id_ls_ls[target_idx][selected]
            assert np.all(np.isin(removed_instance, candidate_instance))
            assert np.all(np.isin(selected_instance, candidate_instance))
            
            non_removed = np.isin(candidate_instance, removed_instance)
            non_selected = np.isin(candidate_instance, selected_instance)
            remains_instance = candidate_instance[non_removed == non_selected]
            # assert len(removed_instance) + len(selected_instance) + len(remains_instance) == len(candidate_instance)
            
            selected_instance_idx[target]["selected"] = selected_instance.tolist()
            selected_instance_idx[target]["remains"] = remains_instance.tolist()
            selected_instance_idx[target]["removed"] = removed_instance.tolist()
        
        return selected_instance_idx
        
    def _save_checkpoint(self, epoch, is_additional_train=False):
        arch = type(self.model).__name__
        
        if is_additional_train:
            state = {
                'arch': arch,
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'self.optimizer': None,
                'scheduler': None,
                'config': self.config
                }
            filename = str(self.config.checkpoint_dir + '{}-ad-ckpt-e{}.pth'.format(self.logger.name, epoch))
            
        else:
            state = {
                'arch': arch,
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'self.optimizer': None,
                'scheduler': None,
                'config': self.config
                }
            filename = str(self.config.checkpoint_dir + '{}-ckpt-e{}.pth'.format(self.logger.name, epoch))
            
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints
        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        if not self.is_test and not self.is_augcheck:
            try:
                self.start_epoch = checkpoint['epoch'] + 1
                self.optimizer.load_state_dict(checkpoint['self.optimizer'])
                self.scheduler.load_state_dict(checkpoint['scheduler'])
                self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
            except:
                print("this is additional training step")
