import argparse
from bert_data_util.meta_data import MINOR_LABEL_IDS, REF_SENT, LABEL_TO_ID
from bert_data_util.augcheck_dataset import TrainAugcheckCustomDataset, AugdataAugcheckCustomDataset, AugCustomDataset, collate_fn_select_aug
from bert_data_util.dataset import CustomDataset, collate_fn
from torch.utils.data import DataLoader
from utils import get_ref_inputids, get_logger, set_deterministic, calc_aug_data_num
from bert_model.model import NormalRE
from transformers import AutoTokenizer, AdamW
from bert_trainer.trainer import Trainer
import datetime
import torch

def main(config):
    # logger
    set_deterministic(config.seed)
    logger = get_logger(config.resume.split('/')[-1] + '_f_' + str(config.factor) + 'w' + str(config.weight) + '_seed_' + str(config.seed) + '_' + config.suffix + '_' +str(config.bs)+ '_' +str(config.lr))
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    logger.info(config)
    ref_labels_id = MINOR_LABEL_IDS[:config.n_ref]
    ref_labels_id = sorted(ref_labels_id)
    ref_sent = [REF_SENT[i] for i in ref_labels_id]
    
    ref_input_ids, ref_mask = get_ref_inputids(tokenizer=tokenizer, ref_sent=ref_sent)
    ref_input_ids, ref_mask = get_ref_inputids(tokenizer=tokenizer, ref_sent=ref_sent)
    model = NormalRE(n_class=len(LABEL_TO_ID), ref_input_ids=ref_input_ids, ref_mask=ref_mask, hidden_size=768, PRE_TRAINED_MODEL_NAME='bert-base-uncased')
    logger.info(model)


    ############### 1) filter noisy instance; select reliable instance -> selected sentence needs attn rank []
    num_instance = {17: 60, 19: 6, 20: 90, 21: 16, 22: 3, 24: 21}
    aug_num, target_num = calc_aug_data_num(num_instance={k: v for k, v in num_instance.items() if k in ref_labels_id}, factor=int(config.factor), selected_r=config.select_r)
    logger.info('aug num {}'.format(aug_num))
    logger.info('target num {}'.format(target_num))

    logger.info("denoising and select instance")
    train_dataset = TrainAugcheckCustomDataset(config.train_data_path, ref_labels_id=ref_labels_id)
    train_loader = DataLoader(train_dataset, config.val_bs, shuffle=False, collate_fn=collate_fn_select_aug, pin_memory=True)
    trainer = Trainer(model=model, 
                        logger=logger, 
                        ref_labels_id=ref_labels_id,
                        val_loader=train_loader, 
                        is_augcheck=True,
                        config=config)
    selected_instance_id = trainer.select_train_instance_withfactor(target_num) # denoise suspected noisy instances
                                           # select reliable instances
                                           # reuturn selected and remains instances
    logger.info('selected instances: {}'.format(selected_instance_id))
    logger.info('stats: {}'.format({k: len(i["selected"]) for k, i in selected_instance_id.items()}))
    del trainer
    torch.cuda.empty_cache()
    
    ############### 2) scoring augmentation of selected instance
    print("scoring aug dataset")
    aug_dataset = AugdataAugcheckCustomDataset(config.aug_data_path, ref_labels_id=ref_labels_id, selected_instance_id=selected_instance_id)
    print(len(aug_dataset))
    aug_loader = DataLoader(aug_dataset, config.val_bs, shuffle=False, collate_fn=collate_fn_select_aug, pin_memory=True)
    trainer = Trainer(model=model, 
                        logger=logger, 
                        ref_labels_id=ref_labels_id,
                        val_loader=aug_loader, 
                        is_augcheck=True,
                        config=config)
    aug_selected_instance_idx = trainer.select_aug_instance_withfactor(aug_num, selected_instance_id)
    logger.info('selected instances from aug: {}'.format(aug_selected_instance_idx))
    logger.info('stats: {}'.format({k: len(i["selected"]) for k, i in aug_selected_instance_idx.items()}))
    del trainer
    torch.cuda.empty_cache()
    
    
    ############### 3) additional train with augdataset
    print("start additional train")
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 0.0001},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.lr, eps=1e-10)
    augtrain_dataset = AugCustomDataset(train_path=config.train_data_path, aug_path=config.aug_data_path, selected_instance_idx=selected_instance_id, aug_s_i_idx=aug_selected_instance_idx, ref_labels_id=ref_labels_id)
    augtrain_loader = DataLoader(augtrain_dataset, config.bs, shuffle=True, collate_fn=collate_fn, pin_memory=True)
    logger.info("total aug dataset len: {} - aug datasetlen : {}".format(len(augtrain_dataset.data), len(augtrain_dataset.aug_data)))
    
    val_dataset = CustomDataset(config.val_data_path, ref_labels_id)
    val_loader = DataLoader(val_dataset, config.val_bs, collate_fn=collate_fn, pin_memory=True)
    trainer = Trainer(model=model, 
                    logger=logger, 
                    data_loader=augtrain_loader, 
                    val_loader=val_loader,
                    ref_labels_id=ref_labels_id,
                    config=config)
    trainer.additional_train(optimizer=optimizer, is_weighted=True, w=config.weight)

if __name__ == "__main__":
    datestr = datetime.datetime.strftime(datetime.datetime.now(), '%y-%m-%d')
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-name", type=str, default='augmentation', help='random seed')
    parser.add_argument("--seed", type=int, default=72, help='random seed')
    parser.add_argument("--total-epochs", type=int, default=6)
    parser.add_argument("--checkpoint-dir", type=str, default='./checkpoint/', help='checkpoint dir')
    parser.add_argument("--source", type=str, default='./source/', help='path for saving predictiction and answer numpy array')
    parser.add_argument("--refval-ep", type=float, default=0.7, help='ep for soft label loss of reference vector output')
    parser.add_argument("--main-ep", type=float, default=0.7, help='ep for soft label loss of main output')
    parser.add_argument("--exp-min-thr", type=float, default=0.6, help='starting threshold for exponential additional train')
    parser.add_argument("--exp-max-thr", type=float, default=0.8, help='last threshold for exponential additional train')
    parser.add_argument("--additional-train-type", type=str, default='exponential', help='stepwise or exponential')
    parser.add_argument("--n-ref", type=int, required=True, help='How many reference vector needs')
    parser.add_argument("--suffix", type=str, default='-ad', help='suffix for saving files/log')
    parser.add_argument("--resume", type=str, required=True, help='if you want to continue train from the checkpoint, enter checkpoint path.')
    parser.add_argument("--tokenizer-path", type=str, default='./roberta-large-tokenizer.pkl', help='path to load tokenizer pickle file')
    parser.add_argument("--log-step", type=int, default=500, help='save log per step')
    parser.add_argument("--device", type=str, default='cuda', help='project name to save file')
    parser.add_argument("--pjt-name", type=str, default='ds_recovery'+datestr, help='project name to save file')
    parser.add_argument("--val-bs", type=int, default=60, help='batch size')
    parser.add_argument("--bs", type=int, default=48, help='batch size')
    parser.add_argument("--lr", type=float, default=5e-06, help='learning rate')
    parser.add_argument("--aug-data-path", required=True, type=str, help='preprocessed augmentation data. json file.')
    parser.add_argument("--train-data-path", required=True, type=str, help='train dataset for model to score. pkl file.')
    parser.add_argument("--val-data-path", default='./data/val.pkl', type=str, help='dev data set. pkl file.')
    parser.add_argument("--denoise-r", default=0, type=float, help='ratio of removing instances with bottom attn score')
    parser.add_argument("--select-r", default=0.3, type=float, help='ratio of selecting instances with top attn score; and one of selecting augmentation for each selected instances')
    parser.add_argument("--max-len", type=int, default=260, help='predefined max_len which is used for preprocessing.')
    parser.add_argument("--factor", type=int, default=8, help='aug size factor')
    parser.add_argument("--weight", type=float, default=30, help='MC weight')

    args = parser.parse_args()

    main(args)
