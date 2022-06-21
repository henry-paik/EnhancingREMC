import argparse
from data_util.meta_data import MINOR_LABEL_IDS, REF_SENT
from data_util.augcheck_dataset import TrainAugcheckCustomDataset, AugdataAugcheckCustomDataset, AugCustomDataset, collate_fn_select_aug
from data_util.dataset import CustomDataset, collate_fn
from torch.utils.data import DataLoader
from utils import get_ref_inputids, get_logger, set_deterministic, calc_aug_data_num
from model.model import NormalRE
from transformers import  AutoTokenizer, AdamW
from trainer.trainer import Trainer
import datetime
import torch

def main(config):
    # logger
    set_deterministic(config.seed)
    logger = get_logger(config.resume.split('/')[-1] + '_with_factor_' + str(config.factor) + '_seed_' + str(config.seed) + config.suffix + '_' + config.additional_train_type)
    tokenizer = AutoTokenizer.from_pretrained('roberta-large')

    # model
    ref_labels_id = MINOR_LABEL_IDS[:config.n_ref]
    ref_labels_id = sorted(ref_labels_id)
    ref_sent = [REF_SENT[i] for i in ref_labels_id]

    ref_input_ids, ref_mask = get_ref_inputids(tokenizer=tokenizer, ref_sent=ref_sent)
    model = NormalRE(n_class=42, ref_input_ids=ref_input_ids, ref_mask=ref_mask, hidden_size=1024, PRE_TRAINED_MODEL_NAME='roberta-large')
    logger.info(model)

    if config.select_r is None:
        val_dataset = CustomDataset(config.train_data_path, labels=ref_labels_id)
        val_loader = DataLoader(val_dataset, config.val_bs, collate_fn=collate_fn, pin_memory=True)
        trainer = Trainer(model=model, 
                    logger=logger, 
                    val_loader=val_loader, 
                    is_test=False,
                    config=config,
                    ref_labels_id=ref_labels_id)
        _v = trainer.get_clean_level()
        config.select_r = _v
        del trainer
        del val_dataset
        del val_loader
        torch.cuda.empty_cache()

    config.refval_ep = 1 - config.select_r
    config.main_ep = 1 - config.select_r
    logger.info(config)

    ############### 1) filter noisy instance; select reliable instance -> selected sentence needs attn rank []
    aug_num, target_num = calc_aug_data_num(num_instance={12:122, 27:23, 29:76, 41:6}, factor=int(config.factor), selected_r=config.select_r)
    print(aug_num)
    print("denoising and select instance")
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
    del train_dataset
    del train_loader
    torch.cuda.empty_cache()

    ############### 2) scoring augmentation of selected instance
    print("scoring aug dataset")
    aug_dataset = AugdataAugcheckCustomDataset(config.aug_data_path, ref_labels_id=ref_labels_id, selected_instance_id=selected_instance_id)
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
    del aug_dataset
    del aug_loader
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
    trainer.additional_train(optimizer=optimizer)

if __name__ == "__main__":
    datestr = datetime.datetime.strftime(datetime.datetime.now(), '%y-%m-%d')
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=81, help='random seed')
    parser.add_argument("--total-epochs", type=int, default=6)
    parser.add_argument("--checkpoint-dir", type=str, default='./checkpoint/', help='checkpoint dir')
    parser.add_argument("--source", type=str, default='./source/', help='path for saving predictiction and answer numpy array')
    parser.add_argument("--factor", type=float, default=4, help='x times augment')
    parser.add_argument("--exp-min-thr", type=float, default=0.6, help='starting threshold for exponential additional train')
    parser.add_argument("--exp-max-thr", type=float, default=0.8, help='last threshold for exponential additional train')
    parser.add_argument("--additional-train-type", type=str, default='exponential', help='stepwise or exponential')
    parser.add_argument("--n-ref", type=int, required=True, help='How many reference vector needs')
    parser.add_argument("--suffix", type=str, default='test', help='suffix for saving files/log')
    parser.add_argument("--resume", type=str, required=True, help='if you want to continue train from the checkpoint, enter checkpoint path.')
    parser.add_argument("--log-step", type=int, default=500, help='save log per step')
    parser.add_argument("--device", type=str, default='cuda', help='project name to save file')
    parser.add_argument("--val-bs", type=int, default=300, help='batch size')
    parser.add_argument("--bs", type=int, default=4, help='batch size')
    parser.add_argument("--lr", type=float, default=5e-6, help='learning rate')
    parser.add_argument("--aug-data-path", default='./data/merge-tacred-gen-roberta-base_sub01_gen300_tau15-2022-03-26_type_inserted_2022-03-26.json', type=str, help='preprocessed augmentation data. json file.')
    parser.add_argument("--train-data-path", default='./data/train.pkl', type=str, help='train dataset for model to score. pkl file.')
    parser.add_argument("--val-data-path", default='./data/dev.pkl', type=str, help='dev data set. pkl file.')
    parser.add_argument("--select-r", default=None, type=float, help='ratio of selecting instances with top attn score; and one of selecting augmentation for each selected instances')
    parser.add_argument("--max-len", type=int, default=512, help='predefined max_len which is used for preprocessing.')
    args = parser.parse_args()

    main(args)
