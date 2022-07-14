import argparse
from bert_data_util.meta_data import MINOR_LABEL_IDS, REF_SENT, LABEL_TO_ID
from bert_data_util.dataset import CustomDataset, collate_fn
from torch.utils.data import DataLoader
from utils import get_ref_inputids, get_logger, set_deterministic
from bert_model.model import NormalRE
from transformers import BertModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
import pickle
from bert_trainer.trainer import Trainer
import datetime

def main(config):
    set_deterministic(config.seed)
    # logger
    logger = get_logger(config.experiments_name + '_' + str(config.weight ) +'_' + str(config.seed)+'_'+str(config.lr)+'_'+str(config.bs))
    print(logger.name)
    logger.info(config)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    labels = list(range(len(LABEL_TO_ID)))
    labels.pop(0)
    
    ref_labels_id = MINOR_LABEL_IDS[:config.n_ref]
    ref_labels_id = sorted(ref_labels_id)
    ref_sent = [REF_SENT[i] for i in ref_labels_id]
    logger.info(ref_labels_id)
    logger.info(ref_sent)
    
    # data
    print('load data')
    train_dataset = CustomDataset(config.data_pkl_path, max_len=config.max_len, ref_labels_id=ref_labels_id)
    data_loader = DataLoader(train_dataset, config.bs, shuffle=True, collate_fn=collate_fn)
    val_dataset = CustomDataset(config.val_data_pkl_path, max_len=config.max_len, ref_labels_id=ref_labels_id)
    val_loader = DataLoader(val_dataset, config.val_bs, collate_fn=collate_fn)
    
    # model
    ref_input_ids, ref_mask = get_ref_inputids(tokenizer=tokenizer, ref_sent=ref_sent)
    model = NormalRE(n_class=len(LABEL_TO_ID), ref_input_ids=ref_input_ids, ref_mask=ref_mask, hidden_size=768, PRE_TRAINED_MODEL_NAME='bert-base-uncased')
    logger.info(model)

    # optimizer, lr
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 0.0001},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.lr, eps=1e-10)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(len(data_loader)/100), num_training_steps=config.total_epochs*len(data_loader)
    )

    trainer = Trainer(model=model, 
                        logger=logger, 
                        optimizer=optimizer, 
                        scheduler=scheduler, 
                        data_loader=data_loader, 
                        val_loader=val_loader,
                        ref_labels_id=ref_labels_id,
                        config=config)
    trainer.train(is_weighted=True, w=config.weight)

if __name__ == "__main__":
    datestr = datetime.datetime.strftime(datetime.datetime.now(), '%y-%m-%d')
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=72, help='random seed')
    parser.add_argument("--refval-ep", type=float, default=0, help='ep for soft label loss of reference vector output')
    parser.add_argument("--main-ep", type=float, default=0, help='ep for soft label loss of main output')
    parser.add_argument("--weight", type=float, default=10, help='MC weight')
    parser.add_argument("--n-ref", type=int, default=6, help='How many reference vector needs')
    parser.add_argument("--max-len", type=int, default=256, help='predefined max_len which is used for preprocessing.')
    parser.add_argument("--experiments-name", type=str, default='reimplement', help='prefix for saving files')
    parser.add_argument("--resume", type=str, default=None, help='if you want to continue train from the checkpoint, enter checkpoint path.')
    parser.add_argument("--need_metric_score", type=bool, default=True, help='compute answer f1 and evidence f1 score')
    parser.add_argument("--log-step", type=int, default=100, help='save log per step')
    parser.add_argument("--checkpoint-dir", type=str, default='./checkpoint/', help='checkpoint dir')
    parser.add_argument("--device", type=str, default='cuda', help='project name to save file')
    parser.add_argument("--bs", type=int, default=48, help='batch size')
    parser.add_argument("--val-bs", type=int, default=100, help='validation batch size')
    parser.add_argument("--total-epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-05, help='learning rate')
    parser.add_argument("--data-pkl-path", default='./data/train.pkl', type=str, help='preprocessed data. should be pickle file.')
    parser.add_argument("--val-data-pkl-path", default='./data/val.pkl', type=str, help='preprocessed data. should be pickle file.')
    
    args = parser.parse_args()

    main(args)
