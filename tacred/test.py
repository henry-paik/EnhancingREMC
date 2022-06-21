# reference: https://github.com/wzhouad/RE_improved_baseline
# python3 test_run_all_at_once.py --resume ./checkpoint/ref3-data_w_whitespace-checkpoint-epoch4.pth_-augcheck_exponential-addtional-checkpoint-epoch --n-ref 3
import argparse
from data_util.meta_data import LABEL_TO_ID, MINOR_LABEL_IDS, REF_SENT
from data_util.dataset import CustomDataset, collate_fn
from torch.utils.data import DataLoader
from utils import get_ref_inputids, get_logger, get_metric_cre, get_metric_rev, get_metric_ori
from model.model import NormalRE
from transformers import RobertaModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from trainer.trainer import Trainer
import datetime
import pandas as pd

def main(config):
    # logger
    logger = get_logger(config.resume.split('/')[-1] + '_' + 'test_all_at_once')
    logger.info(config)
    tokenizer = AutoTokenizer.from_pretrained('roberta-large')

    # model
    ref_labels_id = MINOR_LABEL_IDS[:config.n_ref]
    ref_labels_id = sorted(ref_labels_id)
    ref_sent = [REF_SENT[i] for i in ref_labels_id]

    ref_input_ids, ref_mask = get_ref_inputids(tokenizer=tokenizer, ref_sent=ref_sent)
    model = NormalRE(n_class=42, ref_input_ids=ref_input_ids, ref_mask=ref_mask, hidden_size=1024, PRE_TRAINED_MODEL_NAME='roberta-large')
    logger.info(model)

    # test score on original tacred dataset
    test_dataset = CustomDataset(config.test_data_path)
    data_loader = DataLoader(test_dataset, config.val_bs, shuffle=False, collate_fn=collate_fn, pin_memory=True)
    trainer = Trainer(model=model, 
                        logger=logger, 
                        val_loader=data_loader, 
                        is_test=True,
                        config=config,
                        ref_labels_id=ref_labels_id)
    trainer.test()
    get_metric_ori(pred=trainer.pred_ls_ls, labels=trainer.ans_ls_ls, ref_labels_id=ref_labels_id, logger=logger)
    
    # test score on revised tacred dataset
    weight = pd.read_csv(config.rev_weight, index_col=0)
    rev_label = weight['true_label_reannotated'].map(lambda x: LABEL_TO_ID[x]).to_numpy()
    w = weight.weight.to_numpy()
    get_metric_rev(pred=trainer.pred_ls_ls, rev_label=rev_label, sample_weight=w, logger=logger)
    
    # test score on CRE
    test_dataset = CustomDataset(config.cre_data_path)
    data_loader = DataLoader(test_dataset, config.val_bs, shuffle=False, collate_fn=collate_fn, pin_memory=True)
    trainer.val_loader = data_loader
    trainer.test()
    get_metric_cre(pred=trainer.pred_ls_ls, challenge_set_path=config.cre_ori_path, logger=logger)

if __name__ == "__main__":
    datestr = datetime.datetime.strftime(datetime.datetime.now(), '%y-%m-%d')
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-data-path", default='./data/test.pkl', type=str, help='preprocessed data. should be pickle file.')
    parser.add_argument("--cre-data-path", default='./data/cre_test.pkl', type=str, help='preprocessed CRE data. should be pickle file.')
    parser.add_argument("--cre-ori-path", default='./data/challenge_set.json', type=str, help='original CRE data. should be pickle file.')
    parser.add_argument("--rev-weight", default='./data/rev_weight.csv', type=str, help='TACRED revisited sample weight')
    parser.add_argument("--source", type=str, default='./source/', help='path for saving predictiction and answer numpy array')
    parser.add_argument("--n-ref", type=int, required=True, help='How many reference vector needs')
    parser.add_argument("--suffix", type=str, default='-test', help='suffix for saving files and log')
    parser.add_argument("--resume", type=str, required=True, help='if you want to continue train from the checkpoint, enter checkpoint path.')
    parser.add_argument("--log-step", type=int, default=100, help='save log per step')
    parser.add_argument("--device", type=str, default='cuda', help='project name to save file')
    parser.add_argument("--val-bs", type=int, default=800, help='test batch size')
    args = parser.parse_args()
    main(args)