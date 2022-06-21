from transformers import RobertaTokenizer, RobertaForMaskedLM, AdamW, get_linear_schedule_with_warmup
from transformers import LineByLineTextDataset
from torch.utils.data import DataLoader, Dataset
from transformers import DataCollatorForLanguageModeling
import torch
import logging
import time
import torch.nn as nn
import argparse

def get_logger(name):
    
    # init
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter\
        ('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter(
        fmt='%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s', datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # handler: file
    timestr = time.strftime('%Y%m%d_%H:%M:%S')
    file_name = 'log/{}_{}.log'.format(name, timestr)

    file_handler = logging.FileHandler(file_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # wrap
    logger.handlers.clear()
    logger.addHandler(file_handler)
    return logger

def validation(val_loader, model, logger, epoch, step):
    model.eval()

    with torch.no_grad():
        val_loss = 0
        loss_func = nn.CrossEntropyLoss()
        for i, val_batch in enumerate(val_loader):
            try:
                input_id, label = tuple(t.to(device) for t in [val_batch['input_ids'], val_batch['labels']])
                out = model(input_id)
                loss = loss_func(out[0].view(-1, len(tokenizer)), label.view(-1))
                val_loss += loss

            except Exception as e:
                print(e)

        logger.info('epoch: {} - step: {} - validation loss: {}'.format(epoch, step, val_loss/len(val_loader)))
    return val_loss/len(val_loader)

def train(model, loader, val_loader, device, f, total_epoch, lr=1e-5, j=0,):
    ####################### scheduler
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-6)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(len(loader)/100), num_training_steps=int(len(loader)*total_epoch)
    )
    ####################### scheduler
    model = model.to(device)
    logger.info(model)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(total_epoch):
        model.train()
        running_loss = 0
        j+=1
        for step, batch in enumerate(loader, 1):
            if step==3:
                break
            print(batch)
            input_id, label = tuple(t.to(device) for t in [batch['input_ids'], batch['labels']])
            optimizer.zero_grad()
            out = model(input_id)
            
            loss = loss_func(out[0].view(-1, len(tokenizer)), label.view(-1))
            loss.backward()
            optimizer.step()
            current_lr = scheduler.get_last_lr()[0]
            scheduler.step()
            running_loss += loss.detach().cpu()
            
            if step % 10 == 0:
                logger.info('lr {}-epochs {}-iter {}-loss avg {}'.format(current_lr, epoch, step, running_loss/step))
        # val at the end of each epoch
        model.save_pretrained('./checkpoint/' + f + '_' + str(j) + '_' + str(step) + '.pt')        
        val_loss = validation(val_loader, model, logger, epoch, step)
        logger.info('val-loss: {}'.format(val_loss))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=15, help='batch size')
    parser.add_argument("--out", type=str, default='tacred-roberta-base-mlm', help='output name')
    parser.add_argument("--total-epochs", type=int, default='tacred-roberta-base-mlm', help='epochs')
    parser.add_argument("--train-data", type=str, default='./data/tacred-mlm-plain-train.txt', help='train data path')
    parser.add_argument("--dev-data", type=str, default='./data/tacred-mlm-plain-dev.txt', help='dev data path')
    parser.add_argument("--block-size", type=int, default=370, help='max len for MLM')
    args = parser.parse_args()

    #################################### 

    ###### model
    model = RobertaForMaskedLM.from_pretrained('roberta-base')

    ###### tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    # load dataset
    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=args.train_data,
        block_size=args.block_size,
    )
    dev_dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=args.dev_data,
        block_size=args.block_size
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    train_loader = DataLoader(dataset, collate_fn=data_collator, batch_size=args.bs)
    dev_loader = DataLoader(dev_dataset, collate_fn=data_collator, batch_size=args.bs)

    logger = get_logger(args.out)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train(model, train_loader, dev_loader, device, total_epoch=args.total_epochs, j=0, f=args.out)
