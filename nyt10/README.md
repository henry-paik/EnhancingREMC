- Disclaimer: This is the raw files. Please use at your own risk.

---

# Prepare data
- Download NYT10m data from `https://github.com/thunlp/OpenNRE`
- Please refer to `https://arxiv.org/pdf/2005.01898.pdf` for the data details.
- Put NYT10m data into `data` dir
- Execute the following commands
```
python3 preprocess_data.py --file-in ./data/nyt10m_train.txt --fn-out train
python3 preprocess_data.py --file-in ./data/nyt10m_val.txt --fn-out val
python3 preprocess_data.py --file-in ./data/nyt10m_test.txt --fn-out test
```

# Train
- Execute the following commands
```
python3 ds_run.py --n-ref 6 --experiments-name ref6-ds
```

# Additional Traning
- Prepare synthetic dataset
- Execute the following commands
```
# e.g.
python3 ds_finetune_mcam.py --n-ref 6 --resume ./checkpoint/ref6-ds_10_72_2e-05_48-ckpt-e1.pth --train-data-path ./data/train.pkl --aug-data-path ./data/merge-nyt10m-gen300-bert-base-increment-p1oksub02_gen300_tau15_fileseed-2022-05-26_aug_2022-05-26.pkl
```

# Test
- Validation code is not provided.
