## Prepare data

- Download TACRED data from [here](https://catalog.ldc.upenn.edu/LDC2018T24)
- Put TACRED data into `data` dir
- Execute the following commands:
```
python3 preprocess_data.py --file-in ./data/train.json --fn-out train
python3 preprocess_data.py --file-in ./data/dev.json --fn-out dev
python3 preprocess_data.py --file-in ./data/test.json --fn-out test
```

## Train

- Execute the following commands:
```
python3 run_tacred.py --n-ref 4 --experiments-name ref4 --data-pkl-path ./data/train.pkl --val-data-pkl-path ./data/dev.pkl
```

## Additional Training

- Prepare synthetic dataset (refer to `./../syn_gen`)
- Execute the following commands
```
# e.g.
python3 finetune_mcam.py --n-ref 4 --resume ./checkpoint/ref4_6-checkpoint-epoch1.pth --seed 6 --refval-ep 0.37 --main-ep 0.37 --select-r 0.63 --factor 8 --aug-data-path ./data/merge-tacred-gen-roberta-base_sub01_gen300_tau15-2022-02-11_type_inserted_2022-01-11.json --suffix sub01tau15
```

## Test

### Challenge RE
- Download `Challenge RE` from [here](https://github.com/shacharosn/CRE)
- Execute the preprocessing script as described in `Prepare data`
```
python3 preprocess_data.py --file-in ./data/challenge_set.json --fn-out cre_test
```
### TACRED-revisited
- Patch labels using `TACRED-revisited` from [here](https://github.com/DFKI-NLP/tacrev)
- Those revised labels will be used
- Test sample weight in `DFKI-NLP/tacrev/tree/master/notebooks/tables.ipynb` is used for `weighted F1 score`
    - summarized weight information about test samples is in `/data/rev_weight.csv`
