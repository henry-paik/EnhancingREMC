# TACRED
## Prepare mlm data
- prepare plain text training data 
- Execute the following commands:
```
python3 tacred_gen_plain_train_data.py --train-data ./../tacred/data/train.json --out tacred-mlm-plain-train
python3 tacred_gen_plain_train_data.py --train-data ./../tacred/data/dev.json --out tacred-mlm-plain-dev
python3 run_mlm.py --bs 15 --pretrained-model roberta-base --train-data ./data/tacred-mlm-plain-train.txt --dev-data ./data/tacred-mlm-plain-dev.txt --out tacred-roberta-base-mlm --block-size 370 --total-epochs 100
```

## Generate synthetic data
- Execute the following commands for all file seed:
```
python3 tacred-gen.py --train-data --train-data ./../nyt10/data/train.json --gen-per-sent 300 --n-sub 0.1 --model-path ./checkpoint/PATH/TO/FINETUNED/MLM --file-seed {1, ...}
```
- merge sperately generate .csv files
	- merge filename should be: merge-{...}.csv
```
cd source
python3 merge_csv.py --part-fn FILE/NAME/BEFORE/FILESEED
python3 merge_csv.py --part-fn tacred-gen-roberta-base_sub01_gen300_tau15
```
- preprocess merged data to align the format as the same as .pkl training data
	- output filename should be: merge-{SUFFIX-STRING}.json
```
cd ..
python3 tacred-preprocess-aug.py --aug-path ./source/merge-{SUFFIX-STRING}.csv
cp ./data/merge-{SUFFIX-STRING}.pkl ./../tacred/data/ 
```

# NYT10m
- Disclaimer: This is the raw files for nyt10m. Please use at your own risk.
## Prepare mlm data
- prepare plain text training data
- Execute the following commands:
```
python3 nyt_gen_plain_train_data.py --train-data ./../nyt10/data/nyt10m_train.txt --out nyt10-mlm-plain-train
python3 nyt_gen_plain_train_data.py --train-data ./../nyt10/data/nyt10m_val.txt --out nyt10-mlm-plain-val
python3 run_mlm.py --bs 48 --pretrained-model bert-base-uncased --train-data ./data/nyt10-mlm-plain-train-sample.txt --dev-data ./data/nyt10-mlm-plain-val-sample.txt --out nyt-bert-base-mlm --block-size 180 --total-epochs 1
```

## Prepare target generation data
- Execute the following commands:
```
python3 nyt-gen-target-mc-data.py --file-in ./../nyt10/data/nyt10m_train.txt
```

## Generate synthetic data
- Execute the following commands:
```
python3 nyt-gen.py --train-data ./data/nyt-10m-6MCgen.pkl --gen-per-sent 300 --n-sub 0.2 --model-path ./checkpoint/PATH/TO/FINETUNED/MLM --file-seed {1, ...}
cd source
python3 merge_csv.py --part-fn FILE/NAME/BEFORE/FILESEED
cd ..
python3 nyt-preprocess-aug.py --aug-path ./source/merge-{SUFFIX-STRING}.csv
cp ./data/merge-{SUFFIX-STRING}.pkl ./../nyt10/data/
```
