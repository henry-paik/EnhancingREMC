# TACRED
## Prepare mlm data
- prepare plain text training data 
- run 
```
python3 tacred_gen_plain_train_data.py --train-data ./../tacred/data/train.json --out tacred-mlm-plain-train
python3 tacred_gen_plain_train_data.py --train-data ./../tacred/data/dev.json --out tacred-mlm-plain-dev
python3 run_mlm.py --bs 15 --pretrained-model roberta-base --train-data ./data/tacred-mlm-plain-train.txt --dev-data ./data/tacred-mlm-plain-dev.txt --out tacred-roberta-base-mlm --block-size 370 --total-epochs 100
```

## Generate synthetic data
- run for all file seed
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
