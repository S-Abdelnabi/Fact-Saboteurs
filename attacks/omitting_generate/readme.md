## Omitting Generate 

<p align="center">
<img src="https://github.com/S-Abdelnabi/Fact-Saboteurs/blob/main/attacks/omitting_paraphrase/omitting.PNG" width="400">
</p>

<p align="center">
<img src="https://github.com/S-Abdelnabi/Fact-Saboteurs/blob/main/attacks/omitting_generate/omitting_generate_examples.PNG" width="950">
</p>

- This attack uses a GPT-2 model that was trained to generate supporting sentences. It creates alternative evidence sentences given the original one as a context. 

-  -  - 
### Model training 
- All models checkpoints are available [here](https://oc.cs.uni-saarland.de/owncloud/index.php/s/yTxPtwNHzp3fzM2).
- We train a GPT-2 model to generate supporting evidence given claims. 
- For environment setup, check this [repository](https://github.com/copenlu/fever-adversarial-attacks/). 
- To train the GPT-2 model, run:

```
python train_gpt2_model.py \
  --dataset_loc <data_dir>/all_data/raw_data/all_train.json \
  --val_dataset <data_dir>/all_data/raw_data/all_dev.json \
  --train_pct 1.0 \
  --n_gpu 1 \
  --n_epochs 20 \
  --seed 1000 \
  --model_dir <gpt2_model_dir>/supports \
  --batch_size 4 \
  --lr 0.00003 \
  --target_class "SUPPORTS" \
  --run_name supports \
  --tags "gpt2 training supports"
  ```
- To train the GPT-2 model with 25% data subset, run:

```
python train_gpt2_model.py \
  --dataset_loc <data_dir>/all_data/raw_data/train_support_25subset.json \
  --val_dataset <data_dir>/all_data/raw_data/dev_support_25subset.json \
  --train_pct 1.0 \
  --n_gpu 1 \
  --n_epochs 20 \
  --seed 1000 \
  --model_dir <gpt2_model_dir>/supports_25subset \
  --batch_size 4 \
  --lr 0.00003 \
  --target_class "SUPPORTS" \
  --run_name supports \
  --tags "gpt2 training supports 25 subsets"
```
- To train the GPT-2 model with 10% data subset, run:

python train_gpt2_model.py \
  --dataset_loc <data_dir>/all_data/raw_data/train_support_10subset.json \
  --val_dataset <data_dir>/all_data/raw_data/dev_support_10subset.json \
  --train_pct 1.0 \
  --n_gpu 1 \
  --n_epochs 20 \
  --seed 1000 \
  --model_dir <gpt2_model_dir>/supports_10subset \
  --batch_size 4 \
  --lr 0.00003 \
  --target_class "SUPPORTS" \
  --run_name supports \
  --tags "gpt2 training supports 10 subsets"
  

-  -  -  
### Evidence generation 

- We then use the GPT-2 model to generate alternative evidence given the original evidence (should have some similarity in context).
- To generate alternative evidence sentences, run:

```
python generate_gpt2_sentences.py \
  --dataset_loc <data_dir>/all_data/preattack_retrieval_formatted/bert_eval2.json \
  --model_loc <gpt2_model_dir>/model.pth\
  --target_class REFUTES SUPPORTS --n_sent 20 --n_evidence 5 \
  --outfile <gpt2_output_dir>/attacks_out_alternative_evidence 
 ```
-  -  - 
### Evidence candidates filtering  

- We then pick the sentence that is the least relevant to the claim (should ideally omit the parts needed to verify the claim).
- To sort based on the retrieval model, run:
```
export PYTHONPATH=<kgat_dir>/retrieval_model

conda activate <kgat_env> 

python retrieval_checker.py --outdir ./output/ \
--test_path <gpt2_output_dir>/attacks_out_alternative_evidence \
--bert_pretrain <kgat_dir>/bert_base \
--checkpoint <kgat_dir>/checkpoint/retrieval_model/model.best.pt \
--name alternative_evidence_supports_refutes_retrieval_check.json \
--all_data <data_dir>/all_data/preattack_retrieval_formatted/bert_eval2.json \
--evi_num 5
```

-  -  -  
 - To convert to KGAT format, run:
```
python convert_to_kgat_evidence_alternative.py \
--infile_orig <data_dir>/all_data/preattack_retrieval_formatted/bert_eval2.json \
--infile_attack_pairs <gpt2_output_dir>/alternative_evidence_supports_refutes_retrieval_check.json \
--outfile <data_dir>/all_data/attacks_out_formatted/alternative_evidence/alternative_evidence_wb_retrieval_check_kgat.json
```
