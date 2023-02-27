## Trained retrieval models
- We provide our trained [checkpoints](https://oc.cs.uni-saarland.de/owncloud/index.php/s/cZ5Jb5kCRkcmRnm). In addition to all [data](https://drive.google.com/drive/folders/1xbSzefjPm4Ii5WQSKX2C5wT5MydBkqcT?usp=sharing).
- If you need to re-train the models, please refer to the KGAT's [repository](https://github.com/thunlp/KernelGAT) for more details. Training data (in the same format as KGAT) is in *retrieval_training* under *raw_data* directory.

## Running the retrieval model without attacks

- This step retrieves evidence given claims (baseline - no attacks). It yields the files in *preattack_retrieval_formatted* data directory. 
- The output is used for predictions later (baseline), or to run the attacks on the retrieved results.

```
python test.py --outdir ./output/ \
--test_path <data_dir>/raw_data/all_dev.json \
--bert_pretrain ../bert_base \
--checkpoint ../checkpoint/retrieval_model/model.best.pt \
--name dev.json

python process_data.py --retrieval_file ./output/dev.json \
--gold_file <data_dir>/raw_data/golden_dev.json \
--output <data_dir>/preattack_retrieval_formatted/bert_eval2.json --test
```
- *golden_dev.json* is under *raw_data* data directory.
- Replace with your required checkpoint. 

## Running the retrieval model with attacks under evidence "replace" assumption
- This step first replaces the original evidence with the attack evidence. This is done for all top-5 evidence sentences. Then it runs the retrieval. 
- As an example, for the *imperceptible_homoglyph* attack:
```
python test_replace_check_top5.py \
--test_path_preattack <data_dir>/preattack_retrieval_formatted/bert_eval2.json \
--test_path_postattack <data_dir>/attacks_out_formatted/imperceptible/imperceptible_homoglyph_kgat.json \ 
--name dev_imperceptible_homoglyph_intermediate.json \
--test_path_all_data <data_dir>/raw_data/all_dev.json
```
- *test_path_preattack* is the no-attack retrieval that was used as an input to the attack, under *preattack_retrieval_formatted* data directory. 
- *test_path_postattack* is the attack output, formatted in the same way as KGAT, under *attacks_out_formatted* data directory.

- Then run:
```
python process_data_replace.py --test \
--retrieval_file_orig <data_dir>/preattack_retrieval_formatted/bert_eval2.json \
--retrieval_file_attack ./output/dev_imperceptible_homoglyph_intermediate.json \
--retrieval_file_orig_attacked <data_dir>/attacks_out_formatted/imperceptible_homoglyph_kgat.json \
--gold_file <data_dir>/golden_dev.json \
--output <data_dir>/postattack_retrieval/imperceptible_homoglyph_retrieved_processed.json 
```
- The output of *process_data_replace.py* is the retrieval after the attack, used further in the verification step, found under *postattack_retrieval* data directory.

## Running the retrieval model with attacks under evidence "replace" assumption - Replace only *n* sentences
```
python test_replace_n_check_top5.py \
--test_path_preattack <data_dir>/preattack_retrieval_formatted/bert_eval2.json \
--test_path_postattack <data_dir>/attacks_out_formatted/imperceptible/imperceptible_homoglyph_kgat.json \
--name dev_n1_imperceptible_homoglyph_intermediate.json \
--test_path_all_data <data_dir>/raw_data/all_dev.json --num_sent 1 

python process_data_replace_n.py --test \
--retrieval_file_orig <data_dir>/preattack_retrieval_formatted/bert_eval2.json \
--retrieval_file_attack ./output/dev_n1_imperceptible_homoglyph_intermediate.json \
--retrieval_file_orig_attacked <data_dir>/attacks_out_formatted/imperceptible/imperceptible_homoglyph_kgat.json \
--gold_file <data_dir>/raw_data/golden_dev.json \
--output <data_dir>/postattack_retrieval/imperceptible/imperceptible_homoglyph_n1_retrieved_processed.json 
```

## Running the retrieval model with attacks under evidence "add" assumption
- This step first *adds* the attack evidence to the original evidence. Then it runs the retrieval. 
- Default number of added sentences is 2.

```
python test_add_check_top5.py \
--test_path_postattack <data_dir>/attacks_out_formatted/supporting_generation/gpt2_generate_refutes_nei_supports_model_trials250_sent2_kgat.json \
--test_path_preattack <data_dir>/preattack_retrieval_formatted/bert_eval2.json \
--test_path_all_data <data_dir>/raw_data/all_dev.json \
--name dev_add_gpt2_generate_refutes_nei_supports_model_trials250_sent2.json
```

python process_data_add.py --test \
--retrieval_file_orig <data_dir>/preattack_retrieval_formatted/bert_eval2.json \
--retrieval_file_attack ./output/dev_add_gpt2_generate_refutes_nei_supports_model_trials250_sent2.json \
--retrieval_file_orig_attacked <data_dir>/attacks_out_formatted/supporting_generation/gpt2_generate_refutes_nei_supports_model_trials250_sent2_kgat.json \
--gold_file <data_dir>/raw_data/golden_dev.json \
--output <data_dir>/postattack_retrieval/gpt2_generate/gpt2_generate_refutes_nei_supports_model_trials250_sent2_retrieved_processed.json 
```

## Running the retrieval model with attacks under evidence "add" assumption - Add *n* sentences only





