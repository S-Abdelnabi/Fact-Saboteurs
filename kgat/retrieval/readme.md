## Trained retrieval models
- We provide our trained checkpoints: [Link](https://oc.cs.uni-saarland.de/owncloud/index.php/s/cZ5Jb5kCRkcmRnm). In addition to all [data](https://drive.google.com/drive/folders/1xbSzefjPm4Ii5WQSKX2C5wT5MydBkqcT?usp=sharing).
- If you need to re-train the models, please refer to the KGAT's [repository](https://github.com/thunlp/KernelGAT) for more details. Training data (in the same format as KGAT) is in *retrieval_training* under *raw_data* directory.

## Running the retrieval model without attacks

- This step retrieves evidence given claims (baseline - no attacks). It yields the files in *preattack_retrieval_formatted*. 
- The output is used for predictions later (baseline), or to run the attacks on the retrieved results.

```
python test.py --outdir ./output/ \
--test_path ../data/all_dev.json \
--bert_pretrain ../bert_base \
--checkpoint ../checkpoint/retrieval_model/model.best.pt \
--name dev.json

python process_data.py --retrieval_file ./output/dev.json --gold_file <data_dir>/golden_dev.json --output <data_dir>/bert_eval2.json --test
```
- *golden_dev.json* is under *raw_data* data directory.
- Replace with your required checkpoint. 

## Running the retrieval model with attacks under evidence "replace" assumption
- As an example, for the *imperceptible_homoglyph* attack:
```
python test_replace_check_top5.py --test_path_postattack <attack_formatted_file> --name dev_intermediate.json

python process_data_replace.py --test --retrieval_file_orig <data_dir>/bert_eval2.json --retrieval_file_attack ./output/dev_intermediate.json --gold_file <data_dir>/golden_dev.json \
--retrieval_file_orig_attacked <data_dir>/imperceptible_homoglyph_kgat.json \
--output attack_retrieved_processed.json \
```

- This step first replaces the original evidence with the attack evidence. This is done for all top-5 evidence sentences. Then it runs the retrieval. 
- *retrieval_file_orig* is the no-attack retrieval, under *preattack_retrieval_formatted* data directory. 
- *retrieval_file_orig_attacked* is the attack output, formatted in the same way as KGAT, under *attacks_out_formatted* data directory.
- The output of *process_data_replace.py* is the retrieval after the attack, used further in the verification step, under *postattack_retrieval* data directory.







