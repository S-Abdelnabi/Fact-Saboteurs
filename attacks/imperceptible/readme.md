## Imperceptible Perturbation attacks 
- For setup, please refer to this [repository](https://github.com/nickboucher/imperceptible).
- We compute the attacks on BERT retrieval or verification models.
- Checkpoints of BERT verification models are [here](https://oc.cs.uni-saarland.de/owncloud/index.php/s/WKqPaHijfBPxoW5).
- Checkpoints of BERT retrieval models are [here](https://oc.cs.uni-saarland.de/owncloud/index.php/s/cZ5Jb5kCRkcmRnm).
- The input and output of attacks are in our [data](https://drive.google.com/drive/folders/1xbSzefjPm4Ii5WQSKX2C5wT5MydBkqcT?usp=sharing) directory.

- - - 

### Imperceptible on a verification model 
- For homoglyphs 
```
python experiment.py -g --pkl_file attack_out_homoglyph --target_model_chkpt <model_dir>/stance_model_chpt/model.best.pt \
--data_path <data_dir>/all_data/preattack_retrieval_pairs/eval_pairs_retrieval2 \
--end_idx -1 --min-perturbs 5 --overwrite --maxiter 3
```

- For delete
```
python experiment.py -d --pkl_file attack_out_delete --target_model_chkpt <model_dir>/stance_model_chpt/model.best.pt  \
--data_path <data_dir>/all_data/preattack_retrieval_pairs/eval_pairs_retrieval2 \
--end_idx -1 --min-perturbs 5 --overwrite --maxiter 3
```

- For swap
```
python experiment.py -r --pkl_file attack_out_iters_replace --target_model_chkpt <model_dir>/stance_model_chpt/model.best.pt  \
--data_path <data_dir>/all_data/preattack_retrieval_pairs/eval_pairs_retrieval2 \
--end_idx -1 --min-perturbs 5 --overwrite --maxiter 3
```
- *target_model_chkpt* is the BERT stance verification model trained on pairs of *<claims,evidence>*. 
- *data_path* is pairs of *<claims,evidence>*. 
- - - 

### Imperceptible on a retrieval model 

```
export PYTHONPATH=<kgat_dir>/retrieval_model

python experiment_retrieval.py -g --pkl_file attack_out_retrieval_homoglyph \
--target_model_chkpt <kgat_models_dir>/retrieval_model/model.best.pt \
--bert_pretrain <kgat_models_dir>/bert_base \
--data_path <data_dir>/all_data/preattack_retrieval_pairs/eval_pairs_retrieval2 \
--min-perturbs 5 --maxiter 3
```

- KGAT uses BERT base models, which can be found [here](https://oc.cs.uni-saarland.de/owncloud/index.php/s/FJW2sNrKXrqmtSe).
- This runs the attack against KGAT retrieval model. 

- - -

### Convert to KGAT format
```
python convert_to_kgat_imperceptible.py \
--infile_preattack_pairs <data_dir>/all_data/preattack_retrieval_pairs/eval_pairs_retrieval2 \
--infile_attack_pairs attack_out_retrieval_homoglyph \
--infile_orig <data_dir>/all_data/preattack_retrieval_formatted/bert_eval2.json \
--outfile <data_dir>/all_data/attacks_out_formatted/imperceptible/imperceptible_retrieval_homoglyph_kgat.json
```
