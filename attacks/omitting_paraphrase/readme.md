## Omitting Paraphrase 

<p align="center">
<img src="https://github.com/S-Abdelnabi/Fact-Saboteurs/blob/main/attacks/omitting_paraphrase/omitting.PNG" width="400">
</p>

<p align="center">
<img src="https://github.com/S-Abdelnabi/Fact-Saboteurs/blob/main/attacks/omitting_paraphrase/omitting_examples.PNG" width="950">
</p>

- This attacks uses an off-the-shelf model to create paraphrases for the evidence. 

- - - 

### Create paraphrases

```
python generate_paraphrases.py \
  --dataset_loc <data_dir>/all_data/preattack_retrieval_formatted/bert_eval2.json \
  --target_class "SUPPORTS" "REFUTES" \
  --outfile paraphrases_supports_refutes_wb_n5 --n_evi 5 
```
- This step creates paraphrases for evidence sentences.
- *n_evi* sets how many evidence sentences to create paraphrases for.
- - - 

### Sort with the retrieval model 
```
export PYTHONPATH=<kgat_dir>/retrieval_model

python retrieval_checker.py --outdir ./output/ \
--test_path paraphrases_supports_refutes_wb_n5 --evi_num 5 \
--bert_pretrain <kgat_dir>/bert_base \
--checkpoint  <kgat_dir>/retrieval_model/model.best.pt \
--name paraphrases_supports_refutes_wb_n5_retrieval_check.json
```
- This step selects the paraphrase that has the least retrieval score compared to the claim.
- - - 

### Convert to KGAT format 

```
python convert_to_kgat_evidence_paraphrase.py \
--infile_orig  <data_dir>/all_data/preattack_retrieval_formatted/bert_eval2.json \
--infile_attack_pairs output/paraphrases_supports_refutes_wb_retrieval_check.json \
--outfile <data_dir>/all_data/attacks_out_formatted/paraphrase_evidence/paraphrases_wb_retrieval_check_kgat.json
```
- This step converts the output to KGAT format.

