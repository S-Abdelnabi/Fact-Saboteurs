## Contextualized replace
- This attack is based on the BERT attack paper. Please refer to the authors' [repository](https://github.com/LinyangLee/BERT-Attack) for setup details. You need to compute the *cos_sim_counter_fitting.npy* file - we don't share this because it's a 16GB file :)
- We adjust the original code to work on entailment (pairs for <claim, evidence>).
- To calculate the attack, we use the [BERT stance verification model](https://github.com/S-Abdelnabi/Fact-Saboteurs/tree/main/stance_verification). 
- This attack edits the top retrieved sentences by the attacker's retrieval model.

---

- To run the attack:

```
python bert_attack.py \
--data_path <data_dir>/preattack_retrieval_pairs/eval_pairs_retrieval2 \
--mlm_path bert-base-uncased \
--tgt_path <checkpoint_dir>/judge_model_chpt/model.best.pt \
--use_sim_mat 1 --num_label 3 --use_bpe 1 --k 48 \
--output_dir <data_dir>/attacks_out_pairs/attack_out \
--start 0 --end -1 \
--threshold_pred_score 1.0e-5 --word_budget 0.15
```
- The *data_path* file is pairs of claims and evidence with their labels - The labels are those of the original claim. These are the sentences retrieved by the attacker's verification model. 
- The *mlm_path* is the huggingface model for masked language modeling.
- The *tgt_path* is the trained stance verification model.
- *output_dir* is the directory to save the output. It is saved as pairs of claims, attack sentences. 

---
- To convert the attack's output to the format used by KGAT, run:

```
python convert_to_kgat.py \
--infile_attack_pairs <data_dir>/attacks_out_pairs/attack_out \
--infile_orig bert_eval2.json \
--outfile <data_dir>/attacks_out_formatted/bert_attack1_kgat.json
```
- *infile_orig* is the attacker's retrieval output before converting it to pairs
