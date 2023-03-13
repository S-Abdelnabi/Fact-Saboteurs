## Omitting Generate 

<p align="center">
<img src="https://github.com/S-Abdelnabi/Fact-Saboteurs/blob/main/attacks/omitting_paraphrase/omitting.PNG" width="400">
</p>

<p align="center">
<img src="https://github.com/S-Abdelnabi/Fact-Saboteurs/blob/main/attacks/omitting_generate/omitting_generate_examples.PNG" width="950">
</p>

- This attack uses a GPT-2 model that was trained to generate supporting sentences. It creates alternative evidence sentences given the original one as a context. 

-  -  - 

- To generate alternative evidence sentences, run:

```
python generate_gpt2_sentences.py \
  --dataset_loc <data_dir>/all_data/preattack_retrieval_formatted/bert_eval2.json \
  --model_loc <gpt2_model_dir>/model.pth\
  --target_class REFUTES SUPPORTS --n_sent 20 --n_evidence 5 \
  --outfile <gpt2_output_dir>/attacks_out_alternative_evidence 
 ```
 
 -  -  - 

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
