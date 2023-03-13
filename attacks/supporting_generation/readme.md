## Supporting Generation

<p align="center">
<img src="https://github.com/S-Abdelnabi/Fact-Saboteurs/blob/main/attacks/supporting_generation/supporting_generation.PNG" width="400">
</p>


<p align="center">
<img src="https://github.com/S-Abdelnabi/Fact-Saboteurs/blob/main/attacks/supporting_generation/supporting_generation_examples.PNG" width="950">
</p>

- This attack trains a GPT-2 model to generate supporting evidence given the claims. 

-  -  - 
### Model training 
- All models checkpoints are available [here](https://oc.cs.uni-saarland.de/owncloud/index.php/s/yTxPtwNHzp3fzM2).
- Check the ["omitting generate"](https://github.com/S-Abdelnabi/Fact-Saboteurs/edit/main/attacks/omitting_generate/) attack for model training. 

-  -  - 
### Generate evidence 
- Generate 2 evidence sentences (*--n_sent*) for each claim. 
- Verify if the evidence supports the claim using the stance verification checkpoint (*--verifier_chkpt*)
- Try 250 samples (*--trials*) to find supporting sentences. This might take a lot of time, you could decrease that number to save time. 

- Convert *refutes* to *supports*: 
```
python generate_gpt2_sentences.py \
  --dataset_loc <data_dir>/all_data/raw_data/all_dev.json \
  --model_loc <gpt2_model_dir>/supports/model.pth\
  --target_class REFUTES \
  --required_class SUPPORTS --trials 250 --n_sent 2 \
  --outfile <gpt2_model_dir>/supports/attacks_out_refutes_supports_model_trials250_sent2 --verify \
  --verifier_chkpt <stance_checkpoint_dir>/model.best.pt
  ```
 - Convert *NEI* to *supports*: 
 ```
 python generate_gpt2_sentences.py \
  --dataset_loc <data_dir>/all_data/raw_data/all_dev.json \
  --model_loc <gpt2_model_dir>/supports/model.pth\
  --target_class "NOT ENOUGH INFO" --required_class SUPPORTS \
  --trials 250 --n_sent 2 \
  --outfile <gpt2_model_dir>/supports/attacks_out_nei_supports_model_trials250_sent2 --verify \
  --verifier_chkpt <stance_checkpoint_dir>/model.best.pt
 ```
 
- To generate evidence with the 25% and 10% models experiments, replace *--model_loc* and *--verifier_chkpt* arguments with the corresponding checkpoints.
```
python convert_to_kgat_generate_fakepage.py \
--infile_orig <data_dir>/all_data/preattack_retrieval_formatted/bert_eval2.json \
--infile_attack_pairs1 <gpt2_model_dir>/supports/attacks_out_refutes_supports_model_trials250_sent2 \
--infile_attack_pairs2 <gpt2_model_dir>/supports/attacks_out_nei_supports_model_trials250_sent2 \
--outfile <data_dir>/all_data/attacks_out_formatted/gpt2_generate/gpt2_generate_refutes_nei_supports_model_trials250_sent2_kgat.json
```

-  -  - 

 ### Convert to KGAT
