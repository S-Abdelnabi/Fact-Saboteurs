# Trained retrieval models
- We provide our trained checkpoints: [Link](https://oc.cs.uni-saarland.de/owncloud/index.php/s/cZ5Jb5kCRkcmRnm).
- If you need to re-train the models, please refer to the KGAT's [repository](https://github.com/thunlp/KernelGAT). Training data (in the same format as KGAT) is available in our shared [Google Drive directory](https://drive.google.com/drive/folders/1xbSzefjPm4Ii5WQSKX2C5wT5MydBkqcT?usp=sharing) (*retrieval_training* under *raw_data*)

# Running the retrieval model (without attacks)

- This step retrieves evidence given claims (baseline - no attacks). It yields the [files](https://drive.google.com/drive/folders/1xbSzefjPm4Ii5WQSKX2C5wT5MydBkqcT?usp=sharing) in *preattack_retrieval_formatted* under *raw_data*. 
- The output is used for predictions later (baseline), or to run the attacks on the retrieved results.

```
python test.py --outdir ./output/ \
--test_path ../data/all_dev.json \
--bert_pretrain ../bert_base \
--checkpoint ../checkpoint/retrieval_model/model.best.pt \
--name dev.json

python process_data.py --retrieval_file ./output/dev.json --gold_file ../data/golden_dev.json --output ../data/bert_eval2.json --test
```
- Replace with your required check point. 


