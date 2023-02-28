# Get KGAT predictions!

- Almost there! once you ran the attacks and the retrieval step, the rest is easy-peasy!
- Our verification's checkpoint can be found [here](https://oc.cs.uni-saarland.de/owncloud/index.php/s/erzCZjekbTwsXDz) (this is the defender's verification model). If you need to re-train the model, please check KGAT's [repository](https://github.com/thunlp/KernelGAT).

# Running KGAT without attacks
- Running on the results of the retrieval. 

```
python test.py --outdir ./output \
--test_path <data_dir>/preattack_retrieval_formatted/bert_eval2.json \
--bert_pretrain ../bert_base \
--checkpoint ../checkpoint/kgat/model.best.pt \
--name dev_eval2.json
```

# Running KGAT without attacks
- Similar, but replace the test_path with the re-retrieval results (using the defender's retrieval model) of the attacks. These files can be found under *postattack_retrieval*.

```
python test.py --outdir ./output \
--test_path <data_dir>/postattack_retrieval/imperceptible/imperceptible_delete_retrieved_processed.json \
--bert_pretrain ../bert_base \
--checkpoint ../checkpoint/kgat/model.best.pt \
--name dev_imperceptible_delete_retrieved_processed.json
```

