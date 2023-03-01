## BERT stance verification 
- Some of the attacks require verification models. 
- We train a BERT model on pairs of *<claim, evidence>*. The labels are *SUP*, *REF*, and *NEI*. This is different from KGAT verification that is trained on graphs of evidence. 
- The checkpoints for the verification models are [available](https://oc.cs.uni-saarland.de/owncloud/index.php/s/WKqPaHijfBPxoW5), containing models trained with the whole dataset or subsets from it.
- The training data for the verification models can be found [here](https://drive.google.com/drive/folders/1xbSzefjPm4Ii5WQSKX2C5wT5MydBkqcT?usp=sharing), under *raw_data/fc_training* 
