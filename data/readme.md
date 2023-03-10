### Overview 

- We share our data of already computed attacks: [Google Drive Link](https://drive.google.com/drive/folders/1xbSzefjPm4Ii5WQSKX2C5wT5MydBkqcT?usp=sharing)
- Contains:
	- **raw_data** 
	    - FEVER dataset (and different versions of it) used to train the different models (both fact-verification and attack models)
	- **preattack_retrieval_formatted**
	    - claims with retrieval output (golden evidence is not added explicitly) 
	    - no attacks are included 
	    - in normal KGAT flow, this is the input to prediction (no attack baseline)
	    - for attacks that first use the retrieval output, this is the input to the attack    
	- **preattack_retrieval_pairs** (intermediate step) 
	    - files in **preattack_retrieval_formatted** are organized as pairs of *<claim,evidence>*
	    - sometimes used as input to run the attacks that edit the sentence. All evidence sentences are labelled with the original label of the example.
	- **attacks_out_pairs** (intermediate step) 
	    - output of attacks as pairs 
	- **attacks_out_formatted** (intermediate step) 
	    - output of attacks formatted in the same style of KGAT, i.e., claims with groups of evidence (attack result). 
	- **postattack_retrieval**
	    - this is the output of retrieval after the attack sentences are added/used to replace the original evidence. 
	    - directory may include different 
	    - this the is ***final*** output of the attack. Used to test the verification model. 
	- **paraphrased_claims**
	    - the final retrieval files for the claim paraphrases experiments (this is then the input to the verification model).
	    
- Directories may include additional readme files to give more information about individual files.
