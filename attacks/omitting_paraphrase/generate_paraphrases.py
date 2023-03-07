import argparse
import torch
from functools import partial
#import pandas as pd
from datareader import EvidenceFeverDataset_ForAttack
import random
from itertools import zip_longest
from tqdm import tqdm
import numpy as np
import json 
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

def get_paraphrases(input_text,num_return_sequences,max_length=60,num_beams=20):
  batch = tokenizer.prepare_seq2seq_batch([input_text],truncation=True,padding='longest',max_length=max_length, return_tensors="pt").to(torch_device)
  translated = model.generate(**batch,max_length=max_length,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)
  tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
  return tgt_text
  
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_loc", help="Root directory of the dataset", required=True, type=str)
    parser.add_argument("--n_gpu", help="The number of GPUs to use", type=int, default=1)
    parser.add_argument("--n_evi", help="number of evidence sentences to replace", type=int, default=2)
    parser.add_argument("--paraphrases", help="how many paraphrasing sentences to generate", type=int, default=20)
    parser.add_argument("--beams", help="num of beams in beam search", type=int, default=30)
    parser.add_argument("--max_length", help="num of beams in beam search", type=int, default=60)
    parser.add_argument("--seed", type=int, help="Random seed", default=1000)
    parser.add_argument('--outfile', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='tuner007/pegasus_paraphrase')
    parser.add_argument("--start", type=int, default=0 )
    parser.add_argument("--end", type=int, default=-1 )
    parser.add_argument("--target_class", nargs="+", help="The labels of the claims to generate evidence for", required=True)

    args = parser.parse_args()
    
    model_name = args.model_name #'tuner007/pegasus_paraphrase' #'sshleifer/distill-pegasus-xsum-16-4' #
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

    print(args.target_class)

    # Set all the seeds
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    #Load the data
    valdset = EvidenceFeverDataset_ForAttack(args.target_class, args.dataset_loc, tokenizer,args.n_evi)
    valdset._dataset = valdset._dataset[args.start:]
    if args.end != -1: valdset._dataset = valdset._dataset[:args.end]


    for instance in tqdm(valdset._dataset, desc="Generation"):
        evidence_paraphrased = [] 
        for evi in instance['evidence']:
            #print(evi[2])
            paraphrases = get_paraphrases(evi[2], args.paraphrases,max_length=args.max_length,num_beams=args.beams) 
            evidence_paraphrased.append(paraphrases)
        new_instance = {'id':instance['id'],'claim':instance['claim'],'evidence_paraphrased':evidence_paraphrased, 'orig_evidence':instance['evidence'],'label':instance['label']}
        with open(args.outfile, "a") as out:
            out.write(json.dumps(new_instance) + "\n")  