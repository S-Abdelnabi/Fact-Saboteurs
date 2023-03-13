import argparse
import torch
from functools import partial
#import pandas as pd
from transformers import GPT2Tokenizer
from transformers import GPT2LMHeadModel
from datareader import GPT2FeverDataset_ForAttack
from datareader import collate_batch_transformer

import random
from itertools import zip_longest
from tqdm import tqdm
from transformers import BertConfig
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
from transformers import BertModel
from transformers import RobertaTokenizer
import numpy as np
from transformers import AutoTokenizer
from transformers import RobertaForSequenceClassification
from transformers import RobertaConfig
from torch.utils.data import DataLoader
from model import inference_model
import json 




LABELS = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2}


def bert_tokenization(claim, evidence, max_length, tokenizer):
    encoded = tokenizer.encode_plus(text=claim, text_pair=evidence, add_special_tokens=True,padding='max_length',truncation =True,max_length=max_length,return_tensors ="pt",return_token_type_ids=True,return_attention_mask=True)
    return encoded

def process_bert_item(claim,evidence,tokenizer,max_length,squeeze=True):
    source_inputs = bert_tokenization(claim, evidence, max_length, tokenizer)
    if squeeze:
        source_ids = source_inputs["input_ids"].squeeze().cuda()
        src_mask = source_inputs["attention_mask"].squeeze().cuda()
        type_ids = source_inputs["token_type_ids"].squeeze().cuda()
    else:
        source_ids = source_inputs["input_ids"].cuda()
        src_mask = source_inputs["attention_mask"].cuda()
        type_ids = source_inputs["token_type_ids"].cuda()
    #print(source_ids.size())
    return {
    "input_ids": source_ids,
    "attention_mask": src_mask,
    "type_ids": type_ids}

def bert_predict(model, batch):
    input_ids, attn_mask, token_type_ids = (
        batch["input_ids"],
        batch["attention_mask"],
        batch["type_ids"]
    )  
    outputs = model(input_ids, attn_mask,token_type_ids).squeeze()
    return outputs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_loc", help="Root directory of the dataset", required=True, type=str)
    parser.add_argument("--n_gpu", help="The number of GPUs to use", type=int, default=1)
    parser.add_argument("--n_sent", help="number of n_sentences to generate", type=int, default=2)
    parser.add_argument("--trials", help="how many times to try to get correct stance sentence", type=int, default=30)
    parser.add_argument("--model_loc", help="Location of the model to use", default="model.pth", type=str)
    parser.add_argument("--target_class", nargs="+", help="The labels of the claims to generate evidence for", required=True)
    parser.add_argument("--required_class", type=str, help="The required stance, SUPPORTS or REFUTES")
    parser.add_argument("--include_label", action="store_true", help="whether to concateate labels in training, should be used when ")
    parser.add_argument("--verify", action='store_true', help="Run against a model trained on Fever")
    parser.add_argument("--verifier_model", help="bert or roberta", default='bert',type=str)
    parser.add_argument("--verifier_chkpt", help="path to the checkpoint", default='bert',type=str)
    parser.add_argument("--seed", type=int, help="Random seed", default=1000)
    parser.add_argument('--outfile', type=str, required=True)
    parser.add_argument("--bert_hidden_dim", default=768, type=int, help="Total batch size for training.")
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout.')
    parser.add_argument("--num_labels", type=int, default=3 )
    parser.add_argument("--start", type=int, default=0 )
    parser.add_argument("--end", type=int, default=-1 )

    args = parser.parse_args()
    print(args.target_class)

    # Set all the seeds
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Create the model
    device = torch.device("cuda:0")
    model = torch.nn.DataParallel(GPT2LMHeadModel.from_pretrained('gpt2')).to(device)
    model.load_state_dict(torch.load(args.model_loc))
    # Load verifier_model 
    if args.verify: 
        if args.verifier_model == 'bert':
            bert_model = BertModel.from_pretrained('bert-base-uncased')
            bert_model = bert_model.cuda()
            bert_fever_model = inference_model(bert_model, args)
            bert_fever_model.load_state_dict(torch.load(args.verifier_chkpt)['model'],strict=False)
            bert_fever_model = bert_fever_model.cuda()
            bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            print('Loaded verifier model')

    #Load the data
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    valdset = GPT2FeverDataset_ForAttack(args.target_class, args.dataset_loc, tokenizer,args.include_label)
    valdset._dataset = valdset._dataset[args.start:]
    if args.end != -1: valdset._dataset = valdset._dataset[:args.end]

    val_dl = DataLoader(
        valdset,
        batch_size=1,
        collate_fn=collate_batch_transformer
    )

    for batch in tqdm(val_dl, desc="Generation"):
        ids = batch[2]
        new_group = {'id':ids[0], 'evidence':[]} 

        batch = batch[0:2]
        batch = tuple(t.to(device) for t in batch)
        input_ids = batch[0]
        masks = batch[1]
        trial_counter = 0 
        generated_sentences = []
        all_probs = []
        while len(new_group['evidence']) < args.n_sent:
            trial_counter += 1
            if trial_counter == args.trials: 
                #print(all_probs)
                hightest_idxs = np.asarray(all_probs).argsort()[::-1][:args.n_sent]
                #print(hightest_idxs)
                #random_index = np.random.randint(low = 0, high = len(generated_sentences), size = args.n_sent-len(new_group['evidence']))
                new_group['evidence'] += [[generated_sentences[i],0] for i in hightest_idxs]
                break 
            output = model.module.generate(input_ids, do_sample=True, top_k=10, temperature=0.7, max_length=1000, use_cache=True)
            output_text = tokenizer.decode(output[0], skip_special_tokens=True)
            #print(output_text)
            parts = output_text.split('||')
            claim = parts[0]
            if args.include_label:
                required_class = parts[1].upper()
                generated_evi_ = parts[2]
            else:
                required_class = args.required_class
                generated_evi_ = parts[1]
            new_group['required_class'] = required_class
            if claim.replace('.','') in generated_evi_: continue #prevent exact matches
            
            #print(output_text)
            new_group['claim'] = claim 
            if args.verify:
                source_inputs = process_bert_item(claim,generated_evi_,bert_tokenizer,500,squeeze=False)
                probs = bert_predict(bert_fever_model, source_inputs)
                probs = torch.softmax(probs, -1)
                #all_probs.append(probs[LABELS[required_class]].item())
                predicted_label = torch.argmax(probs)    
                #print(LABELS[required_class])
                #print(predicted_label.item())
                if predicted_label != LABELS[required_class]: 
                    all_probs.append(probs[LABELS[required_class]].item())
                    generated_sentences.append(generated_evi_)
                    continue                
            new_group['evidence'].append([generated_evi_,1])    
        
        with open(args.outfile, "a") as out:
            out.write(json.dumps(new_group) + "\n")  