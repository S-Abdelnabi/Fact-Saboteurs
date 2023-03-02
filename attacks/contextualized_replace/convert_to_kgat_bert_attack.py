import os
import json
import argparse
import numpy as np


def read_file_orig_examples(data_path):
    orig_examples = {}
    with open(data_path) as fin:
        for step, line in enumerate(fin):
            instance = json.loads(line.strip())
            id_ = int(instance["id"])
            label = instance['label']
            orig_examples[id_] = label             
    return orig_examples

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile_attack_pairs')
    parser.add_argument('--infile_orig')
    parser.add_argument('--outfile')
    args = parser.parse_args()
    orig_examples_labels = read_file_orig_examples(args.infile_orig)
    examples = {}
    evid_counter = 0 
    with open(args.infile_attack_pairs, 'r') as f:
        for line in f:
            sublines = json.loads(line)
            claim = sublines['claim']
            id_ = int(sublines['id_in_fever'])
            if not id_ in examples:  
                evid_counter = 0 
                examples[id_] = {}
                examples[id_]['id'] = id_
                examples[id_]['claim'] = claim
                examples[id_]['evidence'] = []
                examples[id_]['label'] = orig_examples_labels[id_]
            one_evidence_title = sublines['title'] 
            one_evidence_text = sublines['adv']                 
            if sublines['success']==4:
                attack_done = 1
            elif sublines['success']==2:
                attack_done = 2
            else: 
                attack_done = 0
            one_evidence_list =  [one_evidence_title,evid_counter,one_evidence_text,attack_done]               
            evid_counter = evid_counter + 1 
            examples[id_]['evidence'].append(one_evidence_list) 

    with open(args.outfile, "w") as out:
        for data in examples.values():
            out.write(json.dumps(data) + "\n")  