import os
import json
import argparse
import numpy as np


def read_examples_orig(data_path):
    examples = {}
    with open(data_path) as fin:
        for step, line in enumerate(fin):
            instance = json.loads(line.strip())
            id_ = int(instance["id"])
            examples[id_] = instance             
    return examples

def read_examples_attack(data_path):
    examples = {}
    with open(data_path) as fin:
        for step, line in enumerate(fin):
            instance = json.loads(line.strip())
            id_ = int(instance["id"])
            examples[id_] = instance             
    return examples
def merge_dicts(*dict_args):
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile_attack_pairs1')
    parser.add_argument('--infile_attack_pairs2',default='')
    parser.add_argument('--infile_orig')
    parser.add_argument('--outfile')
    args = parser.parse_args()
    orig_examples = read_examples_orig(args.infile_orig)
    attack_examples1 = read_examples_attack(args.infile_attack_pairs1)
    if args.infile_attack_pairs2 != '':
        attack_examples2 = read_examples_attack(args.infile_attack_pairs2)
        attack_examples = merge_dicts(attack_examples1,attack_examples2)
    else:
        attack_examples = attack_examples1
    examples = {}
    evid_counter = 0 
    for orig_instance_ids in orig_examples:
        orig_instance = orig_examples[orig_instance_ids]
        claim = orig_instance['claim']
        id_ = int(orig_instance['id'])
        label = orig_instance['label']
        if len(orig_instance["evidence"]) == 0: continue
        if not id_ in attack_examples: 
            continue
        else: 
            old_evidence = orig_instance["evidence"]
            new_evidence = attack_examples[id_]["evidence"]
            top_page = old_evidence[0][0]
            examples[id_] = {}
            examples[id_]['id'] = id_
            examples[id_]['claim'] = claim
            examples[id_]['evidence'] = []
            examples[id_]['label'] = orig_instance["label"]
            for evi_idx_,evi_ in enumerate(new_evidence):
                one_evidence_list = [top_page,0,evi_[0],1]
                examples[id_]['evidence'].append(one_evidence_list)               
    with open(args.outfile, "w") as out:
        for data in examples.values():
            out.write(json.dumps(data) + "\n")  