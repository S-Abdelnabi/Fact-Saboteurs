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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile_attack_pairs')
    parser.add_argument('--infile_orig')
    parser.add_argument('--outfile')
    args = parser.parse_args()
    orig_examples = read_examples_orig(args.infile_orig)
    attack_examples = read_examples_attack(args.infile_attack_pairs)

    examples = {}
    evid_counter = 0 
    for orig_instance_ids in orig_examples:
        orig_instance = orig_examples[orig_instance_ids]
        claim = orig_instance['claim']
        id_ = int(orig_instance['id'])
        if not id_ in attack_examples: 
            continue 
        else: 
            old_evidence = orig_instance["evidence"]
            evidence_edited = attack_examples[id_]["paraphrased_evidence"]
            examples[id_] = {}
            examples[id_]['id'] = id_
            examples[id_]['claim'] = claim
            examples[id_]['evidence'] = []
            examples[id_]['label'] = orig_instance["label"]
            found_evidence = []
            #print(id_)
            for evi_idx_,evi_ in enumerate(evidence_edited):
                if evi_idx_ < len(old_evidence):
                    found_evidence.append(str(old_evidence[evi_idx_][0]+str(old_evidence[evi_idx_][1])))
                    one_evidence_list = [old_evidence[evi_idx_][0],old_evidence[evi_idx_][1],evi_[0][2],1]
                    examples[id_]['evidence'].append(one_evidence_list) 
            ##fill in the rest    
            for evi_idx_,evi_ in enumerate(old_evidence):
                if str(evi_[0]+str(evi_[1])) in found_evidence: continue 
                one_evidence_list = [evi_[0],evi_[1],evi_[2],0]
                examples[id_]['evidence'].append(one_evidence_list)                
    with open(args.outfile, "w") as out:
        for data in examples.values():
            out.write(json.dumps(data) + "\n")  