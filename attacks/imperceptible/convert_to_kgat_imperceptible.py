import os
import json
import argparse
import numpy as np
import pickle
 
def read_file_orig_examples(data_path):
    orig_examples = {}
    with open(data_path) as fin:
        for step, line in enumerate(fin):
            instance = json.loads(line.strip())
            id_ = int(instance["id"])
            label = instance['label']
            orig_examples[id_] = label             
    return orig_examples

def get_file_ids_titles(data_path,keep_counter):
    ids_titles = []
    with open(data_path) as fin:
        for step, line in enumerate(fin):
            sublines = line.strip().split("\t")
            if keep_counter:
                ids_titles.append([sublines[5],sublines[1],sublines[6]])       
            else:
                ids_titles.append([sublines[5],sublines[1]])           
    return ids_titles



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile_attack_pairs')
    parser.add_argument('--infile_preattack_pairs')
    parser.add_argument('--keep_counter',action='store_true')
    parser.add_argument('--infile_orig')
    parser.add_argument('--outfile')
    args = parser.parse_args()
    orig_examples_labels = read_file_orig_examples(args.infile_orig)
    example_ids = get_file_ids_titles(args.infile_preattack_pairs,args.keep_counter)
    print(len(example_ids))
    examples = {}
    evid_counter = 0 
    examples_count = 0 
    with open(args.infile_attack_pairs, 'rb') as f:
        perturbs_test = pickle.load(f)
    for key_ in perturbs_test.keys():
        budgets = list(perturbs_test[key_].keys())
        for i in range(0,len(perturbs_test[key_][budgets[0]])): #all exampless
            example_info = example_ids[i]
            id_ = example_info[0]
            title_ = example_info[1]
            id_ = int(id_)
            #print(id_)
            instance_orig = perturbs_test[key_][budgets[0]][i]
            claim = instance_orig['claim']
            if not id_ in examples and id_ != ' ':  
                if not args.keep_counter: evid_counter = 0 
                examples[id_] = {}
                examples[id_]['id'] = id_
                examples[id_]['claim'] = claim
                examples[id_]['evidence'] = []
                examples[id_]['label'] = orig_examples_labels[id_]
            if i in perturbs_test[key_][budgets[-1]]:
                one_evidence_text = perturbs_test[key_][budgets[-1]][i]['adv_example']
                attack_done = 1
            else:
                one_evidence_text = perturbs_test[key_][budgets[0]][i]['input']
                attack_done = 0
            if args.keep_counter: evid_counter = example_info[2]
            one_evidence_list =  [title_,evid_counter,one_evidence_text,attack_done]               
            if not args.keep_counter: evid_counter = evid_counter + 1 
            if id_ != ' ': examples[id_]['evidence'].append(one_evidence_list) 

    with open(args.outfile, "w") as out:
        for data in examples.values():
            out.write(json.dumps(data) + "\n")  