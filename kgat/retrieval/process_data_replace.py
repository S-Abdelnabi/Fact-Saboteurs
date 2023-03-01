import json
import os
import argparse
import re 

def process_sent(sentence):
    sentence = re.sub(" \-LSB\-.*?\-RSB\-", "", sentence)
    sentence = re.sub("\-LRB\- \-RRB\- ", "", sentence)
    sentence = re.sub(" -LRB-", " ( ", sentence)
    sentence = re.sub("-RRB-", " )", sentence)
    sentence = re.sub("--", "-", sentence)
    sentence = re.sub("``", '"', sentence)
    sentence = re.sub("''", '"', sentence)
    return sentence

def read_file_groups_pre(data_path):
    evidence_per_id = dict()
    with open(data_path) as fin:
        for step, line in enumerate(fin):
            instance = json.loads(line.strip())
            claim = instance['claim']
            id = int(instance['id'])
            evidence_list = []
            for evidence in instance['evidence']:
                evidence_list.append(process_sent(evidence[2]))
            evidence_per_id[id] = evidence_list
    return evidence_per_id

def read_file_groups_post(data_path):
    evidence_per_id = dict()
    with open(data_path) as fin:
        for step, line in enumerate(fin):
            instance = json.loads(line.strip())
            claim = instance['claim']
            id = int(instance['id'])
            evidence_list = []
            for evidence in instance['evidence']:
                evidence_list.append(evidence[2])
            evidence_per_id[id] = evidence_list
    return evidence_per_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold_file')
    parser.add_argument('--retrieval_file_attack')
    parser.add_argument('--retrieval_file_orig')
    parser.add_argument('--retrieval_file_orig_attacked')
    parser.add_argument('--output')
    parser.add_argument('--test', action='store_true', default=False)
    args = parser.parse_args()
    
    preattack_sentences = read_file_groups_pre(args.retrieval_file_orig)
    postattack_sentences = read_file_groups_post(args.retrieval_file_orig_attacked)


    filter_dict = dict()
    data_dict = dict()
    golden_dict = dict()
    with open(args.gold_file) as f:
        for line in f:
            data = json.loads(line.strip())
            id = data["id"] 
            if id in postattack_sentences.keys(): 
                preattack = preattack_sentences[id]
                postattack = postattack_sentences[id]
                #print('***')
                #print(len(preattack))
                #print(len(postattack))
                #print('***')
            else: continue 
            data_dict[data["id"]] = {"id": data["id"], "evidence":[], "claim": data["claim"]}
            if "label" in data:
                data_dict[data["id"]]["label"] = data["label"]
            if not args.test:                
                for evidence in data["evidence"]:
                    evid_sent = process_sent(evidence[2])
                    sent_count = 0 
                    found_count = -1                                        
                    for pre_sent in preattack:
                        if evid_sent == pre_sent: 
                            #print('****')
                            #print(id)
                            #print(evid_sent)
                            #print(pre_sent)
                            #print(postattack[sent_count])
                            #print('****')
                            found_count = sent_count
                        sent_count = sent_count + 1
                    if found_count != -1:  #some items have more than 5 in golden evidence file
                        #print('***')
                        #print(evid_sent)
                        evid_sent = postattack[found_count]  
                        #print(evid_sent)
                        #print('***')
                    #else: print(id)                    
                    data_dict[data["id"]]["evidence"].append([evidence[0], evidence[1], evid_sent, 1.0])
                    string = str(data["id"]) + "_" + evidence[0] + "_" + str(evidence[1])
                    golden_dict[string] = 1
    with open(args.retrieval_file_attack) as f:
        for line in f:
            data = json.loads(line.strip())
            for step, evidence in enumerate(data["evidence"]):
                string = str(data["id"]) + "_" + str(evidence[0]) + "_" + str(evidence[1])
                if string not in golden_dict and string not in filter_dict:
                    data_dict[data["id"]]["evidence"].append([evidence[0], evidence[1], evidence[2], evidence[4]])
                    filter_dict[string] = 1
    with open(args.output, "w") as out:
        for data in data_dict.values():
            evidence_tmp = data["evidence"]
            evidence_tmp = sorted(evidence_tmp, key=lambda x:x[3], reverse=True)
            data["evidence"] = evidence_tmp[:5]
            out.write(json.dumps(data) + "\n")