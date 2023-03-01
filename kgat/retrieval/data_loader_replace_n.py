import os
import torch
import numpy as np
import json
import re
from torch.autograd import Variable


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def tok2int_sent(sentence, tokenizer, max_seq_length):
    """Loads a data file into a list of `InputBatch`s."""
    sent_a, sent_b = sentence
    tokens_a = tokenizer.tokenize(sent_a)

    tokens_b = None
    if sent_b:
        tokens_b = tokenizer.tokenize(sent_b)
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

    tokens =  ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids = [0] * len(tokens)
    if tokens_b:
        tokens = tokens + tokens_b + ["[SEP]"]
        segment_ids += [1] * (len(tokens_b) + 1)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    padding = [0] * (max_seq_length - len(input_ids))

    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids





def tok2int_list(src_list, tokenizer, max_seq_length, max_seq_size=-1):
    inp_padding = list()
    msk_padding = list()
    seg_padding = list()
    for step, sent in enumerate(src_list):
        input_ids, input_mask, input_seg = tok2int_sent(sent, tokenizer, max_seq_length)
        inp_padding.append(input_ids)
        msk_padding.append(input_mask)
        seg_padding.append(input_seg)
    #if max_seq_size != -1:
    #    inp_padding = inp_padding[:max_seq_size]
    #    msk_padding = msk_padding[:max_seq_size]
    #    seg_padding = seg_padding[:max_seq_size]
    #    inp_padding += ([[0] * max_seq_length] * (max_seq_size - len(inp_padding)))
    #    msk_padding += ([[0] * max_seq_length] * (max_seq_size - len(msk_padding)))
    #    seg_padding += ([[0] * max_seq_length] * (max_seq_size - len(seg_padding)))
    return inp_padding, msk_padding, seg_padding


class DataLoaderTest(object):
    ''' For data iteration '''

    def __init__(self, data_path_all, data_path_preattack, data_path_attack, tokenizer, args, cuda=True, batch_size=64, n=2):
        self.cuda = cuda

        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_len = args.max_len
        self.evi_num = args.evi_num
        self.data_path_all = data_path_all
        self.data_path_preattack = data_path_preattack
        self.data_path_attack = data_path_attack
        self.n = n 
        self.preattack_sentences = self.read_file_groups_pre(data_path_preattack)
        self.postattack_sentences = self.read_file_groups_post(data_path_attack)
        inputs, ids, evi_list = self.read_file_pairs_WithReplace(data_path_all)
        self.inputs = inputs
        self.ids = ids
        self.evi_list = evi_list

        self.total_num = len(inputs)
        self.total_step = np.ceil(self.total_num * 1.0 / batch_size)
        self.step = 0

    def process_sent(self, sentence):
        sentence = re.sub(" \-LSB\-.*?\-RSB\-", "", sentence)
        sentence = re.sub("\-LRB\- \-RRB\- ", "", sentence)
        sentence = re.sub(" -LRB-", " ( ", sentence)
        sentence = re.sub("-RRB-", " )", sentence)
        sentence = re.sub("--", "-", sentence)
        sentence = re.sub("``", '"', sentence)
        sentence = re.sub("''", '"', sentence)
        return sentence

    def process_sent_extended(self,sentence):
        sentence = re.sub(" \-LSB\-.*?\-RSB\-", "", sentence)
        sentence = re.sub("\-LRB\- \-RRB\- ", "", sentence)
        sentence = re.sub(" -LRB-", " ( ", sentence)
        sentence = re.sub("-RRB-", " ) ", sentence)
        sentence = re.sub("RRB", " ) ", sentence)
        sentence = re.sub("LRB", " ( ", sentence)
        sentence = re.sub("--", " - ", sentence)
        sentence = re.sub("``", ' " ', sentence)
        sentence = re.sub("''", ' " ', sentence)
        sentence = re.sub("'", " ' ", sentence)
        sentence = sentence.replace(".", " .")
        sentence = sentence.replace(":", " : ")
        sentence = sentence.replace("?", " ? ")
        sentence = sentence.replace(")", " ) ")
        sentence = sentence.replace("(", " ( ")
        sentence = sentence.replace("!", "")
        sentence = sentence.replace("#", "")
        sentence = sentence.replace("$", "")
        sentence = sentence.replace("%", "")
        sentence = sentence.replace("&", "")
        sentence = sentence.replace("+", "")
        sentence = sentence.replace("-", "")
        sentence = sentence.replace("//", "")
        sentence = sentence.replace("/\/", "")
        sentence = sentence.replace("<", "")
        sentence = sentence.replace("=", "")
        sentence = sentence.replace(">", "")
        sentence = sentence.replace("@", "")
        sentence = sentence.replace("[", "")
        sentence = sentence.replace("]", "")
        sentence = sentence.replace("_", "")
        sentence = sentence.replace("@", "")
        sentence = sentence.replace(",", " , ")
        return sentence
        
    def process_wiki_title(self, title):
        title = re.sub("_", " ", title)
        title = re.sub(" -LRB-", " ( ", title)
        title = re.sub("-RRB-", " )", title)
        title = re.sub("-COLON-", ":", title)
        return title

    def read_file_pairs_WithReplace(self, data_path):
        inputs = list()
        ids = list()
        evi_list = list()
        print(len(self.postattack_sentences.keys()))
        with open(data_path) as fin:
            for step, line in enumerate(fin):
                instance = json.loads(line.strip())
                claim = instance['claim']
                id = int(instance['id'])
                if not id in self.postattack_sentences.keys(): continue 
                #if not id == 91198: continue # for debugging 
                preattack = self.preattack_sentences[id]
                postattack = self.postattack_sentences[id]
                for evidence in instance['evidence']:
                    ids.append(id)
                    evid_sent = self.process_sent(evidence[2])
                    attack_done = 0 
                    sent_count = 0 
                    found_count = -1                                        
                    for pre_sent in preattack:
                        if evid_sent == pre_sent: 
                            found_count = sent_count
                        sent_count = sent_count + 1
                    if found_count != -1: 
                        #print('****')
                        #print(postattack[found_count][2])  
                        #print(evid_sent)
                        #print('****')                        
                        evid_sent = postattack[found_count][2]
                        attack_done = postattack[found_count][3]
                    inputs.append([self.process_sent(claim), evid_sent])
                    evi_list.append([evidence[0],evidence[1],evid_sent,attack_done]) #attack_done is whether attack succeeded
        return inputs, ids, evi_list

    def read_file_groups_pre(self, data_path):
        evidence_per_id = dict()
        with open(data_path) as fin:
            for step, line in enumerate(fin):
                instance = json.loads(line.strip())
                claim = instance['claim']
                id = int(instance['id'])
                evidence_list = []
                for evidence in instance['evidence']:
                    evidence_list.append(self.process_sent(evidence[2]))
                evidence_per_id[id] = evidence_list
        return evidence_per_id

    def read_file_groups_post(self, data_path):
        evidence_per_id = dict()
        with open(data_path) as fin:
            for step, line in enumerate(fin):
                instance = json.loads(line.strip())
                claim = instance['claim']
                id = int(instance['id'])
                evidence_list = []
                evidence_list_pre = []
                #print('***')
                for count, evidence in enumerate(instance['evidence']):
                    if int(evidence[3]) == 1 or int(evidence[3]) == 2: 
                        evidence_list.append(evidence)
                        evidence_list_pre.append(self.preattack_sentences[id][count])
                evidence_list = evidence_list[0:self.n]
                evidence_list_pre = evidence_list_pre[0:self.n]
                evidence_per_id[id] = evidence_list
                self.preattack_sentences[id] = evidence_list_pre
                #print(evidence_list)
                #print(self.preattack_sentences[id])
                #if len(self.preattack_sentences[id]) != len(evidence_per_id[id]): print('error')
                #print('***')
        return evidence_per_id

    def shuffle(self):
        np.random.shuffle(self.examples)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self._n_batch

    def next(self):
        ''' Get the next batch '''
        if self.step < self.total_step:
            inputs = self.inputs[self.step * self.batch_size : (self.step+1)*self.batch_size]
            ids = self.ids[self.step * self.batch_size: (self.step + 1) * self.batch_size]
            evi_list = self.evi_list[self.step * self.batch_size: (self.step + 1) * self.batch_size]
            inp, msk, seg = tok2int_list(inputs, self.tokenizer, self.max_len, -1)
            inp_tensor_input = Variable(
                torch.LongTensor(inp))
            msk_tensor_input = Variable(
                torch.LongTensor(msk))
            seg_tensor_input = Variable(
                torch.LongTensor(seg))
            if self.cuda:
                inp_tensor_input = inp_tensor_input.cuda()
                msk_tensor_input = msk_tensor_input.cuda()
                seg_tensor_input = seg_tensor_input.cuda()
            self.step += 1
            return inp_tensor_input, msk_tensor_input, seg_tensor_input, ids, evi_list
        else:
            self.step = 0
            raise StopIteration()