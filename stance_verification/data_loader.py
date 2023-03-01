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
    return inp_padding, msk_padding, seg_padding


class DataLoader(object):
    ''' For data iteration '''

    def __init__(self, data_path, tokenizer, args, labels_dict, test=False, cuda=True, batch_size=64):
        self.cuda = cuda

        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.labels_dict = labels_dict
        self.max_len = args.max_len
        self.evi_num = args.evi_num
        self.data_path = data_path
        self.test = test
        examples = self.read_file(data_path)
        self.examples = examples
        self.total_num = len(examples)
        self.total_step = self.total_num / batch_size
        if not test: self.shuffle()
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

    def process_wiki_title(self, title):
        title = re.sub("_", " ", title)
        title = re.sub(" -LRB-", " ( ", title)
        title = re.sub("-RRB-", " )", title)
        title = re.sub("-COLON-", ":", title)
        return title

    def convert_labels_to_int(self,labels_batch):
        labels_int_batch = list()
        for label in labels_batch:
            labels_int_batch.append(self.labels_dict[label])
        return labels_int_batch 
        

    def read_file(self, data_path):
        examples = list()
        with open(data_path) as fin:
            for step, line in enumerate(fin):
                sublines = line.strip().split("\t")
                claim = sublines[0]
                title = sublines[1]
                evidence = sublines[2]
                label = sublines[3]
                examples.append([self.process_sent(claim), self.process_sent(evidence),label])
        return examples


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
            examples = self.examples[self.step * self.batch_size : (self.step+1)*self.batch_size]
            pairs = list()
            labels = list()
            
            for example in examples:
                pairs.append([example[0],example[1]])
                labels.append(example[2])
            inp, msk, seg = tok2int_list(pairs, self.tokenizer, self.max_len)
            labels_int = self.convert_labels_to_int(labels) 

            inp_tensor = Variable(
                torch.LongTensor(inp))
            msk_tensor = Variable(
                torch.LongTensor(msk))
            seg_tensor = Variable(
                torch.LongTensor(seg))
                
            labels_int_tensor = Variable(
                torch.LongTensor(labels_int))

            if self.cuda:
                inp_tensor = inp_tensor.cuda()
                msk_tensor = msk_tensor.cuda()
                seg_tensor = seg_tensor.cuda()
                labels_int_tensor = labels_int_tensor.cuda()
            self.step += 1
            return inp_tensor, msk_tensor, seg_tensor, labels_int_tensor
        else:
            self.step = 0
            if not self.test: self.shuffle()
            raise StopIteration()
