from typing import AnyStr
from typing import List
from typing import Tuple
from typing import Set
import unicodedata
import json
import random
import string

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

target_labels_dict = {"SUPPORTS":"REFUTES", "REFUTES":"SUPPORTS", "NOT ENOUGH INFO":"SUPPORTS"}

def text_to_batch_transformer(claims, tokenizer, evidence, labels,include_label) -> Tuple[List, List]:
    """Turn a piece of text into a batch for transformer model
    :return: A list of IDs and a mask
    """
    if include_label:
        texts = [f"{claim}||{label.lower()}||{evid}" for claim,label,evid in zip(claims,labels,evidence)]
        #print(texts)
        input_ids = [tokenizer.encode(t, max_length=tokenizer.model_max_length-1,truncation=True) + [tokenizer.eos_token_id] for t in texts]
        masks = [[1] * len(i) for i in input_ids]
        return input_ids, masks
    
    texts = [f"{claim}||{evid}" for claim,evid in zip(claims,evidence)]
    #print(texts)
    input_ids = [tokenizer.encode(t, max_length=tokenizer.model_max_length-1,truncation=True) + [tokenizer.eos_token_id] for t in texts]

    masks = [[1] * len(i) for i in input_ids]

    return input_ids, masks

def text_to_batch_transformer_ForAttack(claims: List, tokenizer: PreTrainedTokenizer,labels,include_label) -> Tuple[List, List]:
    """Turn a piece of text into a batch for transformer model
    :return: A list of IDs and a mask
    """
    if include_label:
        #print(target_labels_dict[labels[0]])
        texts = [f"{claim}||{target_labels_dict[label].lower()}||" for claim,label in zip(claims,labels)]
        #print(texts)
        input_ids = [tokenizer.encode(t, max_length=tokenizer.model_max_length,truncation=True) for t in texts]
        masks = [[1] * len(i) for i in input_ids]
        return input_ids, masks

        
    texts = [f"{claim}||" for claim in claims]
    input_ids = [tokenizer.encode(t, max_length=tokenizer.model_max_length,truncation=True) for t in texts]

    masks = [[1] * len(i) for i in input_ids]

    return input_ids, masks


def collate_batch_transformer(input_data: Tuple) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = [i[0][0] for i in input_data]
    #print(len(input_ids))
    masks = [i[1][0] for i in input_data]
    #print(len(masks))
    ids_list = [i[2] for i in input_data] 
    #print(ids_list)
    max_length = max([len(i) for i in input_ids])

    input_ids = [(i + [0] * (max_length - len(i))) for i in input_ids]
    masks = [(m + [0] * (max_length - len(m))) for m in masks]

    assert (all(len(i) == max_length for i in input_ids))
    assert (all(len(m) == max_length for m in masks))
    return torch.tensor(input_ids), torch.tensor(masks),ids_list 


def collate_batch_transformer_with_index(input_data: Tuple) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List]:
    return collate_batch_transformer(input_data) + ([i[-1] for i in input_data],)


class GPT2FeverDataset(Dataset):
    def __init__(self, labels, data_dir: str, tokenizer: PreTrainedTokenizer, include_label = False):
        super().__init__()
        self._dataset = []
        self.tokenizer = tokenizer
        self.labels = labels
        self.reader(data_dir)
        self.include_label = include_label
    def reader(self, path):
        with open(path) as f:
            for idx, line in enumerate(f):
                instance = json.loads(line)
                if instance['label'] not in self.labels: continue
                claim = instance['claim']
                id_ = instance['id']
                for evi_ in instance['evidence']:
                    if evi_[3] != 1: continue ##not gold evidence
                    self._dataset.append({"id":id_, "claim": claim, "evidence": evi_[2], "label":instance['label']})
        for i in range(0,5):
            if i < len(self._dataset):
                print('****')        
                print(self._dataset[i])    
                print('****')                     
    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, item):
        """
        :return:
        {'id': 163803,
        'label': 'SUPPORTS',
        'claim': 'Ukrainian Soviet Socialist Republic was a founding participant of the UN.',
        'evidence': [['Ukrainian_Soviet_Socialist_Republic', 7,
            'The Ukrainian SSR was a founding member of the United Nations , although it was legally represented by the
            All-Union state in its affairs with countries outside of the Soviet Union .'
        ]]}
        """
        row = self._dataset[item]
        claim = row['claim']
        evidence = row['evidence']
        id_ = row['id']
        label = row['label']
        input_ids, masks = text_to_batch_transformer([claim], self.tokenizer, [evidence],[label],self.include_label)
        return input_ids, masks, id_, item
        

class GPT2FeverDataset_ForAttack(Dataset):
    def __init__(self, labels, data_dir: str, tokenizer: PreTrainedTokenizer,include_label=False):
        super().__init__()
        self._dataset = []
        self.tokenizer = tokenizer
        self.labels = labels
        self.reader(data_dir)
        self.include_label = include_label
    def reader(self, path):
        with open(path) as f:
            for idx, line in enumerate(f):
                instance = json.loads(line)
                if instance['label'] not in self.labels: continue
                claim = instance['claim']
                id_ = instance['id']
                self._dataset.append({"id":id_, "claim": claim, "label":instance['label']})
        for i in range(0,5):
            if i < len(self._dataset):
                print('****')        
                print(self._dataset[i])    
                print('****')                   
    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, item):
        """
        :return:
        {'id': 163803,
        'label': 'SUPPORTS',
        'claim': 'Ukrainian Soviet Socialist Republic was a founding participant of the UN.',
        'evidence': [['Ukrainian_Soviet_Socialist_Republic', 7,
            'The Ukrainian SSR was a founding member of the United Nations , although it was legally represented by the
            All-Union state in its affairs with countries outside of the Soviet Union .'
        ]]}
        """
        row = self._dataset[item]
        claim = row['claim']
        id_ = row['id']
        label = row['label']
        input_ids, masks = text_to_batch_transformer_ForAttack([claim], self.tokenizer,[label],self.include_label)
        return input_ids, masks, id_, item