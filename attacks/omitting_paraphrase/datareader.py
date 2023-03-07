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
        

class EvidenceFeverDataset_ForAttack(Dataset):
    def __init__(self, labels, data_dir: str, tokenizer: PreTrainedTokenizer,num_evidence):
        super().__init__()
        self._dataset = []
        self.tokenizer = tokenizer
        self.labels = labels
        self.num_evidence = num_evidence
        self.reader(data_dir)
    def reader(self, path):
        with open(path) as f:
            for idx, line in enumerate(f):
                instance = json.loads(line)
                if instance['label'] not in self.labels: continue
                claim = instance['claim']
                id_ = instance['id']
                evidence = instance['evidence'][0:self.num_evidence]         
                self._dataset.append({"id":id_, "claim": claim, "label":instance['label'],'evidence':evidence})
        for i in range(0,5):
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
        return row