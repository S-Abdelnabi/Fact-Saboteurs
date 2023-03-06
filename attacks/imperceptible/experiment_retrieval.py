#!/usr/bin/env python3

import sys
import re 
import torch
import pickle
import json
import numpy as np
import pandas as pd
from timeit import timeit
from abc import ABC
from typing import List, Tuple, Callable, Dict
from scipy.optimize import differential_evolution
from tqdm.auto import tqdm
from argparse import ArgumentParser
from time import process_time, sleep, time
from torch.nn.functional import softmax
from os.path import exists
from logging import getLogger, WARNING
from getpass import getpass
from string import punctuation
from models import inference_model
from textdistance import levenshtein
from transformers import BertConfig, BertTokenizer
from bert_model import BertForSequenceEncoder

from transformers import BertForSequenceClassification, BertForMaskedLM
from transformers import BertModel
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset, Dataset

class CustomDataset(Dataset):
    def __init__(self, input_ids, masks, segments):
        self.input_ids = input_ids
        self.masks = masks
        self.segments = segments
    def __len__(self):
        return self.input_ids.size(0) 
		
    def __getitem__(self, idx):
        input_ids = self.input_ids[idx,:]
        mask = self.masks[idx,:]
        segments = self.segments[idx,:] 
        return input_ids,mask,segments
        
def _get_masked(words):
    len_text = len(words)
    masked_words = []
    if len_text == 1: ##special case
        return ['[UNK]']
    for i in range(len_text - 1):
        masked_words.append(words[0:i] + ['[UNK]'] + words[i + 1:])
    # list of words
    return masked_words

def get_important_scores(words_claim, words_evidence, tgt_model, orig_prob, orig_label, orig_probs, tokenizer, batch_size):
    masked_words = _get_masked(words_evidence) 
    texts = [' '.join(words) for words in masked_words]  # list of text of masked words for evidence only 
    all_input_ids = []
    all_masks = []
    all_segs = []
    for text in texts:
        evidence_tokens = tokenizer.tokenize(text)
        input_ids, input_mask, segment_ids = prepare_tokenization(sub_words_claim, evidence_tokens, max_length, tokenizer, padding=True, max_evidence_len=max_evidence_len)         
        all_input_ids.append(input_ids)
        all_masks.append(input_mask)
        all_segs.append(segment_ids)
        
    inputs = torch.tensor(all_input_ids, dtype=torch.long).to('cuda')
    masks = torch.tensor(all_masks, dtype=torch.long).to('cuda')
    segments = torch.tensor(all_segs, dtype=torch.long).to('cuda')
    eval_data = CustomDataset(inputs,masks,segments)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)
    leave_1_probs = []
    for batch in eval_dataloader:
        input_batch,masks_batch,segment_batch = batch
        bs = input_batch.size(0)
        leave_1_prob_batch = tgt_model(input_batch,masks_batch,segment_batch)  # B num-label
        leave_1_probs.append(leave_1_prob_batch)
    leave_1_probs = torch.cat(leave_1_probs, dim=0)  # words, num-label
    leave_1_probs = torch.softmax(leave_1_probs, -1)  #
    leave_1_probs_argmax = torch.argmax(leave_1_probs, dim=-1)
    import_scores = (orig_prob
                     - leave_1_probs[:, orig_label]
                     +
                     (leave_1_probs_argmax != orig_label).float()
                     * (leave_1_probs.max(dim=-1)[0] - torch.index_select(orig_probs, 0, leave_1_probs_argmax))
                     ).data.cpu().numpy()
    return import_scores

  # --- Constants ---

# Zero width space
ZWSP = chr(0x200B)
# Zero width joiner
ZWJ = chr(0x200D)
# Zero width non-joiner
ZWNJ = chr(0x200C)
# Unicode Bidi override characters
PDF = chr(0x202C)
LRE = chr(0x202A)
RLE = chr(0x202B)
LRO = chr(0x202D)
RLO = chr(0x202E)
PDI = chr(0x2069)
LRI = chr(0x2066)
RLI = chr(0x2067)
# Backspace character
BKSP = chr(0x8)
# Delete character
DEL = chr(0x7F)
# Carriage return character
CR = chr(0xD)

# Load Unicode Intentional homoglyph characters
intentionals = dict()
with open("intentional.txt", "r") as f:
    for line in f.readlines():
        if len(line.strip()):
            if line[0] != '#':
                line = line.replace("#*", "#")
                _, line = line.split("#", maxsplit=1)
                if line[3] not in intentionals:
                    intentionals[line[3]] = []
                intentionals[line[3]].append(line[7])


label_map =  {'SUPPORTS': 0, 'REFUTES': 1, 'NOT ENOUGH INFO': 2, 'none': 3}

def process_sent(sentence):
    sentence = re.sub(" \-LSB\-.*?\-RSB\-", "", sentence)
    sentence = re.sub("\-LRB\- \-RRB\- ", "", sentence)
    sentence = re.sub(" -LRB-", " ( ", sentence)
    sentence = re.sub("-RRB-", " )", sentence)
    sentence = re.sub("--", "-", sentence)
    sentence = re.sub("``", '"', sentence)
    sentence = re.sub("''", '"', sentence)
    return sentence
    
def read_file(data_path):
    features = list()
    with open(data_path) as fin:
        for step, line in enumerate(fin):
            sublines = line.strip().split("\t")
            claim = process_sent(sublines[0])
            evidence = process_sent(sublines[2])
            title = sublines[1]
            label = sublines[3]
            example_label = sublines[4]
            if example_label == 'none': example_label = label
            features.append({'claim':claim, 'title': title, 'evidence': evidence, 'gold_label': example_label, 'id_in_fever': sublines[5], 'id_in_file': step}) 
    return features
    
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
            
def _tokenize(claim, evidence, tokenizer, arch_type):
    claim = claim.lower()
    evidence = evidence.lower()
    tokens_claim = tokenizer.tokenize(claim)
    tokens_evidence = tokenizer.tokenize(evidence)
    #_truncate_seq_pair(tokens_claim, tokens_evidence, max_seq_length - 3)

    if arch_type == 'bert':
        tokens = ["[CLS]"] + tokens_claim + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        tokens = tokens + tokens_evidence + ["[SEP]"]
        segment_ids += [1] * (len(tokens_evidence) + 1)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = torch.tensor(segment_ids).cuda()
    elif arch_type == 'roberta':
        tokens = [tokenizer.cls_token] + tokens_claim + [tokenizer.sep_token]
        tokens = tokens + tokens_evidence + [tokenizer.sep_token]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = None     
    input_ids = torch.tensor(input_ids).cuda()
    input_mask = torch.tensor(input_mask).cuda()
    return input_ids, input_mask, segment_ids 

def target_model_predict(model, input_ids, input_mask, segment_ids, arch_type):
    if arch_type == 'bert': logits = model(input_ids.unsqueeze(0),input_mask.unsqueeze(0),segment_ids.unsqueeze(0))
    elif arch_type == 'roberta': logits = model(input_ids.unsqueeze(0),input_mask.unsqueeze(0))
    return logits 

# --- Classes ---

class Swap():
    """Represents swapped elements in a string of text."""
    def __init__(self, one, two):
        self.one = one
        self.two = two
    
    def __repr__(self):
        return f"Swap({self.one}, {self.two})"

    def __eq__(self, other):
        return self.one == other.one and self.two == other.two

    def __hash__(self):
        return hash((self.one, self.two))


class Objective(ABC):
  """ Abstract class representing objectives for scipy's genetic algorithms."""

  def __init__(self, model, input: str, ref_translation: str, max_perturbs: int, distance: Callable[[str,str],int]):
    if not model:
      raise ValueError("Must supply model.")
    if not input:
      raise ValueError("Must supply input.")

    self.model = model
    self.input: str = input
    self.ref_translation = ref_translation
    self.max_perturbs: int = max_perturbs
    self.distance: Callable[[str,str],int] = distance
    self.output = self.model.translate(self.input)




class InvisibleCharacterObjective(Objective):
  """Class representing an Objective which injects invisible characters."""

  def __init__(self, model, input: str, ref_translation: str, max_perturbs: int, invisible_chrs: List[str] = [ZWJ,ZWSP,ZWNJ], distance: Callable[[str,str],int] = levenshtein.distance):
    super().__init__(model, input, ref_translation, max_perturbs, distance)
    self.invisible_chrs: List[str] = invisible_chrs

  def bounds(self) -> List[Tuple[float, float]]:
    return [(0,len(self.invisible_chrs)-1), (-1, len(self.input)-1)] * self.max_perturbs

  def candidate(self, perturbations: List[float]) -> str:
    candidate = [char for char in self.input]
    for i in range(0, len(perturbations), 2):
      inp_index = integer(perturbations[i+1])
      if inp_index >= 0:
        inv_char = self.invisible_chrs[natural(perturbations[i])]
        candidate = candidate[:inp_index] + [inv_char] + candidate[inp_index:]
    return ''.join(candidate)


class HomoglyphObjective(Objective):

  def __init__(self, model, input: str, ref_translation: str, max_perturbs: int, distance: Callable[[str,str],int] = levenshtein.distance, homoglyphs: Dict[str,List[str]] = intentionals):
    super().__init__(model, input, ref_translation, max_perturbs, distance)
    self.homoglyphs = homoglyphs
    self.glyph_map = []
    for i, char in enumerate(self.input):
      if char in self.homoglyphs:
        charmap = self.homoglyphs[char]
        charmap = list(zip([i] * len(charmap), charmap))
        self.glyph_map.extend(charmap)

  def bounds(self) -> List[Tuple[float, float]]:
    return [(-1, len(self.glyph_map)-1)] * self.max_perturbs

  def candidate(self, perturbations: List[float]) -> str:
    candidate = [char for char in self.input]  
    for perturb in map(integer, perturbations):
      if perturb >= 0:
        i, char = self.glyph_map[perturb]
        candidate[i] = char
    return ''.join(candidate)


class ReorderObjective(Objective):

  def __init__(self, model, input: str, ref_translation: str, max_perturbs: int, distance: Callable[[str,str],int] = levenshtein.distance):
    super().__init__(model, input, ref_translation, max_perturbs, distance)

  def bounds(self) -> List[Tuple[float, float]]:
    return [(-1,len(self.input)-1)] * self.max_perturbs

  def candidate(self, perturbations: List[float]) -> str:
    def swaps(els) -> str:
      res = ""
      for el in els:
          if isinstance(el, Swap):
              res += swaps([LRO, LRI, RLO, LRI, el.one, PDI, LRI, el.two, PDI, PDF, PDI, PDF])
          elif isinstance(el, str):
              res += el
          else:
              for subel in el:
                  res += swaps([subel])
      return res

    _candidate = [char for char in self.input]
    for perturb in map(integer, perturbations):
      if perturb >= 0 and len(_candidate) >= 2:
        perturb = min(perturb, len(_candidate) - 2)
        _candidate = _candidate[:perturb] + [Swap(_candidate[perturb+1], _candidate[perturb])] + _candidate[perturb+2:]

    return swaps(_candidate)


class DeletionObjective(Objective):
  """Class representing an Objective which injects deletion control characters."""

  def __init__(self, model, input: str, ref_translation: str, max_perturbs: int, distance: Callable[[str,str],int] = levenshtein.distance, del_chr: str = BKSP, ins_chr_min: str = '!', ins_chr_max: str = '~'):
    super().__init__(model, input, ref_translation, max_perturbs, distance)
    self.del_chr: str = del_chr
    self.ins_chr_min: str = ins_chr_min
    self.ins_chr_max: str = ins_chr_max

  def bounds(self) -> List[Tuple[float, float]]:
    return [(-1,len(self.input)-1), (ord(self.ins_chr_min),ord(self.ins_chr_max))] * self.max_perturbs

  def candidate(self, perturbations: List[float]) -> str:
    candidate = [char for char in self.input]
    for i in range(0, len(perturbations), 2):
      idx = integer(perturbations[i])
      if idx >= 0:
        char = chr(natural(perturbations[i+1]))
        candidate = candidate[:idx] + [char, self.del_chr] + candidate[idx:]
        for j in range(i,len(perturbations), 2):
          perturbations[j] += 2
    return ''.join(candidate)


class MnliObjective():

  def __init__(self, model, arch_type, tokenizer, input: str, claim: str, label:int, max_perturbs: int):
    if not model:
      raise ValueError("Must supply model.")
    if not input:
      raise ValueError("Must supply input.")
    if not claim:
      raise ValueError("Must supply claim.")
    if label == None:
      raise ValueError("Must supply label.")
    self.model = model
    self.input: str = input
    self.claim: str = claim
    self.label: int = label
    self.max_perturbs: int = max_perturbs
    self.arch_type = arch_type 
    self.tokenizer = tokenizer 

  def objective(self) -> Callable[[List[float]], float]:
    def _objective(perturbations: List[float]) -> float:
      candidate: str = self.candidate(perturbations)
      input_ids, input_mask, segment_ids = _tokenize(self.claim, candidate, self.tokenizer, self.arch_type)
      predict = target_model_predict(self.model, input_ids, input_mask, segment_ids, self.arch_type) ##predict is the score of the sentence. we want to minize this.
      return predict.cpu().detach().numpy()[0]
    return _objective

  def differential_evolution(self, verbose=False, maxiter=3, popsize=32, polish=False) -> str:
    start = process_time()
    result = differential_evolution(self.objective(), self.bounds(),
                                    disp=verbose, maxiter=maxiter,
                                    popsize=popsize, polish=polish)
    end = process_time()
    candidate = self.candidate(result.x)
    input_ids, input_mask, segment_ids = _tokenize(self.claim, candidate, self.tokenizer, self.arch_type)
    predict = target_model_predict(self.model, input_ids, input_mask, segment_ids, self.arch_type)    
    probs = predict.cpu().detach().numpy()[0]
    orig_input_ids, orig_input_mask, orig_segment_ids = _tokenize(self.claim, self.input, self.tokenizer, self.arch_type)
    inp_predict = target_model_predict(self.model, orig_input_ids, orig_input_mask, orig_segment_ids, self.arch_type)   
    inp_probs = inp_predict.cpu().detach().numpy()[0]
    return  {
              'adv_example': candidate,
              'adv_example_enc': result.x,
              'input': self.input,
              'claim': self.claim,
              'correct_label_index': self.label,
              'adv_predictions': probs,
              'input_prediction': inp_probs,
              'attack_success': True,
              'adv_generation_time': end - start,
              'budget': self.max_perturbs,
              'maxiter': maxiter,
              'popsize': popsize
            }


class InvisibleCharacterMnliObjective(MnliObjective, InvisibleCharacterObjective):
  
  def __init__(self, model, arch_type, tokenizer, input: str, claim: str, label:int, max_perturbs: int, invisible_chrs: List[str] = [ZWJ,ZWSP,ZWNJ]):
    super().__init__(model, arch_type, tokenizer, input, claim, label, max_perturbs)
    self.invisible_chrs = invisible_chrs


class HomoglyphMnliObjective(MnliObjective, HomoglyphObjective):
  
  def __init__(self, model, arch_type, tokenizer, input: str, claim: str, label:int, max_perturbs: int, homoglyphs: Dict[str,List[str]] = intentionals):
    super().__init__(model, arch_type, tokenizer, input, claim, label, max_perturbs)
    self.homoglyphs = homoglyphs
    self.glyph_map = []
    for i, char in enumerate(self.input):
      if char in self.homoglyphs:
        charmap = self.homoglyphs[char]
        charmap = list(zip([i] * len(charmap), charmap))
        self.glyph_map.extend(charmap)


class ReorderMnliObjective(MnliObjective, ReorderObjective):
  
  def __init__(self, model, arch_type, tokenizer, input: str, claim: str, label:int, max_perturbs: int):
    super().__init__(model, arch_type, tokenizer, input, claim, label, max_perturbs)


class DeletionMnliObjective(MnliObjective, DeletionObjective):
  
  def __init__(self, model, arch_type, tokenizer, input: str, claim: str, label:int, max_perturbs: int, del_chr: str = BKSP, ins_chr_min: str = '!', ins_chr_max: str = '~'):
    super().__init__(model, arch_type, tokenizer, input, claim, label, max_perturbs)
    self.del_chr: str = del_chr
    self.ins_chr_min: str = ins_chr_min
    self.ins_chr_max: str = ins_chr_max

# --- Helper Functions ---

def some(*els):
    """Returns the arguments as a tuple with Nones removed."""
    return tuple(filter(None, tuple(els)))

def swaps(chars: str) -> set:
    """Generates all possible swaps for a string."""
    def pairs(chars, pre=(), suf=()):
        orders = set()
        for i in range(len(chars)-1):
            prefix = pre + tuple(chars[:i])
            suffix = suf + tuple(chars[i+2:])
            swap = Swap(chars[i+1], chars[i])
            pair = some(prefix, swap, suffix)
            orders.add(pair)
            orders.update(pairs(suffix, pre=some(prefix, swap)))
            orders.update(pairs(some(prefix, swap), suf=suffix))
        return orders
    return pairs(chars) | {tuple(chars)}

def unswap(el: tuple) -> str:
    """Reverts a tuple of swaps to the original string."""
    if isinstance(el, str):
        return el
    elif isinstance(el, Swap):
        return unswap((el.two, el.one))
    else:
        res = ""
        for e in el:
            res += unswap(e)
        return res

def uniswap(els):
    res = ""
    for el in els:
        if isinstance(el, Swap):
            res += uniswap([LRO, LRI, RLO, LRI, el.one, PDI, LRI, el.two, PDI, PDF, PDI, PDF])
        elif isinstance(el, str):
            res += el
        else:
            for subel in el:
                res += uniswap([subel])
    return res

def natural(x: float) -> int:
    """Rounds float to the nearest natural number (positive int)"""
    return max(0, round(float(x)))

def integer(x: float) -> int:
    """Rounds float to the nearest int"""
    return round(float(x))

def detokenize(tokens: List[str]) -> str:
  output = ""
  for index, token in enumerate(tokens):
    if (len(token) == 1 and token in punctuation) or index == 0:
      output += token
    else:
      output += ' ' + token
  return output


def mnli_experiment(model, objective, data, file, min_budget, max_budget, maxiter, popsize, exp_label, overwrite,tokenizer,arch_type, max_seq_length):
  if overwrite or not exists(file):
    perturbs = { exp_label: { '0': dict() } }
  else:
    with open(file, 'rb') as f:
      perturbs = pickle.load(f)
    if exp_label not in perturbs:
      perturbs[exp_label] = dict()
    if '0' not in perturbs[exp_label]:
      perturbs[exp_label]['0'] = dict()
  for test in data:
    if test['id_in_file'] not in perturbs[exp_label]['0']:
      #change tokenization and model prediction 
      test['claim'] = test['claim'].split(' ')
      test['evidence'] = test['evidence'].split(' ')
      _truncate_seq_pair(test['claim'],test['evidence'],max_seq_length) ##need to test this.
      test['claim'] = ' '.join(test['claim'])
      test['evidence'] = ' '.join(test['evidence'])
      input_ids, input_mask, segment_ids = _tokenize(test['claim'], test['evidence'], tokenizer, arch_type)
      probs = target_model_predict(model, input_ids, input_mask, segment_ids, arch_type)
      probs = probs.cpu().detach().numpy()[0]
      label = label_map[test['gold_label']]
      perturbs[exp_label]['0'][test['id_in_file']] = {
          'adv_example': test['evidence'],
          'adv_example_enc': [],
          'input': test['evidence'],
          'claim': test['claim'],
          'correct_label_index': label,
          'adv_predictions': probs,
          'input_prediction': probs,
          'adv_generation_time': 0,
          'budget': 0,
          'maxiter': maxiter,
          'popsize': popsize
        }
  attack_attempted = 0 
  attack_success = 0 
  with tqdm(total=len(data)*(max_budget-min_budget+1), desc="Adv. Examples") as pbar:
    for budget in range(min_budget, max_budget+1):
      if str(budget) not in perturbs[exp_label]:
        perturbs[exp_label][str(budget)] = dict()
      for test in data:
        if perturbs[exp_label]['0'][test['id_in_file']]['correct_label_index'] == label_map['NOT ENOUGH INFO']: 
          sleep(0.1)
          pbar.update(1)
          continue ##don't target NEI 
        if test['id_in_file'] not in perturbs[exp_label][str(budget)]:
          attack_attempted += 1 
          obj = objective(model, arch_type, tokenizer, test['evidence'], test['claim'], label_map[test['gold_label']], budget)
          example = obj.differential_evolution(maxiter=maxiter, popsize=popsize)
          perturbs[exp_label][str(budget)][test['id_in_file']] = example
          with open(file, 'wb') as f:
            pickle.dump(perturbs, f)
        else:
          # Required for progress bar to update correctly
          sleep(0.1)
        pbar.update(1)
  return attack_attempted



def load_retrieval(model_type,model_checkpt):
  # Load pre-trained MNLI model
  print("Loading Fever classification model.")
  if model_type == 'bert':   
    tokenizer = BertTokenizer.from_pretrained(args.bert_pretrain, do_lower_case=False)
    base_bert_model = BertForSequenceEncoder.from_pretrained(args.bert_pretrain)
    base_bert_model = base_bert_model.cuda()
    model = inference_model(base_bert_model, args)
    model.load_state_dict(torch.load(args.target_model_chkpt)['model'])
    model = model.cuda()
    model.eval()
    print("Model loaded successfully.")
  if model_type == 'roberta':    
    print('Retrieval model has to be BERT')
  return tokenizer, model 


# -- CLI ---

if __name__ == '__main__':
  parser = ArgumentParser(description='Adversarial NLP Experiments.')
  technique = parser.add_mutually_exclusive_group(required=True)
  technique.add_argument('-i', '--invisible-chars', action='store_true', help="Use invisible character perturbations.")
  technique.add_argument('-g', '--homoglyphs', action='store_true', help="Use homoglyph perturbations.")
  technique.add_argument('-r', '--reorderings', action='store_true', help="Use reordering perturbations.")
  technique.add_argument('-d', '--deletions', action='store_true', help="Use deletion perturbations.")
  parser.add_argument('-c', '--cpu', action='store_true', help="Use CPU for ML inference instead of CUDA.")
  parser.add_argument('--bert_pretrain', required=True)
  parser.add_argument('--pkl_file', help="File to contain Python pickled output.")
  parser.add_argument('--target_model_arch', type=str, default='bert')
  parser.add_argument("--bert_hidden_dim", default=768, type=int, help="bert dim")
  parser.add_argument('--dropout', type=float, default=0.6, help='Dropout.')
  parser.add_argument("--num_labels", type=int, default=3)
  parser.add_argument('--target_model_chkpt', type=str)
  parser.add_argument('--data_path', type=str)
  parser.add_argument('--start_idx', type=int,default=0)
  parser.add_argument('--end_idx', type=int,default=-1)
  parser.add_argument('--max_len', type=int,default=150,help='maximum sequence length in words')
  parser.add_argument('-n', '--num-examples', type=int, default=500, help="Number of adversarial examples to generate.")
  parser.add_argument('-l', '--min-perturbs', type=int, default=1, help="The lower bound (inclusive) of the perturbation budget range.")
  parser.add_argument('-u', '--max-perturbs', type=int, default=5, help="The upper bound (inclusive) of the perturbation budget range.")
  parser.add_argument('-a', '--maxiter', type=int, default=10, help="The maximum number of iterations in the genetic algorithm.")
  parser.add_argument('-p', '--popsize', type=int, default=32, help="The size of the population in the genetic algorithm.")
  parser.add_argument('-o', '--overwrite', action='store_true', help="Overwrite existing results file instead of resuming.")
  targeted = parser.add_mutually_exclusive_group()
  targeted.add_argument('-x', '--targeted', action='store_true', help="Perform a targeted attack.")
  targeted.add_argument('-X', '--targeted-no-logits', action='store_true', help="Perform a targeted attack without access to inference result logits.")
  args = parser.parse_args()


tokenizer, retrieval_model = load_retrieval(args.target_model_arch, args.target_model_chkpt)
mnli_test = read_file(args.data_path)
start = args.start_idx 
end = args.end_idx if args.end_idx!= -1 else len(mnli_test)
data = mnli_test[start:end]
print('Start index: '+str(start))
print('End index: '+str(end))
print(f"Loaded {len(data)} strings from corpus.")
print('*****')

if args.targeted:
    print('Retrieval-based cannot be targeted')
    exit() 
else:
    if args.invisible_chars:
        print(f"Starting invisible characters MNLI experiment.")
        objective = InvisibleCharacterMnliObjective
        label = "mnli_invisibles_untargeted"
    elif args.homoglyphs:
        print(f"Starting homoglyphs MNLI experiment.")
        objective = HomoglyphMnliObjective
        label = "mnli_homoglyphs_untargeted"
    elif args.reorderings:
        print(f"Starting reorderings MNLI experiment.")
        objective = ReorderMnliObjective
        label = "mnli_reorderings_untargeted"
    elif args.deletions:
        print(f"Starting deletions MNLI experiment.")
        objective = DeletionMnliObjective
        label = "mnli_deletions_untargeted"

    success_ratio = mnli_experiment(retrieval_model, objective, data, args.pkl_file, args.min_perturbs, args.max_perturbs, args.maxiter, args.popsize, label, args.overwrite,tokenizer,args.target_model_arch, args.max_len)

print(f"Experiment complete. Results written to {args.pkl_file}.")
print('Success ratio out of attempted attacks: '+str(success_ratio))
