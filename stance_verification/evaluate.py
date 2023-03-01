import random, os
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.autograd import Variable
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
#from pytorch_pretrained_bert.optimization import BertAdam

from model import stance_model
from data_loader import DataLoader
from transformers import RobertaTokenizer, RobertaModel
import logging
import json
import torch.nn as nn

logger = logging.getLogger(__name__)

def correct_prediction(prob,labels):
    correct = 0.0
    _,pred_lbls = torch.max(prob, dim=1)
    correct = pred_lbls.eq(labels).sum()

    #indices of zeros     
    index_zeros = ((labels == 0).nonzero(as_tuple=True)[0])
    tot_zeros = len(index_zeros) 
    correct_zeros = pred_lbls[index_zeros].eq(labels[index_zeros]).sum().item()
    #indices of ones     
    index_ones = ((labels == 1).nonzero(as_tuple=True)[0])
    tot_ones = len(index_ones)
    correct_ones = pred_lbls[index_ones].eq(labels[index_ones]).sum().item() 
    #indices of twos     
    index_two = ((labels == 2).nonzero(as_tuple=True)[0])
    tot_twos = len(index_two)
    correct_two = pred_lbls[index_two].eq(labels[index_two]).sum().item() 
    return correct, correct_zeros, correct_ones, correct_two, tot_zeros, tot_ones, tot_twos

def eval_model(model, validset_reader):
    model.eval()
    correct_pred = 0.0
    correct_zeros = 0.0 
    correct_ones = 0.0 
    correct_twos = 0.0 
    
    tot_zeros = 0.0 
    tot_ones = 0.0 
    tot_twos = 0.0 
  
    for inp_tensor, msk_tensor, labels_int_tensor in validset_reader:
        with torch.no_grad():
            prob = model(inp_tensor, msk_tensor)
            returned = correct_prediction(prob, labels_int_tensor)
            correct_pred += returned[0] 
            correct_zeros += returned[1] 
            correct_ones += returned[2] 
            correct_twos += returned[3] 
            
            tot_zeros += returned[4] 
            tot_ones += returned[5] 
            tot_twos += returned[6] 

    dev_accuracy = correct_pred / validset_reader.total_num
    class0_accuracy = correct_zeros / tot_zeros
    class1_accuracy = correct_ones / tot_ones
    class2_accuracy = correct_twos / tot_twos
    return dev_accuracy, class0_accuracy, class1_accuracy, class2_accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout.')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--valid_path', help='valid path')
    parser.add_argument("--roberta_hidden_dim", default=768, type=int, help="Total batch size for training.")
    parser.add_argument("--valid_batch_size", default=8, type=int, help="Total batch size for predictions.")
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument("--num_labels", type=int, default=3)
    parser.add_argument("--max_len", default=120, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--eval_step", default=2000, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    args = parser.parse_args()
    labels_dict = {'SUPPORTS': 0, 'REFUTES': 1, 'NOT ENOUGH INFO': 2}

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base") 
    logger.info("loading validation set")
    validset_reader = DataLoader(args.valid_path, tokenizer, args, labels_dict, batch_size=args.valid_batch_size, test=True)
    roberta_model = RobertaModel.from_pretrained("roberta-base")
    roberta_model = roberta_model.cuda()
    model = stance_model(roberta_model, args)
    model.load_state_dict(torch.load(args.checkpoint)['model'])
    model = model.cuda()  
    tot_acc, class0_acc, class1_acc, class2_acc = eval_model(model, validset_reader)
    print('Accuracy: '+str(tot_acc))
    print('Class 0 - Supports: '+str(class0_acc))  
    print('Class 1 - Refutes: '+str(class1_acc))      
    print('Class 2 - NEI: '+str(class2_acc))      