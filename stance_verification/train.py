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

from model import inference_model
from data_loader import DataLoader
from transformers import BertTokenizer, BertModel
import logging
import json
import torch.nn as nn

logger = logging.getLogger(__name__)


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    if 1.0 - x >= 0.0:
        return 1.0 - x
    return 0.0

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
  
    for inp_tensor, msk_tensor, seg_tensor, labels_int_tensor in validset_reader:
        with torch.no_grad():
            prob = model(inp_tensor, msk_tensor, seg_tensor)
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

def train_model(model, args, trainset_reader, validset_reader):
    save_path = args.outdir + '/model'
    best_acc = 0.0
    running_loss = 0.0
    t_total = int(
        trainset_reader.total_num / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
    tot_batches = int(trainset_reader.total_num / args.train_batch_size * args.num_train_epochs)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)   
    #BertAdam(optimizer_grouped_parameters,lr=args.learning_rate,warmup=args.warmup_proportion,t_total=t_total)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(t_total*args.warmup_proportion), num_training_steps =t_total)
    global_step = 0
    crit = nn.CrossEntropyLoss()
    for epoch in range(int(args.num_train_epochs)):
        optimizer.zero_grad()
        for inp_tensor, msk_tensor, seg_tensor, labels_int_tensor in trainset_reader:
            model.train()
            score = model(inp_tensor, msk_tensor, seg_tensor)
            loss = crit(score, labels_int_tensor)
            running_loss += loss.item()
            loss.backward()
            global_step += 1
            if global_step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                logger.info('Epoch: {0}, Step: {1}/{2}, Loss: {3}'.format(epoch, global_step,tot_batches, (running_loss / global_step)))
            if global_step % (args.eval_step * args.gradient_accumulation_steps) == 0:
                logger.info('Start eval!')
                eval_acc,class0_acc,class1_acc,class2_acc = eval_model(model, validset_reader)
                logger.info('Dev acc: {0}'.format(eval_acc))
                logger.info('Class 0 - Supports: {0}'.format(class0_acc))
                logger.info('Class 1 - Refutes: {0}'.format(class1_acc))
                logger.info('Class 2 - NEI: {0}'.format(class2_acc))      
                if eval_acc >= best_acc:
                    best_acc = eval_acc
                    torch.save({'epoch': epoch,
                                'model': model.state_dict()}, save_path + ".best.pt")
                    logger.info("Saved best epoch {0}, best acc {1}".format(epoch, best_acc))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout.')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--train_path', help='train path')
    parser.add_argument('--valid_path', help='valid path')
    parser.add_argument("--train_batch_size", default=16, type=int, help="Total batch size for training.")
    parser.add_argument("--bert_hidden_dim", default=768, type=int, help="Total batch size for training.")
    parser.add_argument("--valid_batch_size", default=8, type=int, help="Total batch size for predictions.")
    parser.add_argument('--outdir', required=True, help='path to output directory')
    parser.add_argument("--pool", type=str, default="att", help='Aggregating method: top, max, mean, concat, att, sum')
    parser.add_argument("--layer", type=int, default=1, help='Graph Layer.')
    parser.add_argument("--num_labels", type=int, default=3)
    parser.add_argument("--evi_num", type=int, default=5, help='Evidence num.')
    parser.add_argument("--max_len", default=240, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--eval_step", default=2000, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=2.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% "
                             "of training.")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=4,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    args = parser.parse_args()
    labels_dict = {'SUPPORTS': 0, 'REFUTES': 1, 'NOT ENOUGH INFO': 2}

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    handlers = [logging.FileHandler(os.path.abspath(args.outdir) + '/train_log.txt'), logging.StreamHandler()]
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.DEBUG,
                        datefmt='%d-%m-%Y %H:%M:%S', handlers=handlers)
    logger.info(args)
    logger.info('Start training!')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') 
    logger.info("loading training set")
    trainset_reader = DataLoader(args.train_path, tokenizer, args, labels_dict, batch_size=args.train_batch_size)
    logger.info("loading validation set")
    validset_reader = DataLoader(args.valid_path, tokenizer, args, labels_dict, batch_size=args.valid_batch_size, test=True)

    logger.info('initializing estimator model')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    bert_model = bert_model.cuda()
    model = inference_model(bert_model, args)
    model = model.cuda()
    train_model(model, args, trainset_reader, validset_reader)