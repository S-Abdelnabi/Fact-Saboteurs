import random, os
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.autograd import Variable
from pytorch_pretrained_bert.tokenization import whitespace_tokenize, BasicTokenizer, BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
from models import inference_model
from data_loader_add import DataLoaderTest
from bert_model import BertForSequenceEncoder
from torch.nn import NLLLoss
import logging
import json

logger = logging.getLogger(__name__)

def save_to_file(all_predict, outpath):
    with open(outpath, "w") as out:
        for key, values in all_predict.items():
            sorted_values = sorted(values, key=lambda x:x[-1], reverse=True)
            data = json.dumps({"id": key, "evidence": sorted_values[:5]})
            out.write(data + "\n")
            
#read sentences that were changed only 
def read_attack_file(attack_data_path):
    evidence_per_id = dict()
    with open(attack_data_path) as fin:
        for step, line in enumerate(fin):
            instance = json.loads(line.strip())
            claim = instance['claim']
            id = int(instance['id'])
            evidence_list = []
            attack_success = []
            for evidence in instance['evidence']:
                if int(evidence[3]) == 1 or int(evidence[3]) == 2:
                    evidence_list.append(evidence[2])
                    attack_success.append(1) #evidence[3] is whether attack succeeded.
            evidence_per_id[id] = {'evidence':evidence_list, 'success':attack_success}
    return evidence_per_id

def get_highest_evidence(all_predict_one_id):
    sorted_values = sorted(all_predict_one_id, key=lambda x:x[-1], reverse=True)
    highest_5 = sorted_values[:5]
    highest_5_evi_only = [evi[2] for evi in highest_5]
    return highest_5_evi_only

def check_against_top5(highest_5, evi_group):
    #array_to_check against the retrieved. 
    count_retrieved = 0 
    count_retrieved_success = 0 
    evi_list_to_check = evi_group['evidence']
    attack_success = evi_group['success']   
    tot_success = np.sum(np.asarray(attack_success)) if len(attack_success) != 0 else 0 
    tot_evidence = len(evi_list_to_check)
    for i in range(0,len(evi_list_to_check)):
        evi = evi_list_to_check[i]
        evi_success = attack_success[i] 
        if evi in highest_5:            
            count_retrieved += 1 #how many sentences in attack file were retrieved. 
            if evi_success == 1: count_retrieved_success += 1 #how many successful attack were retrieved
    return count_retrieved, count_retrieved_success, tot_evidence, tot_success 

def process_all_predict(all_predict_retrieved,evidence_to_check):    
    count_in_top5 = 0
    count_success_in_top5 = 0 
    tot_evidence_num = 0 
    tot_success_num = 0 
    for key in evidence_to_check.keys():
        #if not key in all_predict_retrieved.keys(): continue #for debugging, remove after. 
        all_evi = all_predict_retrieved[key]  
        highest_5 = get_highest_evidence(all_evi)
        returned_list = check_against_top5(highest_5, evidence_to_check[key])
        count_in_top5 += returned_list[0]
        count_success_in_top5 += returned_list[1]
        print('key: ' +str(key)+ ', same sentences: '+str(returned_list[0]))
        tot_evidence_num += returned_list[2]
        tot_success_num += returned_list[3]
    return count_in_top5/tot_evidence_num, count_success_in_top5/tot_success_num
    
def eval_model(model, validset_reader):
    model.eval()
    all_predict = dict()
    #count = 0 
    for inp_tensor, msk_tensor, seg_tensor, ids, evi_list in validset_reader:
        #count += 1 
        probs = model(inp_tensor, msk_tensor, seg_tensor)
        probs = probs.tolist()
        assert len(probs) == len(evi_list)
        for i in range(len(probs)):
            if ids[i] not in all_predict:
                all_predict[ids[i]] = []
            all_predict[ids[i]].append(evi_list[i] + [probs[i]]) 
        #if count == 1000: break # for debugging 
    return all_predict



parser = argparse.ArgumentParser()
parser.add_argument('--exclude_gold', action='store_true')
parser.add_argument('--test_path_all_data', default='../data/all_dev.json', help='train path')
parser.add_argument('--test_path_preattack', default='../data/bert_eval2.json', help='train path')
parser.add_argument('--test_path_postattack', default='../data/lexical_attack/lexical_eval2_kgat.json', help='train path')
parser.add_argument('--name', default='dev_eval2_intermediate.json', help='train path')
parser.add_argument("--batch_size", default=32, type=int, help="Total batch size for training.")
parser.add_argument('--outdir', default='./output/', help='path to output directory')
parser.add_argument('--bert_pretrain', default='../bert_base')
parser.add_argument('--checkpoint', default='../checkpoint/retrieval_model/model.best.pt')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout.')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument("--bert_hidden_dim", default=768, type=int, help="Total batch size for training.")
parser.add_argument("--layer", type=int, default=1, help='Graph Layer.')
parser.add_argument("--num_labels", type=int, default=3)
parser.add_argument("--evi_num", type=int, default=50, help='Evidence num. to add')
parser.add_argument("--threshold", type=float, default=0.0, help='Evidence num.')
parser.add_argument("--max_len", default=120, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
args = parser.parse_args()

if not os.path.exists(args.outdir):
    os.mkdir(args.outdir)
args.cuda = not args.no_cuda and torch.cuda.is_available()
handlers = [logging.FileHandler(os.path.abspath(args.outdir) + '/train_log.txt'), logging.StreamHandler()]
logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.DEBUG,
                        datefmt='%d-%m-%Y %H:%M:%S', handlers=handlers)
logger.info(args)
logger.info('Start training!')

tokenizer = BertTokenizer.from_pretrained(args.bert_pretrain, do_lower_case=False)
logger.info("loading training set")
validset_reader = DataLoaderTest(args.test_path_all_data, args.test_path_preattack, args.test_path_postattack, tokenizer, args, batch_size=args.batch_size, n_sent = args.evi_num,exclude_gold = args.exclude_gold)

logger.info('initializing estimator model')
bert_model = BertForSequenceEncoder.from_pretrained(args.bert_pretrain)
bert_model = bert_model.cuda()
model = inference_model(bert_model, args)
model.load_state_dict(torch.load(args.checkpoint)['model'])
model = model.cuda()
logger.info('Start eval!')
save_path = args.outdir + "/" + args.name
predict_dict = eval_model(model, validset_reader)
save_to_file(predict_dict, save_path)
logger.info('Start processing!')
attack_sentences = read_attack_file(args.test_path_postattack)
percentage_retrieved, percentaged_retrieved_perturbed = process_all_predict(predict_dict,attack_sentences)
print('Retrieved: '+str(percentage_retrieved)+' of sentences in: '+args.test_path_postattack)
print('Retrieved: '+str(percentaged_retrieved_perturbed)+' of perturbed sentences in: '+args.test_path_postattack)