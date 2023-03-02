import warnings
import os

import torch
import torch.nn as nn
import json
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset, Dataset
from transformers import BertConfig, BertTokenizer
from transformers import BertForSequenceClassification, BertForMaskedLM
from transformers import BertModel

import copy
import argparse
import numpy as np
import re

from bert_stance.model import inference_model


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)

filter_words = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost',
                'alone', 'along', 'already', 'also', 'although', 'am', 'among', 'amongst', 'an', 'and', 'another',
                'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', "aren't", 'around', 'as',
                'at', 'back', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides',
                'between', 'beyond', 'both', 'but', 'by', 'can', 'cannot', 'could', 'couldn', "couldn't", 'd', 'didn',
                "didn't", 'doesn', "doesn't", 'don', "don't", 'down', 'due', 'during', 'either', 'else', 'elsewhere',
                'empty', 'enough', 'even', 'ever', 'everyone', 'everything', 'everywhere', 'except', 'first', 'for',
                'former', 'formerly', 'from', 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'he', 'hence',
                'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his',
                'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'isn', "isn't", 'it', "it's",
                'its', 'itself', 'just', 'latter', 'latterly', 'least', 'll', 'may', 'me', 'meanwhile', 'mightn',
                "mightn't", 'mine', 'more', 'moreover', 'most', 'mostly', 'must', 'mustn', "mustn't", 'my', 'myself',
                'namely', 'needn', "needn't", 'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none',
                'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only',
                'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'per',
                'please', 's', 'same', 'shan', "shan't", 'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow',
                'something', 'sometime', 'somewhere', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs',
                'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein',
                'thereupon', 'these', 'they', 'this', 'those', 'through', 'throughout', 'thru', 'thus', 'to', 'too',
                'toward', 'towards', 'under', 'unless', 'until', 'up', 'upon', 'used', 've', 'was', 'wasn', "wasn't",
                'we', 'were', 'weren', "weren't", 'what', 'whatever', 'when', 'whence', 'whenever', 'where',
                'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while',
                'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'with', 'within', 'without', 'won',
                "won't", 'would', 'wouldn', "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're", "you've",
                'your', 'yours', 'yourself', 'yourselves',
                '(', ')', '\"']
filter_words = set(filter_words)

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
        

def get_sim_embed(embed_path, sim_path):
    id2word = {}
    word2id = {}

    with open(embed_path, 'r', encoding='utf-8') as ifile:
        for line in ifile:
            word = line.split()[0]
            if word not in id2word:
                id2word[len(id2word)] = word
                word2id[word] = len(id2word) - 1

    cos_sim = np.load(sim_path)
    return cos_sim, word2id, id2word
    
def process_sent(sentence):
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
    sentence = re.sub(' +', ' ', sentence)
    return sentence
LABELS = {'SUPPORTS': 0, 'REFUTES': 1, 'NOT ENOUGH INFO': 2, 'none': 3}

#Read data. Format file: eval_pairs_retrieval2 
def read_file(data_path):
    features = list()
    with open(data_path) as fin:
        for step, line in enumerate(fin):
            sublines = line.strip().split("\t")
            claim = process_sent(sublines[0])
            evidence = process_sent(sublines[2])
            title = sublines[1]
            label = sublines[3]
            label_int = LABELS[label]
            example_label = sublines[4]
            example_label = LABELS[example_label]
            if example_label == 3: example_label = label_int
            features.append([claim, title, evidence, example_label, sublines[5], step]) 
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
            


class Feature(object):
    def __init__(self, claim, title, evidence, label, id_, file_index):
        self.label = label
        self.claim = claim
        self.evidence = evidence
        self.final_adverse = evidence
        self.query = 0
        self.change = 0
        self.success = 0
        self.sim = 0.0
        self.changes = []
        self.id_ = id_
        self.file_index = file_index
        self.title = title


def _tokenize(claim, evidence, tokenizer, max_seq_length):
    claim = claim.lower()
    evidence = evidence.lower()
    words_claim = claim.split(' ')
    words_evidence = evidence.split(' ') 
    sub_words_claim = []
    sub_words_evidence = []
    keys = []
    for word in words_claim:
        sub = tokenizer.tokenize(word)
        sub_words_claim += sub
        ##keys.append([index, index + len(sub)])
        ##index += len(sub)

 ##evidence starts after the claim, consider cls and sep tokens
    index = 0 
    for word in words_evidence:
        sub = tokenizer.tokenize(word)
        sub_words_evidence += sub
        keys.append([index, index + len(sub)])
        index += len(sub)
    _truncate_seq_pair(sub_words_claim, sub_words_evidence, max_seq_length - 3)

    ##truncate the evidence words, according to the previous trunction. 
    word_count = 0 
    sub_index = 0 
    for word in words_evidence:
        sub = tokenizer.tokenize(word)
        sub_index += len(sub)
        if sub_index > len(sub_words_evidence): break ##reached truncation, don't include this word.
        word_count += 1

            
    truncated_word_evidence = words_evidence[0:word_count] 
    start_evidence = len(sub_words_claim) + 2
    return words_claim, truncated_word_evidence, sub_words_claim, sub_words_evidence, keys, start_evidence


def _get_masked(words):
    len_text = len(words)
    masked_words = []
    if len_text == 1: ##special case
        return ['[UNK]']
    for i in range(len_text - 1):
        masked_words.append(words[0:i] + ['[UNK]'] + words[i + 1:])
    # list of words
    return masked_words

def prepare_tokenization(claim_tokens, evidence_tokens, max_length, tokenizer, padding=False,max_evidence_len=512):
    tokens =  ["[CLS]"] + claim_tokens + ["[SEP]"]
    segment_ids = [0] * len(tokens) 
    evidence_tokens = evidence_tokens[0:max_evidence_len]
    tokens = tokens + evidence_tokens + ["[SEP]"]
    segment_ids += [1] * (len(evidence_tokens) + 1)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    if padding: 
        padding = [0] * (max_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
    return input_ids, input_mask, segment_ids 
    
def get_important_scores(sub_words_claim, words_evidence, tgt_model, orig_prob, orig_label, orig_probs, tokenizer, batch_size, max_length):
    masked_words = _get_masked(words_evidence) 
    texts = [' '.join(words) for words in masked_words]  # list of text of masked words for evidence only 
    all_input_ids = []
    all_masks = []
    all_segs = []
    max_evidence_len = max_length - len(sub_words_claim) - 3 
    #print(len(tokenizer.tokenize(' '.join(words_evidence))))
    for text in texts:
        evidence_tokens = tokenizer.tokenize(text)
        input_ids, input_mask, segment_ids = prepare_tokenization(sub_words_claim, evidence_tokens, max_length, tokenizer, padding=True, max_evidence_len=max_evidence_len) 
        #print(max_length)
        #print(len(input_ids))        
        all_input_ids.append(input_ids)
        all_masks.append(input_mask)
        all_segs.append(segment_ids)
        
    inputs = torch.tensor(all_input_ids, dtype=torch.long).to('cuda')
    #print(inputs.size())
    #print(words_evidence)
    #print(texts)
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
        #print(input_batch.size())
        #print(leave_1_prob_batch.size())
        leave_1_probs.append(leave_1_prob_batch)
    leave_1_probs = torch.cat(leave_1_probs, dim=0)  # words, num-label
    leave_1_probs = torch.softmax(leave_1_probs, -1)  #
    #print(leave_1_probs.size())
    leave_1_probs_argmax = torch.argmax(leave_1_probs, dim=-1)
    #print(leave_1_probs_argmax.size())
    #print(orig_prob)
    #print(orig_label)
    import_scores = (orig_prob
                     - leave_1_probs[:, orig_label]
                     +
                     (leave_1_probs_argmax != orig_label).float()
                     * (leave_1_probs.max(dim=-1)[0] - torch.index_select(orig_probs, 0, leave_1_probs_argmax))
                     ).data.cpu().numpy()
    #print('***')
    #print(len(words_evidence))
    #print(words_evidence)                    
    #print(import_scores)
    #print(len(import_scores))
    #print('***')
    return import_scores


def get_substitues(substitutes, tokenizer, mlm_model, use_bpe, substitutes_score=None, threshold=3.0):
    # substitues L,k
    # from this matrix to recover a word
    words = []
    sub_len, k = substitutes.size()  # sub-len, k

    if sub_len == 0:
        return words
        
    elif sub_len == 1:
        for (i,j) in zip(substitutes[0], substitutes_score[0]):
            if threshold != 0 and j < threshold:
                break
            words.append(tokenizer._convert_id_to_token(int(i)))
    else:
        if use_bpe == 1:
            words = get_bpe_substitues(substitutes, tokenizer, mlm_model)
        else:
            return words
    #
    # print(words)
    return words


def get_bpe_substitues(substitutes, tokenizer, mlm_model):
    # substitutes L, k

    substitutes = substitutes[0:12, 0:4] # maximum BPE candidates

    # find all possible candidates 

    all_substitutes = []
    for i in range(substitutes.size(0)):
        if len(all_substitutes) == 0:
            lev_i = substitutes[i]
            all_substitutes = [[int(c)] for c in lev_i]
        else:
            lev_i = []
            for all_sub in all_substitutes:
                for j in substitutes[i]:
                    lev_i.append(all_sub + [int(j)])
            all_substitutes = lev_i

    # all substitutes  list of list of token-id (all candidates)
    c_loss = nn.CrossEntropyLoss(reduction='none')
    word_list = []
    # all_substitutes = all_substitutes[:24]
    all_substitutes = torch.tensor(all_substitutes) # [ N, L ]
    all_substitutes = all_substitutes[:24].to('cuda')
    # print(substitutes.size(), all_substitutes.size())
    N, L = all_substitutes.size()
    word_predictions = mlm_model(all_substitutes)[0] # N L vocab-size
    ppl = c_loss(word_predictions.view(N*L, -1), all_substitutes.view(-1)) # [ N*L ] 
    ppl = torch.exp(torch.mean(ppl.view(N, L), dim=-1)) # N  
    _, word_list = torch.sort(ppl)
    word_list = [all_substitutes[i] for i in word_list]
    final_words = []
    for word in word_list:
        tokens = [tokenizer._convert_id_to_token(int(i)) for i in word]
        text = tokenizer.convert_tokens_to_string(tokens)
        final_words.append(text)
    return final_words


def attack(feature, tgt_model, mlm_model, tokenizer, k, batch_size, max_length=512, cos_mat=None, w2i={}, i2w={}, use_bpe=1, threshold_pred_score=0.3, word_budget=0.2, embs_distance=0.4):
    # MLM-process
    words_claim, words_evidence, sub_words_claim, sub_words_evidence, keys, start_evidence_idx = _tokenize(feature.claim, feature.evidence, tokenizer, max_length)

    # original label'
    input_ids, input_mask, segment_ids = prepare_tokenization(sub_words_claim, sub_words_evidence, max_length, tokenizer, padding=False)
    input_ids = torch.tensor(input_ids).cuda()  
    input_mask = torch.tensor(input_mask).cuda()
    segment_ids = torch.tensor(segment_ids).cuda() 
    
    seq_len = input_ids.size(0)
    orig_probs = tgt_model(input_ids.unsqueeze(0),
                           input_mask.unsqueeze(0),
                           segment_ids.unsqueeze(0)
                           ).squeeze()
    orig_probs = torch.softmax(orig_probs, -1)
    orig_label = torch.argmax(orig_probs)
    current_prob = orig_probs.max()
        
    if orig_label != feature.label:
        feature.success = 3
        return feature

    
    word_predictions = mlm_model(input_ids.unsqueeze(0),
                           input_mask.unsqueeze(0),
                           segment_ids.unsqueeze(0))[0].squeeze()  # seq-len(sub) vocab
    word_predictions = torch.softmax(word_predictions,-1)
    word_pred_scores_all, word_predictions = torch.topk(word_predictions, k, -1)  # seq-len k
    word_predictions = word_predictions[0:seq_len-1, :] #don't include the last sep token 
    word_pred_scores_all = word_pred_scores_all[0:seq_len-1, :]
    #print(word_predictions.size())

    word_predictions_evidence = word_predictions[start_evidence_idx:, :] #take predictions for evidence only 
    word_pred_scores_all_evidence = word_pred_scores_all[start_evidence_idx:,]
    #print('**')
    #print(sub_words_evidence[0])
    #print(word_predictions_evidence.size())
    #print(word_predictions_evidence[0])
    #top_k_words_test_tmp = [tokenizer._convert_id_to_token(int(word_predictions_evidence[0][i].data.cpu().numpy())) for i in range(k)]
    #print(top_k_words_test_tmp)
    #print(word_pred_scores_all_evidence[0])
    #print('**')    
    
    #print(word_predictions_evidence.size())
    #print(len(sub_words_evidence))
    #print('***')    
    #print(seq_len)
    important_scores = get_important_scores(sub_words_claim, words_evidence, tgt_model, current_prob, orig_label, orig_probs,tokenizer, batch_size, max_length)
    feature.query += int(len(words_evidence))
    list_of_index = sorted(enumerate(important_scores), key=lambda x: x[1], reverse=True)
    #print(list_of_index)
    #print('***')
    final_words = copy.deepcopy(words_evidence)

    for top_index in list_of_index:
        if feature.change > int(word_budget * (len(words_evidence))):
            feature.success = 1  # exceed
            return feature

        tgt_word = words_evidence[top_index[0]]
        if tgt_word in filter_words:
            continue
        #we don't need this check because we truncated the sequence already in tokenize.
        #if (keys[top_index[0]][0]+start_evidence_idx) > max_length - 2:
            #continue

        substitutes = word_predictions_evidence[keys[top_index[0]][0]: keys[top_index[0]][1]]#L,k
        word_pred_scores = word_pred_scores_all_evidence[keys[top_index[0]][0]: keys[top_index[0]][1]]

        substitutes = get_substitues(substitutes, tokenizer, mlm_model, use_bpe, word_pred_scores, threshold_pred_score)


        most_gap = 0.0
        candidate = None

        for substitute_ in substitutes:
            substitute = substitute_

            if substitute == tgt_word:
                continue  # filter out original word
            if '##' in substitute:
                continue  # filter out sub-word and words with weird characters.

            if substitute in filter_words:
                continue
            if substitute in w2i and tgt_word in w2i:
                #print(cos_mat[w2i[substitute]][w2i[tgt_word]])
                if cos_mat[w2i[substitute]][w2i[tgt_word]] < embs_distance:   
                    continue
            temp_replace = final_words
            temp_replace[top_index[0]] = substitute
            temp_text = tokenizer.convert_tokens_to_string(temp_replace)
            ### repeat tokenization
            temp_text_sub_words = tokenizer.tokenize(temp_text)
            input_ids_tmp, input_mask_tmp, segment_ids_tmp = prepare_tokenization(sub_words_claim, temp_text_sub_words, max_length, tokenizer, padding=False)
            input_ids_tmp = torch.tensor(input_ids_tmp).cuda()  
            input_mask_tmp = torch.tensor(input_mask_tmp).cuda()
            segment_ids_tmp = torch.tensor(segment_ids_tmp).cuda()
    
            #print(input_mask_tmp.size())
            seq_len = input_ids_tmp.size(0)
            temp_prob = tgt_model(input_ids_tmp.unsqueeze(0),
                           input_mask_tmp.unsqueeze(0),
                           segment_ids_tmp.unsqueeze(0)).squeeze()
            feature.query += 1
            temp_prob = torch.softmax(temp_prob, -1)                           
            temp_label = torch.argmax(temp_prob)

            if temp_label != orig_label:
                feature.change += 1
                final_words[top_index[0]] = substitute
                feature.changes.append([keys[top_index[0]][0], substitute, tgt_word])
                feature.final_adverse = temp_text
                feature.success = 4
                return feature
            else:

                label_prob = temp_prob[orig_label]
                gap = current_prob - label_prob
                if gap > most_gap:
                    most_gap = gap
                    candidate = substitute

        if most_gap > 0:
            feature.change += 1
            feature.changes.append([keys[top_index[0]][0], candidate, tgt_word])
            current_prob = current_prob - most_gap
            final_words[top_index[0]] = candidate

    feature.final_adverse = (tokenizer.convert_tokens_to_string(final_words))
    feature.success = 2
    return feature


def evaluate(features):
    do_use = 0
    use = None
    sim_thres = 0
    # evaluate with USE

    if do_use == 1:
        cache_path = ''
        import tensorflow as tf
        import tensorflow_hub as hub
    
        class USE(object):
            def __init__(self, cache_path):
                super(USE, self).__init__()

                self.embed = hub.Module(cache_path)
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                self.sess = tf.Session()
                self.build_graph()
                self.sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

            def build_graph(self):
                self.sts_input1 = tf.placeholder(tf.string, shape=(None))
                self.sts_input2 = tf.placeholder(tf.string, shape=(None))

                sts_encode1 = tf.nn.l2_normalize(self.embed(self.sts_input1), axis=1)
                sts_encode2 = tf.nn.l2_normalize(self.embed(self.sts_input2), axis=1)
                self.cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
                clip_cosine_similarities = tf.clip_by_value(self.cosine_similarities, -1.0, 1.0)
                self.sim_scores = 1.0 - tf.acos(clip_cosine_similarities)

            def semantic_sim(self, sents1, sents2):
                sents1 = [s.lower() for s in sents1]
                sents2 = [s.lower() for s in sents2]
                scores = self.sess.run(
                    [self.sim_scores],
                    feed_dict={
                        self.sts_input1: sents1,
                        self.sts_input2: sents2,
                    })
                return scores[0]

            use = USE(cache_path)


    acc = 0
    origin_success = 0
    total = 0
    total_q = 0
    total_change = 0
    total_word = 0
    for feat in features:
        if feat.success > 2:

            if do_use == 1:
                sim = float(use.semantic_sim([feat.seq], [feat.final_adverse]))
                if sim < sim_thres:
                    continue
            
            acc += 1
            total_q += feat.query
            total_change += feat.change
            total_word += len(feat.evidence.split(' '))

            if feat.success == 3:
                origin_success += 1

        total += 1

    print(origin_success)
    print(acc)
    suc = float(acc / total)

    query = float(total_q / acc)
    change_rate = float(total_change / total_word)

    origin_acc = 1 - origin_success / total
    after_atk = 1 - suc

    print('acc/aft-atk-acc {:.6f}/ {:.6f}, query-num {:.4f}, change-rate {:.4f}'.format(origin_acc, after_atk, query, change_rate))


def dump_features(features, output):
    outputs = []

    for feature in features:
        instance = {'label': feature.label,
                        'success': feature.success,
                        'change': feature.change,
                        'num_word': len(feature.evidence.split(' ')),
                        'query': feature.query,
                        'title': feature.title,
                        'evidence': feature.evidence,                        
                        'changes': feature.changes,
                        'claim': feature.claim,
                        'adv': feature.final_adverse,
                        'id_in_fever': feature.id_,
                        'id_in_file': feature.file_index
                        }

        with open(output, "a") as f:
            f.write(json.dumps(instance) + "\n")
    #print('finished dump')


def run_attack():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="./data/xxx")
    parser.add_argument("--mlm_path", type=str, help="xxx mlm")
    parser.add_argument("--tgt_path", type=str, help="xxx classifier")

    parser.add_argument("--output_dir", type=str, help="train file")
    parser.add_argument("--use_sim_mat", type=int, help='whether use cosine_similarity to filter out atonyms')
    parser.add_argument("--start", type=int, help="start step, for multi-thread process")
    parser.add_argument("--end", type=int, help="end step, for multi-thread process")
    parser.add_argument("--num_labels", type=int, )
    parser.add_argument("--use_bpe", type=int, )
    parser.add_argument("--k", type=int, )
    parser.add_argument("--threshold_pred_score", type=float, )
    parser.add_argument("--word_budget", type=float, )
    parser.add_argument("--embs_distance", type=float, default=0.4 )
    parser.add_argument("--bert_hidden_dim", default=768, type=int, help="Total batch size for training.")
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout.')




    args = parser.parse_args()
    data_path = str(args.data_path)
    mlm_path = str(args.mlm_path)
    tgt_path = str(args.tgt_path)
    output_dir = str(args.output_dir)
    num_labels = args.num_labels
    use_bpe = args.use_bpe
    k = args.k
    start = args.start
    threshold_pred_score = args.threshold_pred_score

    print('start process')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    config_atk = BertConfig.from_pretrained(mlm_path)
    mlm_model = BertForMaskedLM.from_pretrained(mlm_path, config=config_atk)
    mlm_model.to('cuda')
    mlm_model.eval()
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    bert_model = bert_model.cuda()
    tgt_model = inference_model(bert_model, args)
    tgt_model.load_state_dict(torch.load(args.tgt_path)['model'])
    tgt_model = tgt_model.cuda()
    tgt_model.eval()    
    features = read_file(data_path)
    if args.end != -1:
        end = args.end
    else:
        end = len(features)
    print('Start: '+str(start)+', end: '+str(end))   
    print('loading sim-embed')
    
    if args.use_sim_mat == 1:
        cos_mat, w2i, i2w = get_sim_embed('counter-fitted-vectors.txt','cos_sim_counter_fitting.npy')
    else:        
        cos_mat, w2i, i2w = None, {}, {}

    print('finish get-sim-embed')
    features_output = []

    with torch.no_grad():
        for index, feature in enumerate(features[start:end]):
            claim, title, evidence, label, id_, file_index = feature
            feat = Feature(claim, title, evidence, label, id_, file_index)
            print('\r number {:d} :'.format(index), end='')
            # print(feat.seq[:100], feat.label)
            feat = attack(feat, tgt_model, mlm_model, tokenizer, k, batch_size=32, max_length=180,
                          cos_mat=cos_mat, w2i=w2i, i2w=i2w, use_bpe=use_bpe,threshold_pred_score=threshold_pred_score, word_budget=args.word_budget, embs_distance=args.embs_distance)

            # print(feat.changes, feat.change, feat.query, feat.success)
            if feat.success > 2:
                if feat.success == 3: print('success - didn"t change')
                else: print('success')
            else:
                print('failed')
            features_output.append(feat)
            dump_features([feat], output_dir)

    evaluate(features_output)
    #dump_features(features_output, output_dir)


if __name__ == '__main__':
    run_attack()