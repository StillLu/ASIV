

import argparse
import csv

import os
import random
import sys
import pickle
import pandas as pd

import time
import copy
import logging

import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset

from transformers import BertModel, BertTokenizer, BertForSequenceClassification, AutoTokenizer, BertForMaskedLM, \
    BertConfig

from transformers import RobertaTokenizer, RobertaModel, RobertaForSequenceClassification, AutoTokenizer

from transformers import RobertaForMaskedLM, RobertaTokenizer, LineByLineTextDataset

import itertools
from itertools import chain, combinations
import random
import transformers

from util import *
from asiv import *

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




data_dir = "Datasets/SST/data"
output_dir = "results/cs"
model_dir = "trained_bert_sst"

##### pretrain bert from SST2 dataset
pretrain_model_dir = "pretrain/SST/BertSST2"

tokenizer_max_len = 50
batch_size = 32
num_train_epochs = 10
max_grad_norm = 10.0

train = pd.read_csv(os.path.join(data_dir, "train.tsv"), sep='\t', header=None, names=['similarity', 's1'])
dev = pd.read_csv(os.path.join(data_dir, "dev.tsv"), sep='\t', header=None, names=['similarity', 's1'])
test = pd.read_csv(os.path.join(data_dir, "test.tsv"), sep='\t', header=None, names=['similarity', 's1'])



train['s1'] = train['s1'].str[:-1]
dev['s1'] = dev['s1'].str[:-1]
test['s1'] = test['s1'].str[:-1]

train = train[train['s1'].str.split().str.len() > 1]
dev = dev[dev['s1'].str.split().str.len() > 1]
test = test[test['s1'].str.split().str.len() > 1]

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

train_data = DataPrecessForBert(tokenizer, train, max_seq_len=tokenizer_max_len)
train_data_loader = DataLoader(train_data, shuffle=True, batch_size=1)

test_data = DataPrecessForBert(tokenizer, test, max_seq_len=tokenizer_max_len)
test_data_loader = DataLoader(test_data, shuffle=False, batch_size=1)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.to(device)
model.load_state_dict(torch.load(os.path.join(data_dir, 'bert_model.pth')))
model.eval()

#### load our pretrain LM
pretrain_model = BertForMaskedLM.from_pretrained(pretrain_model_dir)
pretrain_model.to(device)
pretrain_model.eval()
pretrain_tokenizer = BertTokenizer.from_pretrained(pretrain_model_dir)

### load bert
pretrain_model_1 = BertForMaskedLM.from_pretrained('bert-base-uncased')
pretrain_model_1.to(device)
pretrain_model_1.eval()
pretrin_tokenizer_1 = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

### bert
cls_token = 101
sep_token = 102
cls = '[CLS]'
sep = '[SEP]'
pad_s = '[PAD]'
knn = 2
pad_t = 0
mask_m = '[MASK]'

for bi_index_t, (input_ids, attention_masks, token_type_ids, labels) in enumerate(test_data_loader):

    start = time.time()

    ########
    ids = input_ids.to(device, dtype=torch.long)
    mask = attention_masks.to(device, dtype=torch.long)
    targets = labels.to(device, dtype=torch.long)
    token_ids = token_type_ids.to(device, dtype=torch.long)

    with torch.no_grad():
        loss, logits = model(input_ids=ids, attention_mask=mask, token_type_ids=token_ids, labels=targets)[:2]

        prob_output = nn.Softmax(dim=1)(logits)
        y_pred = torch.log_softmax(logits, dim=1).argmax(dim=1)

    input_ids_ay = input_ids.numpy()[0]
    attention_masks = attention_masks.numpy()[0]

    sat_id_ori = np.where(input_ids_ay == cls_token)[0][0]
    end_id_ori = np.where(input_ids_ay == sep_token)[0][0]

    # feature_set = np.delete(input_ids_ay, all_com)

    all_w = []
    for bi_index, (input_ids_bs, attention_mask, token_type_ids, labels) in enumerate(train_data_loader):
        input_ids_bs = input_ids_bs.numpy()[0]

        sat_id = np.where(input_ids_bs == cls_token)[0][0]
        end_id = np.where(input_ids_bs == sep_token)[0][0]

        if end_id >= end_id_ori:
            all_w.append(input_ids_bs)

    sat_idn = np.where(input_ids_ay == cls_token)[0][0]
    end_idn = np.where(input_ids_ay == sep_token)[0][0]

    all_subset = list(combinations([i for i in range(1, end_idn)], 2))

    n_token = end_idn - 1
    A1 = np.zeros((n_token, n_token))
    A2 = np.zeros((n_token, n_token))
    A3 = np.zeros((n_token, n_token))
    A4 = np.zeros((n_token, n_token))

    for i in range(1, end_idn):
        for j in range(1, end_idn):

            if i != j:
                head = [i]
                tail = [j]

                all_com = head + tail
                all_com.sort()

                a1, a2 = compute_subset_shapley_con_full_1(cls_token, sep_token, all_com, head, tail, all_w,
                                                           input_ids_ay, mask, token_ids,
                                                           targets, end_id_ori, y_pred, model, pretrain_model,
                                                           tokenizer, pad_s, pad_t, mask_m, device)

                a3 = compute_subset_shapley_con_full_2(cls_token, sep_token, all_com, head, tail, all_w, input_ids_ay,
                                                       mask, token_ids,
                                                       targets, end_id_ori, y_pred, model, pretrain_model,
                                                       tokenizer, pad_s, pad_t, mask_m, device)

                a4 = compute_subset_shapley_con_full_2(cls_token, sep_token, all_com, head, tail, all_w, input_ids_ay,
                                                       mask, token_ids,
                                                       targets, end_id_ori, y_pred, model, pretrain_model_1,
                                                       tokenizer, pad_s, pad_t, mask_m, device)

                A1[head[0] - 1][tail[0] - 1] = a1
                A2[head[0] - 1][tail[0] - 1] = a2
                A3[head[0] - 1][tail[0] - 1] = a3
                A4[head[0] - 1][tail[0] - 1] = a4


    logger.info("index=%s", str(bi_index_t))

    list_nodes, list_scores = Phi_PageRank(A1.T, shapley_values=None, dmp=0.85)
    sort_index = np.argsort(np.array(list_scores)).tolist()[0]
    aa1 = [i + 1 for i in sort_index]

    logger.info("Complex example: %s", [i + 1 for i in sort_index])

    list_nodes, list_scores = Phi_PageRank(A2.T, shapley_values=None, dmp=0.85)
    sort_index = np.argsort(np.array(list_scores)).tolist()[0]
    aa2 = [i + 1 for i in sort_index]

    logger.info("Complex example: %s", [i + 1 for i in sort_index])

    pickle.dump(aa1, open(os.path.join(output_dir, "cs11" + str(bi_index_t) + ".pkl"), "wb"))
    pickle.dump(aa2, open(os.path.join(output_dir, "cs22" + str(bi_index_t) + ".pkl"), "wb"))

    list_nodes, list_scores = Phi_PageRank(A3.T, shapley_values=None, dmp=0.85)
    sort_index = np.argsort(np.array(list_scores)).tolist()[0]
    aa3 = [i + 1 for i in sort_index]

    logger.info("Complex example: %s", [i + 1 for i in sort_index])

    pickle.dump(aa3, open(os.path.join(output_dir, "cs33" + str(bi_index_t) + ".pkl"), "wb"))

    list_nodes, list_scores = Phi_PageRank(A4.T, shapley_values=None, dmp=0.85)
    sort_index = np.argsort(np.array(list_scores)).tolist()[0]
    aa4 = [i + 1 for i in sort_index]

    logger.info("Complex example: %s", [i + 1 for i in sort_index])

    pickle.dump(aa4, open(os.path.join(output_dir, "cs77" + str(bi_index_t) + ".pkl"), "wb"))




































