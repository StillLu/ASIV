
import argparse
import csv
import logging
import os
import random
import sys
import pickle
import pandas as pd

import matplotlib.pyplot as plt


import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm, trange

from transformers import BertConfig, BertForSequenceClassification, BertModel, BertTokenizer, AutoTokenizer


import itertools
from util import *


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def main():

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default="../Datasets/SST/data",
                        type=str,
                        help="The input data dir.")

    parser.add_argument("--output_dir",
                        default="testacc_sst",
                        type=str,
                        help="The output directory where the model predictions.")

    parser.add_argument("--trained_model_dir",
                        default="trained_bert_sst",
                        type=str,
                        help="Where is the fine-tuned BERT model?")

    parser.add_argument("--max_seq_length",
                        default=50,
                        type=int,
                        help="Sequences longer than this will be truncated, and sequences shorter than this will be padded.")

    parser.add_argument("--batch_size",
                        default=32,
                        type=int,
                        help="mini batch size")

    parser.add_argument("--num_train_epochs",
                        default=10,
                        type=int,
                        help="total number of training epochs to perform.")

    parser.add_argument("--num_labels",
                        default=2,
                        type=int,
                        help="the size of label set")


    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    args = parser.parse_args()

    seed_everything(args.seed)

    if os.path.exists(args.trained_model_dir) and os.listdir(args.trained_model_dir):
        logger.info(
            "WARNING: Trained model directory ({}) already exists and is not empty.".format(args.trained_model_dir))
    if not os.path.exists(args.trained_model_dir):
        os.makedirs(args.trained_model_dir)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        logger.info("WARNING: Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Prepare dataloader
    '''
    train = pd.read_pickle(os.path.join(args.data_dir, 'train.pkl'))
    dev = pd.read_pickle(os.path.join(args.data_dir, 'dev.pkl'))
    test = pd.read_pickle(os.path.join(args.data_dir, 'test.pkl'))
    '''

    train = pd.read_csv(os.path.join(args.data_dir, "train.tsv"), sep='\t', header=None, names=['similarity', 's1'])
    dev = pd.read_csv(os.path.join(args.data_dir, "dev.tsv"), sep='\t', header=None, names=['similarity', 's1'])
    test = pd.read_csv(os.path.join(args.data_dir, "test.tsv"), sep='\t', header=None, names=['similarity', 's1'])

    train['s1'] = train['s1'].str[:-1]
    dev['s1'] = dev['s1'].str[:-1]
    test['s1'] = test['s1'].str[:-1]

    ### remove the instance that contains single word
    train = train[train['s1'].str.split().str.len() > 1]
    dev = dev[dev['s1'].str.split().str.len() > 1]
    test = test[test['s1'].str.split().str.len() > 1]

    logger.info("  train size = %d", len(train))
    logger.info("  dev size = %d", len(dev))
    logger.info("  test size = %d", len(test))

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    train_data = DataPrecessForBert(tokenizer, train, max_seq_len=args.max_seq_length)
    train_data_loader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size)

    test_data = DataPrecessForBert(tokenizer, test, max_seq_len=args.max_seq_length)
    test_data_loader = DataLoader(test_data, shuffle=False, batch_size=args.batch_size)

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', args.num_labels)

    model.to(device)

    optimizer = ret_optimizer(model)
    ## linear
    scheduler = ret_scheduler(optimizer, int(len(train_data) / args.batch_size) * args.num_train_epochs)

    model.train()
    for epoch in range(args.num_train_epochs):
        train_loss, train_accuracy = 0, 0
        for bi_index, (input_ids, attention_mask, token_type_ids, labels) in enumerate(train_data_loader):
            ids = input_ids.to(device, dtype=torch.long)
            mask = attention_mask.to(device, dtype=torch.long)
            targets = labels.to(device, dtype=torch.long)
            token_ids = token_type_ids.to(device, dtype=torch.long)

            optimizer.zero_grad()

            loss, logits = model(input_ids=ids, attention_mask=mask, token_type_ids=token_ids, labels=targets)[:2]

            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            # scheduler.step()

            train_loss += loss.item() * len(ids)
            train_correct = acc_comp(logits, targets)

            train_accuracy += train_correct.detach().cpu().numpy()

            # scheduler.step(train_accuracy / len(train))

        logger.info("  avg train loss = %f", train_loss / len(train))
        logger.info("  avg train accuracy = %f", train_accuracy / len(train))

    torch.save(model.state_dict(), os.path.join(args.model_dir, 'bert_model.pth'))

    model.eval()
    test_loss, test_accuracy = 0, 0

    for bi_index, (input_ids, attention_mask, token_type_ids, labels) in enumerate(test_data_loader):
        ids = input_ids.to(device, dtype=torch.long)
        mask = attention_mask.to(device, dtype=torch.long)
        targets = labels.to(device, dtype=torch.long)
        token_ids = token_type_ids.to(device, dtype=torch.long)

        with torch.no_grad():
            loss, logits = model(input_ids=ids, attention_mask=mask, token_type_ids=token_ids, labels=targets)[:2]
            test_correct = acc_comp(logits, targets)
            test_loss += loss.item() * len(ids)
            test_accuracy += test_correct.detach().cpu().numpy()

    logger.info("  avg test loss = %f", test_loss / len(test))
    logger.info("  avg test accuracy = %f", test_accuracy / len(test))


if __name__ == "__main__":
    main()




