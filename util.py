
#### all required library

import argparse
import csv
import os
import random
import sys
import pickle
import time
import math

import numpy as np
import torch
from torch import nn

from torch.utils.data import DataLoader, Dataset

from transformers import AdamW, BertConfig, BertForSequenceClassification, BertModel, BertTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import torch.autograd as autograd
from scipy import stats


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


##### classifier


class BertClassifier(nn.Module):
    def __init__(self, bert_model, num_labels, dropout_rate):
        super(BertClassifier, self).__init__()
        self.num_labels = num_labels

        self.bert = bert_model.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask):
        pooled_output = self.bert(input_ids, attention_mask=attention_mask)[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        nll = nn.CrossEntropyLoss()
        #nll = LabelSmoothingCrossEntropy()
        #loss = nll(logits, targets)

        return logits, pooled_output




class XLBertClassifier(nn.Module):
    def __init__(self, bert_model, num_labels, dropout_rate):
        super(XLBertClassifier, self).__init__()
        self.num_labels = num_labels

        self.bert = bert_model.from_pretrained("xlnet-base-cased")
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask):
        pooled_output = self.bert(input_ids, attention_mask=attention_mask)[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        nll = nn.CrossEntropyLoss()
        #nll = LabelSmoothingCrossEntropy()
        #loss = nll(logits, targets)

        return logits, pooled_output


class ROBertClassifier(nn.Module):
    def __init__(self, bert_model, num_labels, dropout_rate):
        super(ROBertClassifier, self).__init__()
        self.num_labels = num_labels

        self.bert = bert_model.from_pretrained("roberta-base")
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask):
        pooled_output = self.bert(input_ids, attention_mask=attention_mask)[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        nll = nn.CrossEntropyLoss()
        #nll = LabelSmoothingCrossEntropy()
        #loss = nll(logits, targets)

        return logits, pooled_output



####  prepare  input


#### bert [CLS] 101 [SEP] 102 [PAD] 0
#### roberta <s> 0 </s> 2  <pad> 1
#### xlnet <cls> 3 <sep> 4 <pad> 5



class TCDataset:
    def __init__(self, text, label, tokenizer, max_len):
        self.texts = text
        self.labels = label

        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]
        idx = index

        inputs = self.tokenizer.__call__(text, None, add_special_tokens=True, max_length=self.max_len,
                                         padding="max_length", truncation=True)
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        return {"ids": torch.tensor(ids, dtype=torch.long),
                "mask": torch.tensor(mask, dtype=torch.long),
                "labels": torch.tensor(label, dtype=torch.long),
                "index": torch.tensor(idx, dtype=torch.long)}



class DataPrecessForBert(Dataset):
    """
    Encoding sentences
    """

    def __init__(self, bert_tokenizer, df, max_seq_len=50):
        super(DataPrecessForBert, self).__init__()
        self.bert_tokenizer = bert_tokenizer
        self.max_seq_len = max_seq_len
        self.input_ids, self.attention_mask, self.token_type_ids, self.labels = self.get_input(df)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx], self.token_type_ids[idx], self.labels[idx]

    # Convert dataframe to tensor
    def get_input(self, df):
        sentences = df['s1'].values
        labels = df['similarity'].values

        # tokenizer
        tokens_seq = list(map(self.bert_tokenizer.tokenize, sentences))  # list of shape [sentence_len, token_len]

        # Get fixed-length sequence and its mask
        result = list(map(self.trunate_and_pad, tokens_seq))

        input_ids = [i[0] for i in result]
        attention_mask = [i[1] for i in result]
        token_type_ids = [i[2] for i in result]

        return (
            torch.Tensor(input_ids).type(torch.long),
            torch.Tensor(attention_mask).type(torch.long),
            torch.Tensor(token_type_ids).type(torch.long),
            torch.Tensor(labels).type(torch.long)
        )

    def trunate_and_pad(self, tokens_seq):
        # Concat '[CLS]' at the beginning
        tokens_seq = ['[CLS]'] + tokens_seq + ['[SEP]']
        # Truncate sequences of which the lengths exceed the max_seq_len
        if len(tokens_seq) > self.max_seq_len:
            tokens_seq = tokens_seq[0: self.max_seq_len-1] + ['[SEP]']
            # Generate padding
        padding = [0] * (self.max_seq_len - len(tokens_seq))
        # Convert tokens_seq to token_ids
        input_ids = self.bert_tokenizer.convert_tokens_to_ids(tokens_seq)
        input_ids += padding
        # Create attention_mask
        attention_mask = [1] * len(tokens_seq) + padding
        # Create token_type_ids
        token_type_ids = [0] * (self.max_seq_len)

        assert len(input_ids) == self.max_seq_len
        assert len(attention_mask) == self.max_seq_len
        assert len(token_type_ids) == self.max_seq_len

        return input_ids, attention_mask, token_type_ids



class DataPrecessForXlnet(Dataset):
    """
    Encoding sentences
    """

    def __init__(self, bert_tokenizer, df, max_seq_len=50):
        super(DataPrecessForXlnet, self).__init__()
        self.bert_tokenizer = bert_tokenizer
        self.max_seq_len = max_seq_len
        self.input_ids, self.attention_mask, self.token_type_ids, self.labels = self.get_input(df)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx], self.token_type_ids[idx], self.labels[idx]

    # Convert dataframe to tensor
    def get_input(self, df):
        sentences = df['s1'].values
        labels = df['similarity'].values

        # tokenizer
        tokens_seq = list(map(self.bert_tokenizer.tokenize, sentences))  # list of shape [sentence_len, token_len]

        # Get fixed-length sequence and its mask
        result = list(map(self.trunate_and_pad, tokens_seq))

        input_ids = [i[0] for i in result]
        attention_mask = [i[1] for i in result]
        token_type_ids = [i[2] for i in result]

        return (
            torch.Tensor(input_ids).type(torch.long),
            torch.Tensor(attention_mask).type(torch.long),
            torch.Tensor(token_type_ids).type(torch.long),
            torch.Tensor(labels).type(torch.long)
        )

    def trunate_and_pad(self, tokens_seq):
        # Concat '[CLS]' at the beginning
        tokens_seq = ['<cls>'] + tokens_seq + ['<sep>']
        # Truncate sequences of which the lengths exceed the max_seq_len
        if len(tokens_seq) > self.max_seq_len:
            tokens_seq = tokens_seq[0: self.max_seq_len-1] + ['<sep>']
            # Generate padding
        padding = [0] * (self.max_seq_len - len(tokens_seq))
        # Convert tokens_seq to token_ids
        input_ids = self.bert_tokenizer.convert_tokens_to_ids(tokens_seq)
        input_ids += [5] * (self.max_seq_len - len(tokens_seq))
        # Create attention_mask
        attention_mask = [1] * len(tokens_seq) + padding
        # Create token_type_ids
        token_type_ids = [0] * (self.max_seq_len)

        assert len(input_ids) == self.max_seq_len
        assert len(attention_mask) == self.max_seq_len
        assert len(token_type_ids) == self.max_seq_len

        return input_ids, attention_mask, token_type_ids


class DataPrecessForRoberta(Dataset):
    """
    Encoding sentences
    """

    def __init__(self, bert_tokenizer, df, max_seq_len=50):
        super(DataPrecessForRoberta, self).__init__()
        self.bert_tokenizer = bert_tokenizer
        self.max_seq_len = max_seq_len
        self.input_ids, self.attention_mask, self.token_type_ids, self.labels = self.get_input(df)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx], self.token_type_ids[idx], self.labels[idx]

    # Convert dataframe to tensor
    def get_input(self, df):
        sentences = df['s1'].values
        labels = df['similarity'].values

        # tokenizer
        tokens_seq = list(map(self.bert_tokenizer.tokenize, sentences))  # list of shape [sentence_len, token_len]

        # Get fixed-length sequence and its mask
        result = list(map(self.trunate_and_pad, tokens_seq))

        input_ids = [i[0] for i in result]
        attention_mask = [i[1] for i in result]
        token_type_ids = [i[2] for i in result]

        return (
            torch.Tensor(input_ids).type(torch.long),
            torch.Tensor(attention_mask).type(torch.long),
            torch.Tensor(token_type_ids).type(torch.long),
            torch.Tensor(labels).type(torch.long)
        )

    def trunate_and_pad(self, tokens_seq):
        # Concat '[CLS]' at the beginning
        tokens_seq = ['<s>'] + tokens_seq + ['</s>']
        # Truncate sequences of which the lengths exceed the max_seq_len
        if len(tokens_seq) > self.max_seq_len:
            tokens_seq = tokens_seq[0: self.max_seq_len-1] + ['</s>']
            # Generate padding
        padding = [0] * (self.max_seq_len - len(tokens_seq))
        # Convert tokens_seq to token_ids
        input_ids = self.bert_tokenizer.convert_tokens_to_ids(tokens_seq)
        input_ids += [1] * (self.max_seq_len - len(tokens_seq))
        # Create attention_mask
        attention_mask = [1] * len(tokens_seq) + padding
        # Create token_type_ids
        token_type_ids = [0] * (self.max_seq_len)

        assert len(input_ids) == self.max_seq_len
        assert len(attention_mask) == self.max_seq_len
        assert len(token_type_ids) == self.max_seq_len

        return input_ids, attention_mask, token_type_ids



##### create dataset and dataloader
def build_dataset(train, dev, test, tokenizer_max_len):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = TCDataset(train.sentence.tolist(), train.label.values, tokenizer, tokenizer_max_len)
    dev_dataset = TCDataset(dev.sentence.tolist(), dev.label.values, tokenizer, tokenizer_max_len)
    test_dataset = TCDataset(test.sentence.tolist(), test.label.values, tokenizer, tokenizer_max_len)

    return train_dataset, dev_dataset, test_dataset


def build_dataloader(train_dataset, dev_dataset, test_dataset, Shuffle, batch_size):
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=Shuffle, num_workers=1)
    dev_data_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    return train_data_loader, dev_data_loader, test_data_loader


def build_dataset_ch(data, tokenizer, tokenizer_max_len):
    ce_dataset = TCDataset(data.sentence.tolist(), data.label.values, tokenizer, tokenizer_max_len)
    return ce_dataset

def build_dataloader_ch(dataset, Shuffle, batch_size):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=Shuffle, num_workers=1)
    return data_loader



##### optimizer


def ret_optimizer(model):

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01
        },
        {
            'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)
    return optimizer



def ret_scheduler(optimizer, num_train_steps):
    sch = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)
    return sch


def ret_scheduler_1(optimizer, Q):
    lrs = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = Q)
    return lrs


##### evaluation

def acc_comp(y_pred, y_test):
    acc = (torch.argmax(y_pred, dim=1) == y_test).sum().float()
    #  acc = (torch.argmax(y_pred,dim=1)==y_test).sum().float() / float(y_test.size(0))

    return acc


def trainer(model, train_data_loader, device, optimizer, scheduler):

    model.train()
    train_loss, train_accuracy = 0, 0
    for bi_train, d in enumerate(train_data_loader):
        ids = d["ids"]
        mask = d["mask"]
        targets = d["labels"]

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.long)

        optimizer.zero_grad()

        output, pooled_output = model(ids, mask)
        loss = nn.CrossEntropyLoss()(output, targets)
        # if torch.cuda.device_count() > 1:
        #  loss = loss.mean()

        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()*len(ids)
        train_correct = acc_comp(output, targets)

        train_accuracy += train_correct.detach().cpu().numpy()

    return train_loss, train_accuracy


def evaluate(model, dev_data_loader, device):

    model.eval()
    eval_loss, eval_accuracy = 0, 0

    for bi_train, d in enumerate(dev_data_loader):
        ids = d["ids"]
        mask = d["mask"]
        targets = d["labels"]

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.long)

        with torch.no_grad():
            output, pooled_output = model(ids, mask)
            loss = nn.CrossEntropyLoss()(output, targets)
            eval_correct = acc_comp(output, targets)
            eval_loss += loss.item()*len(ids)
            eval_accuracy += eval_correct.detach().cpu().numpy()

    return eval_loss, eval_accuracy
