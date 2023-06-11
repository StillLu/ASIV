
import argparse
import csv

import os
import random
import sys
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import copy

import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset,TensorDataset

import itertools
from itertools import chain, combinations
from itertools import islice


sampling_instance = 1000 ### the number of permutations for computing value funtion
pretrain_k = 1  ### in-domain sampling
maximum_length = 50



def compute_pretrain_s(cls_token, sep_token, new_x_sorted, tokenizer, pad_s, pretrain_model, model, device, targets,
                       token_ids, y_pred,all_npi_s,mask,pad_t,mask_m):

    new_input_ids = [[cls_token] + x_sorted + [sep_token] for x_sorted in new_x_sorted]
    text = [tokenizer.convert_ids_to_tokens(input_ids) for input_ids in new_input_ids]
    # all_npi = [[np_i] * pretrain_k for np_i in all_npi]

    new_text_all = [[mask_m if x == pad_s else x for x in sub_text] for sub_text in text]

    all_tokens_tensor = []
    for new_text in new_text_all:
        indexed_tokens = tokenizer.convert_tokens_to_ids(new_text)
        all_tokens_tensor.append(torch.from_numpy(np.array(indexed_tokens)))

    all_input_t = TensorDataset(torch.stack(all_tokens_tensor))
    data_loader = DataLoader(all_input_t, shuffle=False, batch_size=100)

    prob_all = []
    for bi_index, input_ids in enumerate(data_loader):
        tokens_tensor = input_ids[0].to('cuda')

        with torch.no_grad():
            predictions = pretrain_model(tokens_tensor)[0]
            prob = torch.nn.Softmax(dim=1)(predictions)

            prob_all.extend(prob)

    all_input = []
    all_mask = []
    all_npi = []
    for ii in range(len(new_input_ids)):

        if pad_s not in text[ii]:
            new_text = new_text_all[ii]
            indexed_tokens = tokenizer.convert_tokens_to_ids(new_text)

            indexed_tokens = indexed_tokens + [pad_t] * (maximum_length - len(indexed_tokens))
            tokens_tensor = torch.from_numpy(np.array(indexed_tokens))

            all_input.append(tokens_tensor)
            all_mask.append(mask.cpu()[0])
            all_npi.append(all_npi_s[ii])

        else:

            prob = prob_all[ii]
            new_text = new_text_all[ii]
            results = []
            all_weight = []

            for i, t in enumerate(new_text):
                if t == mask_m:
                    sub_re = []
                    top_k_weights, top_k_indices = torch.topk(prob[i], pretrain_k, sorted=True)
                    # predicted_index = torch.argmax(predictions[0, i]).item()
                    a = []
                    for j in range(len(top_k_indices)):
                        predicted_token = tokenizer.convert_ids_to_tokens([top_k_indices[j]])[0]
                        sub_re.append(predicted_token)
                        a.append(top_k_weights[j].item())
                        # print(top_k_weights[j])
                    results.append(sub_re)
                    all_weight.append(a)

            all_weights = [sum(i) / new_text.count(mask_m) for i in zip(*all_weight)]

            ori_id = []
            for k in range(len(new_text)):
                if new_text[k] == mask_m:
                    ori_id.append(k)

            for i in range(pretrain_k):

                new_seq = new_text
                # new_att = [0] * len(new_seq)
                #new_att = [1 if x == 0 else 1 for x in new_input_ids[ii]]

                for k in ori_id:
                    new_seq[k] = results[ori_id.index(k)][i]
                #  new_att[k] = all_weight[ori_id.index(k)][i]

                indexed_tokens = tokenizer.convert_tokens_to_ids(new_seq)

                indexed_tokens = indexed_tokens + [pad_t] * (maximum_length - len(indexed_tokens))

                #new_att = new_att + [pad_t] * (maximum_length - len(new_att))

                tokens_tensor = torch.from_numpy(np.array(indexed_tokens))
                # tokens_tensor = tokens_tensor.to(device, dtype=torch.long)

                #mask = torch.from_numpy(np.array(new_att))
                # mask = mask.to(device, dtype=torch.long)

                all_mask.append(mask.cpu()[0])
                all_input.append(tokens_tensor)

            all_npi.append(np_i_new for np_i_new in [all_npi_s[ii]] * pretrain_k)

    return all_input, all_mask, all_npi





def generate_full_s(input_ids_bs, end_id_ori, head, tail, n_features, O_n, x_, loc_fid_head, loc_fid_tail):
    w_feature_set = input_ids_bs[:end_id_ori]

    new_w_set = np.delete(w_feature_set, head + tail)
    new_w_set = new_w_set[1:]
    new_w_set = np.insert(new_w_set, len(new_w_set), 60000)
    new_w_set = np.insert(new_w_set, len(new_w_set), 60001)

    w_ = np.zeros((n_features))
    w_[:] = new_w_set[O_n[:]]

    b_1 = np.zeros((n_features))
    b_1[:] = np.concatenate((x_[0:loc_fid_head + 1], w_[loc_fid_head + 1:]))  # v(Si)

    return w_feature_set, new_w_set, w_, b_1


def generate_sort_head(w_feature_set, head, tail, b_1, x_p, input_ids_ay, input_ids_bs, a1, new_l, O_n1, loc_fid_head,
                       loc_fid_tail):

    xx_p = [input_ids_ay[i] for i in tail]
    w_p = [input_ids_bs[i] for i in new_l]

    com_p = x_p + w_p

    b3 = np.where(b_1 == 60001)[0][0]
    b4 = np.insert(b_1, b3, com_p)
    b31 = np.where(b4 == 60000)[0][0]
    b44 = np.insert(b4, b31, xx_p)

    indices = np.where(b44 == 60000)
    b44 = np.delete(b44, indices)
    indices = np.where(b44 == 60001)
    b44 = np.delete(b44, indices)
    b5 = b44

    h = copy.deepcopy(O_n1)
    h[loc_fid_head:loc_fid_head] = a1 + new_l
    index = h.index(maximum_length+55)
    h[index:index] = tail
    h.remove(maximum_length+56)
    h.remove(maximum_length+55)

    u = zip(list(b5), h)
    x_sorted = [wd for wd, idx in sorted(u, key=lambda x: x[1])]

    return x_sorted

def generate_sort_head_p(w_feature_set, head, tail, b_1, x_p, input_ids_ay, input_ids_bs, a1, new_l, O_n1, loc_fid_head,
                       loc_fid_tail):

    #xx_p = [input_ids_ay[i] for i in head]
    w_p = [input_ids_bs[i] for i in new_l]
    ww_p = [input_ids_bs[i] for i in tail]

    com_p = x_p + w_p

    b3 = np.where(b_1 == 60000)[0][0]
    b4 = np.insert(b_1, b3, com_p)
    b31 = np.where(b4 == 60001)[0][0]
    b44 = np.insert(b4, b31, ww_p)

    indices = np.where(b44 == 60000)
    b44 = np.delete(b44, indices)
    indices = np.where(b44 == 60001)
    b44 = np.delete(b44, indices)
    b5 = b44

    h = copy.deepcopy(O_n1)
    h[loc_fid_head:loc_fid_head] = a1 + new_l
    index = h.index(maximum_length+55)
    h[index:index] = tail
    h.remove(maximum_length+56)
    h.remove(maximum_length+55)

    u = zip(list(b5), h)
    x_sorted = [wd for wd, idx in sorted(u, key=lambda x: x[1])]

    return x_sorted


def generate_sort_tail(w_feature_set, head, tail, b_1, x_p, input_ids_ay, input_ids_bs, a1, new_l, O_n1, loc_fid_head,
                       loc_fid_tail):
    xx_p = [input_ids_ay[i] for i in head]
    w_p = [input_ids_bs[i] for i in new_l]
    ww_p = [input_ids_bs[i] for i in head]

    com_p = x_p + w_p

    b3 = np.where(b_1 == 60000)[0][0]
    b4 = np.insert(b_1, b3, com_p)
    b31 = np.where(b4 == 60001)[0][0]
    b44 = np.insert(b4, b31, ww_p)

    indices = np.where(b44 == 60000)
    b44 = np.delete(b44, indices)
    indices = np.where(b44 == 60001)
    b44 = np.delete(b44, indices)
    b5 = b44

    h = copy.deepcopy(O_n1)
    h[loc_fid_head:loc_fid_head] = a1 + new_l
    index = h.index(maximum_length+55)
    h[index:index] = tail
    h.remove(maximum_length+56)
    h.remove(maximum_length+55)

    u = zip(list(b5), h)
    x_sorted = [wd for wd, idx in sorted(u, key=lambda x: x[1])]

    return x_sorted


####### random sampling: masking + replace the words from training corpus

def compute_subset_shapley_con_full_1(cls_token, sep_token, all_com, head, tail, all_w, input_ids_ay, mask, token_ids,
                                    targets, end_id_ori, y_pred, model, pretrain_model, tokenizer, pad_s,pad_t,mask_m, device):

    all_head_com = list(chain.from_iterable(combinations(head, r) for r in range(len(head) + 1)))
    all_tail_com = list(chain.from_iterable(combinations(tail, r) for r in range(len(tail) + 1)))

    all_head_input_s = []
    all_head_npi_s = []
    all_tail_input_s = []
    all_tail_npi_s = []
    all_head_mask_s = []
    all_tail_mask_s = []

    all_head_input_d = []
    all_head_npi_d = []
    all_head_mask_d = []
    all_tail_input_d = []
    all_tail_npi_d = []
    all_tail_mask_d = []

    all_head_x_sorted = []

    all_tail_x_sorted = []

    feature_set = np.delete(input_ids_ay, all_com)

    sat_idn = np.where(feature_set == cls_token)[0][0]
    end_idn = np.where(feature_set == sep_token)[0][0]

    new_feature_set = feature_set[1:end_idn]

    #### original index vs new index
    all_b = {}
    for i in range(len(new_feature_set)):

        a = new_feature_set[i]

        b = np.where(input_ids_ay == a)[0]

        if len(b) == 1:
            all_b[i] = b[0]

        else:

            while True:
                bj = random.sample(list(b), 1)[0]
                if bj not in all_b.values() and bj not in all_com:
                    break
            all_b[i] = bj

    new_feature_set = np.insert(new_feature_set, len(new_feature_set), 60000)
    all_b[len(new_feature_set) - 1] = maximum_length+55

    new_feature_set = np.insert(new_feature_set, len(new_feature_set), 60001)
    all_b[len(new_feature_set) - 1] = maximum_length+56

    ##### feature_index
    fid_head = len(new_feature_set) - 1
    fid_tail = len(new_feature_set) - 2

    ##### perturbation
    n_features = len(new_feature_set)
    new_mask = mask.cpu()[0]
    for input_ids_bs in all_w[:sampling_instance]:

        O_n = np.zeros(n_features, dtype='int')
        x_ = np.zeros(n_features)

        O_n[:] = np.random.permutation(n_features)
        loc_fid_head = np.where(O_n[:] == fid_head)[0][0]
        loc_fid_tail = np.where(O_n[:] == fid_tail)[0][0]

        O_n1 = [all_b[O_n[i]] for i in range(len(O_n))]

        #####
        x_ = np.zeros(n_features)
        x_[:] = new_feature_set[O_n[:]]

        w_feature_set_s, new_w_set_s, w_s, b_1_s = generate_full_s(input_ids_bs, end_id_ori, head, tail, n_features,
                                                                   O_n, x_, loc_fid_head, loc_fid_tail)

        w_feature_set_d, new_w_set_d, w_d, b_1_d = generate_full_s(np.zeros(maximum_length), end_id_ori, head, tail, n_features,
                                                                   O_n, x_, loc_fid_head, loc_fid_tail)


        for i in range(len(all_head_com)):

            np_i = (-1) ** (len(head) - len(list(all_head_com[i])))

            a1 = list(all_head_com[i])
            new_l = [i for i in head if i not in a1]

            x_p = [input_ids_ay[i] for i in a1]

            x_sorted_s = generate_sort_head(w_feature_set_s, head, tail, b_1_s, x_p, input_ids_ay, input_ids_bs, a1,
                                            new_l, O_n1, loc_fid_head, loc_fid_tail)

            x_sorted_d = generate_sort_head(w_feature_set_d, head, tail, b_1_d, x_p, input_ids_ay, np.zeros(maximum_length), a1,
                                            new_l,
                                            O_n1, loc_fid_head, loc_fid_tail)

            new_input_ids_s = [cls_token] + x_sorted_s + [sep_token] + [pad_t] * (maximum_length - end_id_ori - 1)
            input_ids_s_s = torch.from_numpy(np.array(new_input_ids_s))

            new_input_ids_d = [cls_token] + x_sorted_d + [sep_token] + [pad_t] * (maximum_length - end_id_ori - 1)
            input_ids_s_d = torch.from_numpy(np.array(new_input_ids_d))



            new_input = [cls_token] + x_sorted_d + [sep_token]
            new_att = [0 if x == 0 else 1 for x in new_input]
            new_att = new_att + [0] * (maximum_length - end_id_ori - 1)
            mask_d = torch.from_numpy(np.array(new_att))

            all_head_input_s.append(input_ids_s_s)
            all_head_npi_s.append(np_i)
            all_head_mask_s.append(new_mask)

            all_head_input_d.append(input_ids_s_d)
            all_head_npi_d.append(np_i)
            all_head_mask_d.append(mask_d)

            all_head_x_sorted.append(x_sorted_d)




        for i in range(len(all_head_com)):

            np_i = (-1) ** (len(head) - len(list(all_head_com[i])))
            a1 = list(all_head_com[i])
            new_l = [i for i in head if i not in a1]

            x_p = [input_ids_ay[i] for i in a1]

            x_sorted_s = generate_sort_head_p(w_feature_set_s, head, tail, b_1_s, x_p, input_ids_ay, input_ids_bs, a1,
                                            new_l, O_n1, loc_fid_head, loc_fid_tail)

            x_sorted_d = generate_sort_head_p(w_feature_set_d, head, tail, b_1_d, x_p, input_ids_ay, np.zeros(maximum_length), a1,
                                            new_l,
                                            O_n1, loc_fid_head, loc_fid_tail)

            new_input_ids_s = [cls_token] + x_sorted_s + [sep_token] + [pad_t] * (maximum_length - end_id_ori - 1)
            input_ids_s_s = torch.from_numpy(np.array(new_input_ids_s))

            new_input_ids_d = [cls_token] + x_sorted_d + [sep_token] + [pad_t] * (maximum_length - end_id_ori - 1)
            input_ids_s_d = torch.from_numpy(np.array(new_input_ids_d))

            new_input = [cls_token] + x_sorted_d + [sep_token]
            new_att = [0 if x == 0 else 1 for x in new_input]
            new_att = new_att + [0] * (maximum_length - end_id_ori - 1)
            mask_d = torch.from_numpy(np.array(new_att))

            all_tail_input_s.append(input_ids_s_s)
            all_tail_npi_s.append(np_i)
            all_tail_mask_s.append(new_mask)

            all_tail_input_d.append(input_ids_s_d)
            all_tail_npi_d.append(np_i)
            all_tail_mask_d.append(mask_d)

            all_tail_x_sorted.append(x_sorted_d)

    #all_head_input_p, all_head_mask_p, all_head_npi_p = compute_pretrain_s(cls_token, sep_token, all_head_x_sorted, tokenizer, pad_s, pretrain_model, model, device, targets,
                       #token_ids, y_pred,all_head_npi_s,mask)

    #all_tail_input_p, all_tail_mask_p, all_tail_npi_p = compute_pretrain_s(cls_token, sep_token, all_tail_x_sorted, tokenizer, pad_s, pretrain_model, model, device, targets,
                       #token_ids, y_pred,all_tail_npi_s,mask)






    slice_indexes = [sampling_instance * len(all_head_com), sampling_instance * len(all_tail_com)]

    all_input_s = all_head_input_s + all_tail_input_s
    all_npi_s = all_head_npi_s + all_tail_npi_s
    all_mask_s = all_head_mask_s + all_tail_mask_s

    all_score_s = value_func_l(all_input_s, all_npi_s, all_mask_s, token_ids, targets, y_pred, model, device)
    all_condi_s = compute_score(slice_indexes, all_score_s)

    all_input_d = all_head_input_d + all_tail_input_d
    all_npi_d = all_head_npi_d + all_tail_npi_d
    all_mask_d = all_head_mask_d + all_tail_mask_d

    all_score_d = value_func_l(all_input_d, all_npi_d, all_mask_d, token_ids, targets, y_pred, model, device)
    all_condi_d = compute_score(slice_indexes, all_score_d)


    return all_condi_s / (sampling_instance), all_condi_d / (sampling_instance)


###### conditional sampling + in-domain sampling

def compute_subset_shapley_con_full_2(cls_token, sep_token, all_com, head, tail, all_w, input_ids_ay, mask, token_ids,
                                    targets, end_id_ori, y_pred, model, pretrain_model, tokenizer, pad_s, pad_t,mask_m,device):

    all_head_com = list(chain.from_iterable(combinations(head, r) for r in range(len(head) + 1)))
    all_tail_com = list(chain.from_iterable(combinations(tail, r) for r in range(len(tail) + 1)))

    all_head_input_s = []
    all_head_npi_s = []
    all_tail_input_s = []
    all_tail_npi_s = []
    all_head_mask_s = []
    all_tail_mask_s = []

    all_head_input_d = []
    all_head_npi_d = []
    all_head_mask_d = []
    all_tail_input_d = []
    all_tail_npi_d = []
    all_tail_mask_d = []

    all_head_x_sorted = []

    all_tail_x_sorted = []

    feature_set = np.delete(input_ids_ay, all_com)

    sat_idn = np.where(feature_set == cls_token)[0][0]
    end_idn = np.where(feature_set == sep_token)[0][0]

    new_feature_set = feature_set[1:end_idn]

    #### original index vs new index
    all_b = {}
    for i in range(len(new_feature_set)):

        a = new_feature_set[i]

        b = np.where(input_ids_ay == a)[0]

        if len(b) == 1:
            all_b[i] = b[0]

        else:

            while True:
                bj = random.sample(list(b), 1)[0]
                if bj not in all_b.values() and bj not in all_com:
                    break
            all_b[i] = bj

    new_feature_set = np.insert(new_feature_set, len(new_feature_set), 60000)
    all_b[len(new_feature_set) - 1] = maximum_length+55

    new_feature_set = np.insert(new_feature_set, len(new_feature_set), 60001)
    all_b[len(new_feature_set) - 1] = maximum_length+56

    ##### feature_index
    fid_head = len(new_feature_set) - 1
    fid_tail = len(new_feature_set) - 2

    ##### perturbation
    n_features = len(new_feature_set)
    new_mask = mask.cpu()[0]
    for input_ids_bs in all_w[:sampling_instance]:

        O_n = np.zeros(n_features, dtype='int')
        x_ = np.zeros(n_features)

        O_n[:] = np.random.permutation(n_features)
        loc_fid_head = np.where(O_n[:] == fid_head)[0][0]
        loc_fid_tail = np.where(O_n[:] == fid_tail)[0][0]

        O_n1 = [all_b[O_n[i]] for i in range(len(O_n))]

        #####
        x_ = np.zeros(n_features)
        x_[:] = new_feature_set[O_n[:]]

        w_feature_set_s, new_w_set_s, w_s, b_1_s = generate_full_s(input_ids_bs, end_id_ori, head, tail, n_features,
                                                                   O_n, x_, loc_fid_head, loc_fid_tail)

        w_feature_set_d, new_w_set_d, w_d, b_1_d = generate_full_s(np.zeros(maximum_length), end_id_ori, head, tail, n_features,
                                                                   O_n, x_, loc_fid_head, loc_fid_tail)


        for i in range(len(all_head_com)):

            np_i = (-1) ** (len(head) - len(list(all_head_com[i])))

            a1 = list(all_head_com[i])
            new_l = [i for i in head if i not in a1]

            x_p = [input_ids_ay[i] for i in a1]

            x_sorted_s = generate_sort_head(w_feature_set_s, head, tail, b_1_s, x_p, input_ids_ay, input_ids_bs, a1,
                                            new_l, O_n1, loc_fid_head, loc_fid_tail)

            x_sorted_d = generate_sort_head(w_feature_set_d, head, tail, b_1_d, x_p, input_ids_ay, np.zeros(maximum_length), a1,
                                            new_l,
                                            O_n1, loc_fid_head, loc_fid_tail)

            new_input_ids_s = [cls_token] + x_sorted_s + [sep_token] + [pad_t] * (maximum_length - end_id_ori - 1)
            input_ids_s_s = torch.from_numpy(np.array(new_input_ids_s))

            new_input_ids_d = [cls_token] + x_sorted_d + [sep_token] + [pad_t] * (maximum_length - end_id_ori - 1)
            input_ids_s_d = torch.from_numpy(np.array(new_input_ids_d))



            new_input = [cls_token] + x_sorted_d + [sep_token]
            new_att = [0 if x == 0 else 1 for x in new_input]
            new_att = new_att + [0] * (maximum_length - end_id_ori - 1)
            mask_d = torch.from_numpy(np.array(new_att))


            all_head_npi_s.append(np_i)

            all_head_x_sorted.append(x_sorted_d)




        for i in range(len(all_head_com)):

            np_i = (-1) ** (len(head) - len(list(all_head_com[i])))
            a1 = list(all_head_com[i])
            new_l = [i for i in head if i not in a1]

            x_p = [input_ids_ay[i] for i in a1]

            x_sorted_s = generate_sort_head_p(w_feature_set_s, head, tail, b_1_s, x_p, input_ids_ay, input_ids_bs, a1,
                                            new_l, O_n1, loc_fid_head, loc_fid_tail)

            x_sorted_d = generate_sort_head_p(w_feature_set_d, head, tail, b_1_d, x_p, input_ids_ay, np.zeros(maximum_length), a1,
                                            new_l,
                                            O_n1, loc_fid_head, loc_fid_tail)

            new_input_ids_s = [cls_token] + x_sorted_s + [sep_token] + [pad_t] * (maximum_length - end_id_ori - 1)
            input_ids_s_s = torch.from_numpy(np.array(new_input_ids_s))

            new_input_ids_d = [cls_token] + x_sorted_d + [sep_token] + [pad_t] * (maximum_length - end_id_ori - 1)
            input_ids_s_d = torch.from_numpy(np.array(new_input_ids_d))

            new_input = [cls_token] + x_sorted_d + [sep_token]
            new_att = [0 if x == 0 else 1 for x in new_input]
            new_att = new_att + [0] * (maximum_length - end_id_ori - 1)
            mask_d = torch.from_numpy(np.array(new_att))


            all_tail_npi_s.append(np_i)

            all_tail_x_sorted.append(x_sorted_d)

    all_head_input_p, all_head_mask_p, all_head_npi_p = compute_pretrain_s(cls_token, sep_token, all_head_x_sorted, tokenizer, pad_s, pretrain_model, model, device, targets,
                       token_ids, y_pred,all_head_npi_s,mask,pad_t,mask_m)

    all_tail_input_p, all_tail_mask_p, all_tail_npi_p = compute_pretrain_s(cls_token, sep_token, all_tail_x_sorted, tokenizer, pad_s, pretrain_model, model, device, targets,
                       token_ids, y_pred,all_tail_npi_s,mask,pad_t,mask_m)






    slice_indexes = [sampling_instance * len(all_head_com), sampling_instance * len(all_tail_com)]



    all_input_p = all_head_input_p + all_tail_input_p
    all_npi_p = all_head_npi_p + all_tail_npi_p
    all_mask_p = all_head_mask_p + all_tail_mask_p

    all_score_p = value_func_l(all_input_p, all_npi_p, all_mask_p, token_ids, targets, y_pred, model, device)
    all_condi_p = compute_score(slice_indexes, all_score_p)


    return all_condi_p / (sampling_instance)




def compute_score(slice_indexes, score_list):
    it = iter(score_list)
    sliced = [sum(list(islice(it, 0, i))) for i in slice_indexes]
    score = sliced[0] - sliced[1]
    return score



def value_func(input_data,input_npi, mask, token_ids,targets,y_pred, model,device):

    mask_it = mask.cpu().repeat(len(input_npi), 1)
    token_ids_it = token_ids.cpu().repeat(len(input_npi), 1)
    targets_it = targets.cpu().repeat(len(input_npi), 1)

    all_input_t = TensorDataset(torch.stack(input_data), mask_it,token_ids_it,targets_it)
    eval_data_loader = DataLoader(all_input_t, shuffle=False, batch_size=500)

    all_pre = []
    for bi_index, (input_ids,mask,token_ids,targets) in enumerate(eval_data_loader):

        input_ids = input_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.float)
        token_ids = token_ids.to(device, dtype=torch.long)
        targets = targets.to(device,dtype=torch.long)

        with torch.no_grad():
            loss, logits = model(input_ids=input_ids, attention_mask=mask, token_type_ids=token_ids, labels=targets)[:2]
            a1 = nn.Softmax(dim=1)(logits)[:, y_pred].cpu().detach().numpy().tolist()
            # print(a1)
            a2 = [item for sublist in a1 for item in sublist]

            all_pre.append(a2)
    all_pre = [item for sublist in all_pre for item in sublist]
    all_score = sum([a*b for a,b in zip(all_pre,input_npi)])
    return all_score


def value_func_p(input_data,input_npi, input_mask, token_ids,targets,y_pred, model,device):

    token_ids_it = token_ids.cpu().repeat(len(input_npi), 1)
    targets_it = targets.cpu().repeat(len(input_npi), 1)

    all_input_t = TensorDataset(torch.stack(input_data),torch.stack(input_mask),token_ids_it, targets_it)
    eval_data_loader = DataLoader(all_input_t, shuffle=False, batch_size=500)

    all_pre = []
    for bi_index, (input_ids,mask,token_ids,targets) in enumerate(eval_data_loader):
        input_ids = input_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.float64)
        token_ids = token_ids.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.long)
        with torch.no_grad():

            loss, logits = model(input_ids=input_ids, attention_mask=mask, token_type_ids=token_ids, labels=targets)[:2]
            a1 = nn.Softmax(dim=1)(logits)[:,y_pred].cpu().detach().numpy().tolist()

            a2 = [item for sublist in a1 for item in sublist]

            pre = nn.Softmax(dim=1)(logits)[0][y_pred]

            all_pre.append(a2)
    all_pre = [item for sublist in all_pre for item in sublist]
    all_score = sum([a * b for a, b in zip(all_pre, input_npi)])
    return all_score



def value_func_l(input_data,input_npi, input_mask, token_ids,targets,y_pred, model,device):

    token_ids_it = token_ids.cpu().repeat(len(input_npi), 1)
    targets_it = targets.cpu().repeat(len(input_npi), 1)

    all_input_t = TensorDataset(torch.stack(input_data),torch.stack(input_mask),token_ids_it, targets_it)
    eval_data_loader = DataLoader(all_input_t, shuffle=False, batch_size=500)

    all_pre = []
    for bi_index, (input_ids,mask,token_ids,targets) in enumerate(eval_data_loader):
        input_ids = input_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.float64)
        token_ids = token_ids.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.long)
        with torch.no_grad():

            loss, logits = model(input_ids=input_ids, attention_mask=mask, token_type_ids=token_ids, labels=targets)[:2]
            a1 = nn.Softmax(dim=1)(logits)[:,y_pred].cpu().detach().numpy().tolist()

            a2 = [item for sublist in a1 for item in sublist]

            pre = nn.Softmax(dim=1)(logits)[0][y_pred]

            all_pre.append(a2)
    all_pre = [item for sublist in all_pre for item in sublist]
    all_score = [a * b for a, b in zip(all_pre, input_npi)]
    return all_score



def Phi_PageRank(phi_plus, shapley_values = None, dmp = 0.85):
    '''
    PageRank on Phi_Plus Matrix
    '''

    pagerank = PageRank(damping_factor=dmp)
    G = nx.from_numpy_matrix(phi_plus , create_using=nx.DiGraph)

    for i in range(phi_plus.shape[0]):
        for j in range(phi_plus.shape[0]):
            if phi_plus[i,j] != 0:
                G.add_weighted_edges_from([(i, j, phi_plus[i,j])])

    xx = list(nx.weakly_connected_components(G))

    S = [G.subgraph(c).copy() for c in nx.weakly_connected_components(G)]

    list_nodes = []
    list_scores = []

    for i in range(len(S)):
        adjacency = nx.adjacency_matrix(S[i])
        if shapley_values is None:
            scores = pagerank.fit_transform(adjacency)
        else:
            scores = pagerank.fit_transform(adjacency, seeds = shapley_values)


        list_nodes.append(np.array(list(xx[i])))
        list_scores.append(scores)

    return list_nodes, list_scores

