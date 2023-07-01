import argparse
import collections
import gzip
import html
import json
import os
import random
import re
import torch
from tqdm import tqdm
import scipy.sparse as sp
import numpy as np
from scipy.io import mmread, mmwrite, mminfo
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Food')
    parser.add_argument('--data_path', type=str, default="./dataset")
    return parser.parse_args()

def load_unit2index(file):
    unit2index = dict()
    with open(file, 'r') as fp:
        for line in fp:
            unit, index = line.strip().split('\t')
            unit2index[unit] = int(index)
    return unit2index
def load_inter(file):
    user_item_list = []
    with open(file, 'r') as fp:
        head_flag = True
        for line in fp:
            if head_flag:
                head_flag = False
                continue
            user_id,seq_item_ids,item_id=line.strip().split('\t')
            item_id_list = seq_item_ids.split(' ')
            user_item_list.append((int(user_id),int(item_id)))
            if len(item_id_list)==1:
                user_item_list.append((int(user_id),int(item_id_list[0])))
    return user_item_list
if __name__ == '__main__':
    args = parse_args()
    user2index = load_unit2index(os.path.join(args.data_path, args.dataset, f'{args.dataset}.user2index'))
    item2index = load_unit2index(os.path.join(args.data_path, args.dataset, f'{args.dataset}.item2index'))
    user_item_list = load_inter(os.path.join(args.data_path, args.dataset, f'{args.dataset}.train.inter'))
    R = sp.dok_matrix((len(user2index), len(item2index)), dtype=np.int32)
    inter_count = 0
    for user, item in tqdm(user_item_list,desc="get user-item graph"):
        R[user,item] += 1
        inter_count += 1
    print(f"user: {len(user2index)}, item: {len(item2index)}, inter: {inter_count}")
    mmwrite(os.path.join(args.data_path, args.dataset, f'{args.dataset}.user-item.mtx'),R)
    user_item = R.tolil().rows
    coclick_items_pairs = []
    for current_user in tqdm(range(user_item.shape[0]),desc="get coclick items"):
        items = user_item[current_user]
        for i in range(len(items)):
            for j in range(i+1,len(items)):
                coclick_items_pairs.append((items[i],items[j]))
    with open(os.path.join(args.data_path, args.dataset, f'{args.dataset}.coclick.items'),"w") as outf:
        for item1, item2 in tqdm(coclick_items_pairs,desc="write into file"):
            outf.write(f"{item1}\t{item2}\n")
