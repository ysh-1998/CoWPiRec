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
import numpy as np
from dataset.preprocessing.utils import check_path, set_device, load_plm, amazon_dataset2fullname

def load_text(file):
    item_text_list = []
    with open(file, 'r') as fp:
        fp.readline()
        for line in fp:
            try:
                item, text = line.strip().split('\t', 1)
            except ValueError:
                item = line.strip()
                text = '.'
            item_text_list.append([item, text])
    return item_text_list

def load_unit2index(file):
    unit2index = dict()
    with open(file, 'r') as fp:
        for line in fp:
            unit, index = line.strip().split('\t')
            unit2index[unit] = int(index)
    return unit2index

def whitening(vecs, new_dim=None):
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(1 / np.sqrt(s)))
    return W[:, :new_dim], mu

def generate_item_embedding(args, item_text_list, item2index, tokenizer, model, word_drop_ratio=-1):
    print(f'Generate Text Embedding by {args.emb_type}: ')
    print(' Dataset: ', args.dataset)

    items, texts = zip(*item_text_list)
    order_texts = [[0]] * len(items)
    for item, text in zip(items, texts):
        order_texts[item2index[item]] = text
    for text in order_texts:
        assert text != [0]

    embeddings = []
    start, batch_size = 0, 16
    pbar = tqdm(total=len(order_texts)//batch_size)
    while start < len(order_texts):
        sentences = order_texts[start: start + batch_size]
        if word_drop_ratio > 0:
            # print(f'Word drop with p={word_drop_ratio}')
            new_sentences = []
            for sent in sentences:
                new_sent = []
                sent = sent.split(' ')
                for wd in sent:
                    rd = random.random()
                    if rd > word_drop_ratio:
                        new_sent.append(wd)
                new_sent = ' '.join(new_sent)
                new_sentences.append(new_sent)
            sentences = new_sentences
        encoded_sentences = tokenizer(sentences, padding=True, max_length=512,
                                      truncation=True, return_tensors='pt').to(args.device)
        outputs = model(**encoded_sentences)
        if args.emb_type == 'CLS':
            cls_output = outputs.last_hidden_state[:, 0, ].detach().cpu()
            embeddings.append(cls_output)
        elif args.emb_type == 'Mean':
            masked_output = outputs.last_hidden_state * encoded_sentences['attention_mask'].unsqueeze(-1)
            mean_output = masked_output[:,1:,:].sum(dim=1) / \
                encoded_sentences['attention_mask'][:,1:].sum(dim=-1, keepdim=True)
            mean_output = mean_output.detach().cpu()
            embeddings.append(mean_output)
        start += batch_size
        pbar.update()
    pbar.close()
    embeddings = torch.cat(embeddings, dim=0).numpy()
    print('Embeddings shape: ', embeddings.shape)
    if args.emb_dim != 768:
        w, mu = whitening(embeddings,new_dim=args.emb_dim)
        embeddings = np.dot((embeddings - mu), w)
    file = os.path.join(args.dataset_path, args.dataset,
                        args.dataset + '.feat' + args.emb_type)
    embeddings.tofile(file)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_pretrain_model', action="store_true")
    parser.add_argument('--pretrain_model_path', type=str, help='pretrain model path')
    parser.add_argument('--save_model_path', type=str, default='./saved/model/CoWPiRec-base')
    parser.add_argument('--dataset_path', type=str, default='./dataset')
    parser.add_argument('--dataset', type=str, default='Scientific')
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of running GPU')
    parser.add_argument('--emb_type', type=str, default='CLS', help='item text emb type, can be CLS or Mean')
    parser.add_argument('--emb_dim', type=int, default=768)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print("dataset:",os.path.join(args.dataset_path,args.dataset))
    print("pretrain_model_path:",args.pretrain_model_path)
    print("save_model_path:",args.save_model_path)
    if args.load_pretrain_model:
        checkpoint = torch.load(args.pretrain_model_path,map_location=torch.device('cpu'))
        model = checkpoint["state_dict"]
        for key in list(model.keys()):
            if "bert" not in key:
                del model[key]
        save_model_path = f"{args.save_model_path}/pytorch_model.bin"
        torch.save(model, save_model_path)
    item_text_list = load_text(os.path.join(args.dataset_path, args.dataset, f'{args.dataset}.text'))
    item2index = load_unit2index(os.path.join(args.dataset_path, args.dataset, f'{args.dataset}.item2index'))
    print("load item_text item2index done")
    # device & plm initialization
    device = set_device(args.gpu_id)
    args.device = device
    plm_tokenizer, plm_model = load_plm(args.save_model_path)
    plm_model = plm_model.to(device)

    # create output dir
    check_path(os.path.join(args.dataset_path, args.dataset))

    # generate PLM emb and save to file
    generate_item_embedding(args, item_text_list, item2index, 
                            plm_tokenizer, plm_model, word_drop_ratio=-1)
