import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from time import time
from recbole.data.dataset import SequentialDataset
from utils import load_tokenize_json,load_unit2index
from transformers import BertTokenizer
from transformers.data.data_collator import DataCollatorForLanguageModeling

class PretrainCoWPiRecDataset(SequentialDataset):
    def __init__(self, config):
        print("load dataset ...")
        start_time = time()
        super().__init__(config)
        end_time = time()
        print(f"load dataset done in {round(end_time-start_time, 2)} seconds")
        self.tokenizer = BertTokenizer.from_pretrained(self.config["model_name"])
        self.datacollecter = DataCollatorForLanguageModeling(self.tokenizer)
        word_graph_path = osp.join(self.config['data_path'],  f'{self.dataset_name}.word_graph.jsonl')
        self.coclick_word_dict = load_tokenize_json(word_graph_path)
        self.item_index2token_ids, \
        self.item_index2coclick_token_ids, \
        self.item_index2attention_mask, \
        self.item_index2masked_labels = self.get_inputs()
        
    def get_inputs(self):
        multi_data = True if osp.exists(osp.join(self.config['data_path'], f'{self.dataset_name}.pt_datasets')) else False
        if multi_data:
            with open(osp.join(self.config['data_path'], f'{self.dataset_name}.pt_datasets'), 'r') as file:
                dataset_names = file.read().strip().split(',')
            self.logger.info(f'Pre-training datasets: {dataset_names}')
            ds2token_dict = []
            index2item_list = []
            for dataset_name in dataset_names:
                token_dict_path = osp.join("/".join(self.config['data_path'].split('/')[:-1]), dataset_name, f'{dataset_name}.tokenize.json')
                item2index_path = osp.join("/".join(self.config['data_path'].split('/')[:-1]), dataset_name, f'{dataset_name}.item2index')
                token_dict = load_tokenize_json(token_dict_path)
                item2index = load_unit2index(item2index_path)
                index2item = {value:key for key,value in item2index.items()}
                ds2token_dict.append(token_dict)
                index2item_list.append(index2item)
        else:
            token_dict_path = osp.join(self.config['data_path'], f'{self.dataset_name}.tokenize.json')
            item2index_path = osp.join(self.config['data_path'], f'{self.dataset_name}.item2index')
            token_dict = load_tokenize_json(token_dict_path)
            item2index = load_unit2index(item2index_path)
            index2item = {value:key for key,value in item2index.items()}
        item_index2token_ids = np.zeros((self.item_num, self.config["max_len"]))
        item_index2coclick_token_ids = np.zeros((self.item_num, self.config["max_len"],self.config["num_neighbors"]))
        item_index2attention_mask = np.zeros((self.item_num, self.config["max_len"]))
        item_index2masked_labels = np.zeros((self.item_num, self.config["max_len"]))
        cls_id, sep_id, pad_id = 101, 102, 0
        for i, token in enumerate(self.field2id_token['item_id']):
            if token == '[PAD]': continue
            if multi_data:
                did, iid = token.split('-')
                token_dict = ds2token_dict[int(did)]
                index2item = index2item_list[int(did)]
                token = iid
            token_ids = token_dict[index2item[int(token)]][:self.config["max_len"]-2]
            token_ids = [cls_id] + token_ids + [pad_id]*(self.config["max_len"]-2-len(token_ids)) + [sep_id]
            coclick_token_ids = []
            for word in token_ids:
                if self.coclick_word_dict.get(word) is not None:
                    coclick_words = self.coclick_word_dict[word]
                    if len(coclick_words) < self.config["num_neighbors"]:
                        coclick_token_ids.append([word]*self.config["num_neighbors"])
                    else:
                        coclick_token_ids.append(coclick_words[:self.config["num_neighbors"]])
                else:
                    coclick_token_ids.append([word]*self.config["num_neighbors"])
            dc = self.datacollecter([token_ids])
            masked_lm_labels = dc["labels"][0].numpy().tolist() # [-100, -100, -100, -100, -100, -100, 2128, -100, ...]
            for j in range(len(masked_lm_labels)):
                if masked_lm_labels[j] != -100:
                    masked_lm_labels[j] = j # [-100, -100, -100, -100, -100, -100, 6, -100, ...]
            token_ids = dc["input_ids"][0].numpy().tolist()
            attention_mask = np.int64(np.array(token_ids) > 0).tolist()
            item_index2token_ids[i] = token_ids # [max_len]
            item_index2coclick_token_ids[i] = coclick_token_ids # [max_len,10]
            item_index2attention_mask[i] = attention_mask # [max_len]
            item_index2masked_labels[i] = masked_lm_labels
        return item_index2token_ids,item_index2coclick_token_ids, item_index2attention_mask,item_index2masked_labels
