import sys
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from recbole.model.sequential_recommender.sasrec import SASRec
from transformers import BertModel,BertConfig,PretrainedConfig

class CoWPiRec(SASRec):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.train_stage = config['train_stage']
        self.temperature = config['temperature']
        self.config = config

        self.item_embedding = None
        self.all_item_token_ids = torch.tensor(dataset.item_index2token_ids,dtype=int) # [n_item, max_len]
        self.all_item_atten_mask = torch.tensor(dataset.item_index2attention_mask,dtype=int) # [n_item, max_len]

        self.bert_config = PretrainedConfig.get_config_dict(config["model_name"])[0]
        self.bert_config["gradient_checkpointing"] = True
        self.bert_config = BertConfig.from_dict(self.bert_config)
        self.bert = BertModel.from_pretrained(config["model_name"],config=self.bert_config)
        if self.config["graph_agg"] == "graphsage":
            self.graph_att_q = torch.nn.Linear(self.bert_config.hidden_size, self.bert_config.hidden_size)
            self.graph_att_k = torch.nn.Linear(self.bert_config.hidden_size, self.bert_config.hidden_size)
            self.act = torch.nn.LeakyReLU()
            self.word_g_trans = torch.nn.Linear(self.bert_config.hidden_size*2, self.bert_config.hidden_size, bias=False)
    
    def qk_attention(self, query, key, value, valid=None):
        """
        :param query: ? * l * a
        :param key: ? * l * a
        :param value: ? * l * v
        :param valid: ? * l
        :return: ? * v
        """
        ele_valid = 1 if valid is None else valid.unsqueeze(dim=-1).float()  # ? * l * 1
        att_v = (query * key).sum(dim=-1, keepdim=True)  # ? * l * 1
        att_exp = (att_v - att_v.max(dim=-2, keepdim=True)[0]).exp() * ele_valid  # ? * l * 1
        att_sum = att_exp.sum(dim=-2, keepdim=True)  # ? * 1 * 1
        sum_valid = 1 if valid is None else ele_valid.sum(dim=-2, keepdim=True).gt(0).float()  # ? * 1 * 1
        att_w = att_exp / (att_sum * sum_valid + 1 - sum_valid)  # ? * l * 1
        result = (att_w * value).sum(dim=-2)  # ? * v
        return result
    
    def graphsage(self, interaction):
        input_ids=interaction["item_seq_token_ids"] # [bs,his_max,max_len]
        coclick_token_ids = interaction["item_seq_coclick_token_ids"] # [bs,his_max,max_len,10]
        self_emb = self.bert.embeddings.word_embeddings(input_ids) # [bs,his_max,max_len,768]
        neb_emb = self.bert.embeddings.word_embeddings(coclick_token_ids) # [bs,his_max,max_len,10,768]
        graph_att_q = self.act(self.graph_att_q(neb_emb).sum(dim=-2, keepdim=True)) # [bs,his_max,max_len,1,768]
        graph_att_k = self.act(self.graph_att_k(neb_emb)) # [bs,his_max,max_len,10,768]
        neb_emb = self.qk_attention(
            query=graph_att_q, key=graph_att_k, value=neb_emb)  # [bs,his_max,max_len,768]
        graph_output = self.act(self.word_g_trans(
                torch.cat([self_emb, neb_emb], dim=-1)))  # [bs,his_max,max_len,768*2] -> [bs,his_max,max_len,768]
        return graph_output
    
    def pretrain(self, interaction):
        item_seq = interaction[self.ITEM_SEQ] # [bs,his_max]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        graph_output = self.graphsage(interaction) # [bs,his_max,max_len,768]
        bert_output = self.bert(
            input_ids=interaction["item_seq_token_ids"].view(-1,self.config["max_len"]), # [bs,his_max,max_len]->[bs*his_max,max_len]
            attention_mask=interaction["item_seq_attention_mask"].view(-1,self.config["max_len"])
        )[0] # [bs*his_max,max_len,768]
        # WGP
        bert_output = bert_output.view(item_seq.shape[0],item_seq.shape[1],bert_output.shape[-2],bert_output.shape[-1]) # [bs,his_max,max_len,768]
        score = torch.matmul(bert_output, graph_output.transpose(-2, -1)) / self.temperature # [bs,his_max,max_len,max_len]
        loss_bert_graph = F.cross_entropy(score.view(-1,self.config["max_len"],self.config["max_len"]),interaction["item_seq_masked_labels"].view(-1,self.config["max_len"])) # [bs,his_max,max_len,max_len] : [bs,his_max,max_len]
        return loss_bert_graph
    
    def calculate_loss(self, interaction):
        if self.train_stage == 'pretrain':
            return self.pretrain(interaction)
