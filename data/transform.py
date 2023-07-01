from logging import getLogger
import random
import torch
from recbole.data.interaction import Interaction


def construct_transform(config):
    if config['transform'] is None:
        logger = getLogger()
        logger.warning('Equal transform')
        return Equal(config)
    else:
        str2transform = {
            'bert_input': BertInput
        }
        return str2transform[config['transform']](config)


class Equal:
    def __init__(self, config):
        pass

    def __call__(self, dataloader, interaction):
        return interaction

class BertInput:
    def __init__(self, config):
        self.logger = getLogger()
        self.logger.info(f'Bert Input Transform in DataLoader.')
        self.config = config
    def __call__(self, dataloader, interaction):
        # get bert input_ids, attention maks, position ids and masked lm labels
        item_seq = interaction['item_id_list'] # [bs,his_max]
        item_index2token_ids = dataloader.dataset.item_index2token_ids
        item_index2attention_mask = dataloader.dataset.item_index2attention_mask
        item_seq_token_ids = torch.tensor(item_index2token_ids[item_seq],dtype=int) # [bs,his_max,max_len]
        item_seq_attention_mask = torch.tensor(item_index2attention_mask[item_seq],dtype=int) # [bs,his_max,max_len]
        pos_item_id = interaction['item_id'] # [bs,]
        pos_item_token_ids = torch.tensor(item_index2token_ids[pos_item_id],dtype=int) # [bs,max_len]
        pos_item_attention_mask = torch.tensor(item_index2attention_mask[pos_item_id],dtype=int) # [bs,max_len]
        item_index2coclick_token_ids = dataloader.dataset.item_index2coclick_token_ids
        item_index2masked_labels = dataloader.dataset.item_index2masked_labels
        # sample neighbor
        item_seq_coclick_token_ids = torch.tensor(item_index2coclick_token_ids[item_seq],dtype=int) # [bs,his_max,max_len,30]
        sample_shape = list(item_seq_coclick_token_ids.shape)
        sample_shape[-1] = self.config["sample_neighbors"]
        sample_index = torch.randint(0,item_seq_coclick_token_ids.shape[-1],sample_shape)
        item_seq_coclick_token_ids = torch.gather(item_seq_coclick_token_ids,-1,sample_index) # [bs,his_max,max_len,10]
        item_seq_masked_labels = torch.tensor(item_index2masked_labels[item_seq],dtype=int) # [bs,his_max,max_len]

        interaction.update(Interaction({
            'item_seq_token_ids': item_seq_token_ids,
            'item_seq_coclick_token_ids': item_seq_coclick_token_ids,
            'item_seq_attention_mask': item_seq_attention_mask,
            'item_seq_masked_labels': item_seq_masked_labels,
            'pos_item_token_ids': pos_item_token_ids,
            'pos_item_attention_mask': pos_item_attention_mask,
        }))
        
        return interaction