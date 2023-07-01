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
            'plm_emb': PLMEmb
        }
        return str2transform[config['transform']](config)


class Equal:
    def __init__(self, config):
        pass

    def __call__(self, dataloader, interaction):
        return interaction


class PLMEmb:
    def __init__(self, config):
        self.logger = getLogger()
        self.logger.info('PLM Embedding Transform in DataLoader.')

    def __call__(self, dataloader, interaction):

        item_seq = interaction['item_id_list']
        plm_embedding = dataloader.dataset.plm_embedding
        item_emb_seq = plm_embedding(item_seq)
        pos_item_id = interaction['item_id']
        pos_item_emb = plm_embedding(pos_item_id)


        interaction.update(Interaction({
            'item_emb_list': item_emb_seq,
            'pos_item_emb': pos_item_emb
        }))

        return interaction
