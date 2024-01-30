import os
import argparse
from logging import getLogger
from recbole.config import Config
# from recbole.trainer.trainer import PretrainTrainer
from trainer import PretrainTrainer
from recbole.utils import init_seed, init_logger

from CoWPiRec import CoWPiRec
from data.dataset import PretrainCoWPiRecDataset
from data.dataloader import CustomizedTrainDataLoader


def pretrain(dataset, **kwargs):
    # configurations initialization
    props = [f'props/CoWPiRec.yaml', f'props/pretrain.yaml']
    print(props)

    # configurations initialization
    config = Config(model=CoWPiRec, dataset=dataset, config_file_list=props, config_dict=kwargs)
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # pretrain data
    pretrain_dataset = PretrainCoWPiRecDataset(config)
    logger.info(pretrain_dataset)
    pretrain_dataset = pretrain_dataset.build()[0]
    pretrain_data = CustomizedTrainDataLoader(config, pretrain_dataset, None, shuffle=True)
    model_data = pretrain_data
    
    # model loading and initialization
    model = CoWPiRec(config, model_data.dataset).to(config['device'])
    # logger.info(model)

    # trainer loading and initialization
    trainer = PretrainTrainer(config, model)
    trainer._build_schedule(pretrain_data)
    # model pre-training
    trainer.pretrain(pretrain_data, show_progress=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='FHCKM', help='dataset name')
    args, unparsed = parser.parse_known_args()
    print(args)

    pretrain(args.d)
