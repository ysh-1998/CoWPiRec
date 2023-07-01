import os
import argparse
from logging import getLogger
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from recbole.utils import init_seed, init_logger

from config import Config
from CoWPiRec import CoWPiRec
from data.dataset import PretrainCoWPiRecDataset,CoWPiRecDataset
from data.dataloader import CustomizedTrainDataLoader,CustomizedFullSortEvalDataLoader
from ddp_trainer import DDPPretrainTrainer


def pretrain(rank, world_size, dataset, **kwargs):
    # configurations initialization
    props = [f'props/CoWPiRec.yaml', f'props/pretrain.yaml']
    if rank == 0:
        print('DDP Pre-training on:', dataset)
        print(props)

    # configurations initialization
    kwargs.update({'ddp': True, 'rank': rank, 'world_size': world_size})
    config = Config(model=CoWPiRec, dataset=dataset, config_file_list=props, config_dict=kwargs)
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    if config['rank'] not in [-1, 0]:
        config['state'] = 'warning'
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
    model = CoWPiRec(config, model_data.dataset)
    # logger.info(model)

    # trainer loading and initialization
    trainer = DDPPretrainTrainer(config, model)
    trainer._build_schedule(pretrain_data)
    # model pre-training
    trainer.pretrain(pretrain_data, show_progress=(rank == 0))

    dist.destroy_process_group()

    return config['model'], config['dataset']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='FHCKM', help='dataset name')
    parser.add_argument('-p', type=str, default='12355', help='port for ddp')
    args, unparsed = parser.parse_known_args()

    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}."
    world_size = n_gpus

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.p

    mp.spawn(pretrain,
             args=(world_size, args.d),
             nprocs=world_size,
             join=True)
