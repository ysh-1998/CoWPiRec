from recbole.utils import ModelType, ensure_dir, get_local_time, set_color
import json
from recbole.data.dataloader import *
from data.dataloader import CustomizedTrainDataLoader,CustomizedFullSortEvalDataLoader
import importlib
import os
import pickle
from recbole.utils.argument_list import dataset_arguments
from recbole.sampler import KGSampler, Sampler, RepeatableSampler
def load_tokenize_json(json_dir):
    id2token_dict = {}
    with open(json_dir) as f:
        for line in f:
            es = json.loads(line)
            id, ids = es["id"], es["ids"]
            if id not in id2token_dict:
                id2token_dict[id] = ids
    return id2token_dict
def load_unit2index(file):
    unit2index = dict()
    with open(file, 'r') as fp:
        for line in fp:
            unit, index = line.strip().split('\t')
            unit2index[unit] = int(index)
    return unit2index
def _get_AE_dataloader(config, phase):
    """Customized function for VAE models to get correct dataloader class.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        phase (str): The stage of dataloader. It can only take two values: 'train' or 'evaluation'.

    Returns:
        type: The dataloader class that meets the requirements in :attr:`config` and :attr:`phase`.
    """
    if phase == 'train':
        return UserDataLoader
    else:
        eval_strategy = config['eval_neg_sample_args']['strategy']
        if eval_strategy in {'none', 'by'}:
            return NegSampleEvalDataLoader
        elif eval_strategy == 'full':
            return FullSortEvalDataLoader
def get_dataloader(config, phase):
    """Return a dataloader class according to :attr:`config` and :attr:`phase`.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        phase (str): The stage of dataloader. It can only take two values: 'train' or 'evaluation'.

    Returns:
        type: The dataloader class that meets the requirements in :attr:`config` and :attr:`phase`.
    """
    register_table = {
        "MultiDAE": _get_AE_dataloader,
        "MultiVAE": _get_AE_dataloader,
        'MacridVAE': _get_AE_dataloader,
        'CDAE': _get_AE_dataloader,
        'ENMF': _get_AE_dataloader,
        'RaCT': _get_AE_dataloader,
        'RecVAE': _get_AE_dataloader,
    }

    if config['model'] in register_table:
        return register_table[config['model']](config, phase)

    model_type = config['MODEL_TYPE']
    if phase == 'train':
        if model_type != ModelType.KNOWLEDGE:
            return CustomizedTrainDataLoader
        else:
            return KnowledgeBasedDataLoader
    else:
        eval_strategy = config['eval_neg_sample_args']['strategy']
        if eval_strategy in {'none', 'by'}:
            return NegSampleEvalDataLoader
        elif eval_strategy == 'full':
            return CustomizedFullSortEvalDataLoader

def load_split_dataloaders(config):
    """Load split dataloaders if saved dataloaders exist and
    their :attr:`config` of dataset are the same as current :attr:`config` of dataset.

    Args:
        config (Config): An instance object of Config, used to record parameter information.

    Returns:
        dataloaders (tuple of AbstractDataLoader or None): The split dataloaders.
    """

    default_file = os.path.join(config['checkpoint_dir'], f'{config["dataset"]}-for-{config["model"]}-dataloader.pth')
    dataloaders_save_path = config['dataloaders_save_path'] or default_file
    if not os.path.exists(dataloaders_save_path):
        return None
    with open(dataloaders_save_path, 'rb') as f:
        train_data, valid_data, test_data = pickle.load(f)
    for arg in dataset_arguments + ['seed', 'repeatable', 'eval_args']:
        if config[arg] != train_data.config[arg]:
            return None
    train_data.update_config(config)
    valid_data.update_config(config)
    test_data.update_config(config)
    logger = getLogger()
    logger.info(set_color('Load split dataloaders from', 'pink') + f': [{dataloaders_save_path}]')
    return train_data, valid_data, test_data

def create_samplers(config, dataset, built_datasets):
    """Create sampler for training, validation and testing.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        dataset (Dataset): An instance object of Dataset, which contains all interaction records.
        built_datasets (list of Dataset): A list of split Dataset, which contains dataset for
            training, validation and testing.

    Returns:
        tuple:
            - train_sampler (AbstractSampler): The sampler for training.
            - valid_sampler (AbstractSampler): The sampler for validation.
            - test_sampler (AbstractSampler): The sampler for testing.
    """
    phases = ['train', 'valid', 'test']
    train_neg_sample_args = config['train_neg_sample_args']
    eval_neg_sample_args = config['eval_neg_sample_args']
    sampler = None
    train_sampler, valid_sampler, test_sampler = None, None, None

    if train_neg_sample_args['strategy'] != 'none':
        if not config['repeatable']:
            sampler = Sampler(phases, built_datasets, train_neg_sample_args['distribution'])
        else:
            sampler = RepeatableSampler(phases, dataset, train_neg_sample_args['distribution'])
        train_sampler = sampler.set_phase('train')

    if eval_neg_sample_args['strategy'] != 'none':
        if sampler is None:
            if not config['repeatable']:
                sampler = Sampler(phases, built_datasets, eval_neg_sample_args['distribution'])
            else:
                sampler = RepeatableSampler(phases, dataset, eval_neg_sample_args['distribution'])
        else:
            sampler.set_distribution(eval_neg_sample_args['distribution'])
        valid_sampler = sampler.set_phase('valid')
        test_sampler = sampler.set_phase('test')

    return train_sampler, valid_sampler, test_sampler
def save_split_dataloaders(config, dataloaders):
    """Save split dataloaders.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        dataloaders (tuple of AbstractDataLoader): The split dataloaders.
    """
    ensure_dir(config['checkpoint_dir'])
    save_path = config['checkpoint_dir']
    saved_dataloaders_file = f'{config["dataset"]}-for-{config["model"]}-dataloader.pth'
    file_path = os.path.join(save_path, saved_dataloaders_file)
    logger = getLogger()
    logger.info(set_color('Saving split dataloaders into', 'pink') + f': [{file_path}]')
    with open(file_path, 'wb') as f:
        pickle.dump(dataloaders, f)
def data_preparation(config, dataset):
    """Split the dataset by :attr:`config['eval_args']` and create training, validation and test dataloader.

    Note:
        If we can load split dataloaders by :meth:`load_split_dataloaders`, we will not create new split dataloaders.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        dataset (Dataset): An instance object of Dataset, which contains all interaction records.

    Returns:
        tuple:
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    dataloaders = load_split_dataloaders(config)
    if dataloaders is not None:
        train_data, valid_data, test_data = dataloaders
    else:
        model_type = config['MODEL_TYPE']
        built_datasets = dataset.build()

        train_dataset, valid_dataset, test_dataset = built_datasets
        train_sampler, valid_sampler, test_sampler = create_samplers(config, dataset, built_datasets)

        if model_type != ModelType.KNOWLEDGE:
            train_data = get_dataloader(config, 'train')(config, train_dataset, train_sampler, shuffle=True,mode="downstream")
        else:
            kg_sampler = KGSampler(dataset, config['train_neg_sample_args']['distribution'])
            train_data = get_dataloader(config, 'train')(config, train_dataset, train_sampler, kg_sampler, shuffle=True)

        valid_data = get_dataloader(config, 'evaluation')(config, valid_dataset, valid_sampler, shuffle=False)
        test_data = get_dataloader(config, 'evaluation')(config, test_dataset, test_sampler, shuffle=False)
        if config['save_dataloaders']:
            save_split_dataloaders(config, dataloaders=(train_data, valid_data, test_data))

    logger = getLogger()
    logger.info(
        set_color('[Training]: ', 'pink') + set_color('train_batch_size', 'cyan') + ' = ' +
        set_color(f'[{config["train_batch_size"]}]', 'yellow') + set_color(' negative sampling', 'cyan') + ': ' +
        set_color(f'[{config["neg_sampling"]}]', 'yellow')
    )
    logger.info(
        set_color('[Evaluation]: ', 'pink') + set_color('eval_batch_size', 'cyan') + ' = ' +
        set_color(f'[{config["eval_batch_size"]}]', 'yellow') + set_color(' eval_args', 'cyan') + ': ' +
        set_color(f'[{config["eval_args"]}]', 'yellow')
    )
    return train_data, valid_data, test_data