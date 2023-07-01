import os
from time import time
from tqdm import tqdm
import torch
import numpy as np
from torch.nn.utils.clip_grad import clip_grad_norm_
from trainer import Trainer
from recbole.utils import set_color, get_gpu_usage,early_stopping,dict2str


class DDPPretrainTrainer(Trainer):
    def __init__(self, config, model):
        super(DDPPretrainTrainer, self).__init__(config, model)
        self.pretrain_epochs = self.epochs
        self.save_step = self.config['save_step']
        self.rank = config['rank']
        self.world_size = config['world_size']
        self.lrank = self._build_distribute(rank=self.rank, world_size=self.world_size)
        self.logger.info(f'Let\'s use {torch.cuda.device_count()} GPUs to train {self.config["model"]} ...')

    def _build_distribute(self, rank, world_size):
        from torch.nn.parallel import DistributedDataParallel
        # 1 set backend
        torch.distributed.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        # 2 get distributed id
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        device_dis = torch.device("cuda", local_rank)
        # 3, 4 assign model to be distributed
        self.model.to(device_dis)
        self.model = DistributedDataParallel(self.model, 
                                             device_ids=[local_rank],
                                             output_device=local_rank).module
        return local_rank

    def save_pretrained_model(self, epoch, saved_model_file):
        r"""Store the model parameters information and training information.

        Args:
            epoch (int): the current epoch id
            saved_model_file (str): file name for saved pretrained model

        """
        state = {
            'config': self.config,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': [optimizer.state_dict() for optimizer in self.optimizers],
        }
        torch.save(state, saved_model_file)

    def _trans_dataload(self, interaction):
        from torch.utils.data import DataLoader
        from torch.utils.data.distributed import DistributedSampler

        #using pytorch dataload to re-wrap dataset
        def sub_trans(dataset):
            dis_loader = DataLoader(dataset=dataset,
                                    batch_size=dataset.shape[0],
                                    sampler=DistributedSampler(dataset, shuffle=False))
            for data in dis_loader:
                batch_data = data

            return batch_data
        #change `interaction` datatype to a python `dict` object.  
        #for some methods, you may need transfer more data unit like the following way.  

        data_dict = {}
        for k, v in interaction.interaction.items():
            data_dict[k] = sub_trans(v)
        return data_dict

    def _train_epoch(self, train_data, epoch_idx,verbose=True, loss_func=None, show_progress=False):
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = []
        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                ncols=100,
                desc=set_color(f"Train {epoch_idx:>5}", 'pink'),
            ) if show_progress else train_data
        )
        for batch_idx, interaction in enumerate(iter_data):
            self.global_step += 1
            interaction = interaction.to(self.device)
            interaction = self._trans_dataload(interaction)
            for optimizer in self.optimizers:
                optimizer.zero_grad()
            loss = loss_func(interaction)
            total_loss.append(loss.item())
            if self.lrank == 0:
                self.tensorboard.add_scalar('train loss', loss, self.global_step)
                self.tensorboard.add_scalar(f"lr",self.optimizers[0].state_dict()['param_groups'][0]['lr'],self.global_step)
            self._check_nan(loss)
            loss.backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            for optimizer,scheduler,schedule in zip(self.optimizers,self.schedulers, self.schedules):
                optimizer.step()
                if scheduler is not None:
                    if schedule == "reduce":
                        scheduler.step(self.best_valid_score)
                    else:
                        scheduler.step()
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))
        return np.mean(total_loss)

    def pretrain(self, train_data, verbose=True, show_progress=False):
        for epoch_idx in range(self.start_epoch, self.pretrain_epochs):
            # train
            training_start_time = time()
            train_loss = self._train_epoch(train_data, epoch_idx,verbose, show_progress=show_progress)
            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            if verbose:
                self.logger.info(train_loss_output)
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss)
            
            if (epoch_idx + 1) % self.save_step == 0:
                saved_model_file = os.path.join(
                    self.checkpoint_dir,
                    '{}-{}.pth'.format(self.config['log_name'], str(epoch_idx + 1))
                )
                self.save_pretrained_model(epoch_idx, saved_model_file)
                update_output = set_color('Saving current', 'blue') + ': %s' % saved_model_file
                if verbose:
                    self.logger.info(update_output)
