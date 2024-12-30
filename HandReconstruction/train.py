import argparse
import os
import torch
import numpy as np
from functools import partial
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from termcolor import colored

from tqdm import tqdm
from data_loader.data_loader import fetch_dataloader
from model.model import fetch_model
from optimizer.optimizer import fetch_optimizer
from loss.loss import compute_loss, compute_metric
from common import tool
from common.manager import Manager
from common.config import Config

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='', type=str, help='Directory containing params.json')
parser.add_argument('--resume', default=None, type=str, help='Path of model weights')
parser.add_argument('--debug', '-d', action='store_true', help='Debug')
parser.add_argument('--only_weights', '-ow', action='store_true', help='Only load model weights or load all train status')


class Trainer():

    def __init__(self, cfg):
        # Config status
        self.cfg = cfg

        # Set logger
        self.logger = tool.set_logger(os.path.join(cfg.base.model_dir, 'train.log'))

        # Fetch dataloader
        self.logger.info(f'Dataset: {cfg.data.name}')
        self.dl, self.ds = fetch_dataloader(cfg)

        # Fetch model
        self.model = fetch_model(cfg)

        # Define optimizer and scheduler
        self.optimizer, self.scheduler = fetch_optimizer(cfg, self.model)

        # Init some recorders
        self.init_status()
        self.init_tb()

    def init_status(self):
        self.epoch = 0
        self.step = 0
        # Train status: model, optimizer, scheduler, epoch, step
        self.train_status = {}
        # Loss status
        self.loss_status = defaultdict(tool.AverageMeter)
        # Metric status: val, test
        self.metric_status = defaultdict(lambda: defaultdict(tool.AverageMeter))
        # Score status: val, test
        self.score_status = {}
        for split in ['val', 'test']:
            self.score_status[split] = {'cur': np.inf, 'best': np.inf}

    def init_tb(self):
        # Tensorboard
        loss_tb_dir = os.path.join(self.cfg.base.model_dir, 'summary/loss')
        os.makedirs(loss_tb_dir, exist_ok=True)
        self.loss_writter = SummaryWriter(log_dir=loss_tb_dir)
        metric_tb_dir = os.path.join(self.cfg.base.model_dir, 'summary/metric')
        os.makedirs(metric_tb_dir, exist_ok=True)
        self.metric_writter = SummaryWriter(log_dir=metric_tb_dir)

    def update_step(self):
        self.step += 1

    def update_epoch(self):
        self.epoch += 1

    def update_loss_status(self, loss, batch_size):
        for k, v in loss.items():
            self.loss_status[k].update(val=v.item(), num=batch_size)

    def update_metric_status(self, metric, split, batch_size):
        for k, v in metric.items():
            self.metric_status[split][k].update(val=v.item(), num=batch_size)
            self.score_status[split]['cur'] = self.metric_status[split][self.cfg.metric.major_metric].avg

    def reset_loss_status(self):
        for k, v in self.loss_status.items():
            self.loss_status[k].reset()

    def reset_metric_status(self, split):
        for k, v in self.metric_status[split].items():
            self.metric_status[split][k].reset()

    def tqdm_info(self, split):
        if split == 'train':
            exp_name = self.cfg.base.model_dir.split('/')[-1]
            print_str = f'{exp_name}, E:{self.epoch:3d}, lr:{self.scheduler.get_last_lr()[0]:.2E}, '
            print_str += f'loss: {self.loss_status["total"].val:.4g}/{self.loss_status["total"].avg:.4g}'
        else:
            print_str = ''
            for k, v in self.metric_status[split].items():
                print_str += f'{k}: {v.val:.4g}/{v.avg:.4g}'
        return print_str

    def print_metric(self, split, only_best=False):
        is_best = self.score_status[split]['cur'] < self.score_status[split]['best']
        color = 'white' if split == 'val' else 'red'
        print_str = ' | '.join(f'{k}: {v.avg:.4g}' for k, v in self.metric_status[split].items())
        if only_best:
            if is_best:
                self.logger.info(colored(f'Best Epoch: {self.epoch}, {split} Results: {print_str}', color, attrs=['bold']))
        else:
            self.logger.info(colored(f'Epoch: {self.epoch}, {split} Results: {print_str}', color, attrs=['bold']))

    def write_loss_to_tb(self, split):
        if self.step % self.cfg.summary.save_summary_steps == 0:
            for k, v in self.loss_status.items():
                self.loss_writter.add_scalar(f'{split}_loss/{k}', v.val, self.step)

    def write_metric_to_tb(self, split):
        for k, v in self.metric_status[split].items():
            self.metric_writter.add_scalar(f'{split}_metric/{k}', v.avg, self.epoch)

    def write_custom_info_to_tb(self, input, output, split):
        pass

    def save_ckpt(self):
        # Save latest and best metrics
        for split in ['val', 'test']:
            if split not in self.dl:
                continue
            latest_metric_path = os.path.join(self.cfg.base.model_dir, f'{split}_metric_latest.json')
            tool.save_dict_to_json(self.metric_status[split], latest_metric_path)
            is_best = self.score_status[split]['cur'] < self.score_status[split]['best']
            if is_best:
                self.score_status[split]['best'] = self.score_status[split]['cur']
                best_metric_path = os.path.join(self.cfg.base.model_dir, f'{split}_metric_best.json')
                tool.save_dict_to_json(self.metric_status[split], best_metric_path)

        # Model states
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'step': self.step,
            'epoch': self.epoch,
            'score_status': self.score_status
        }

        # Save middle checkpoint
        if self.epoch % 20 == 0:
            middle_ckpt_path = os.path.join(self.cfg.base.model_dir, f'model_{self.epoch}.pth')
            torch.save(state, middle_ckpt_path)

        # Save latest checkpoint
        if self.epoch % self.cfg.summary.save_latest_freq == 0:
            latest_ckpt_path = os.path.join(self.cfg.base.model_dir, 'model_latest.pth')
            torch.save(state, latest_ckpt_path)

        # Save latest and best checkpoints
        for split in ['val', 'test']:
            if split not in self.dl:
                continue
            # Above code has updated the best score to cur
            is_best = self.score_status[split]['cur'] == self.score_status[split]['best']
            if is_best:
                self.logger.info(f'Current is {split} best, score={self.score_status[split]["best"]:.3f}')
                # Save best checkpoint
                if self.epoch > self.cfg.summary.save_best_after:
                    best_ckpt_path = os.path.join(self.cfg.base.model_dir, f'{split}_model_best.pth')
                    torch.save(state, best_ckpt_path)

    def load_ckpt(self):
        state = torch.load(self.cfg.base.resume)

        ckpt_component = []

        if 'state_dict' in state and self.model is not None:
            self.model.load_state_dict(state['state_dict'])
            ckpt_component.append('net')

        if not self.cfg.base.only_weights:
            if 'optimizer' in state and self.optimizer is not None:
                self.optimizer.load_state_dict(state['optimizer'])
                ckpt_component.append('opt')

            if 'scheduler' in state and self.scheduler is not None:
                self.scheduler.load_state_dict(state['scheduler'])
                ckpt_component.append('sch')

            if 'step' in state:
                self.step = state['step']
                ckpt_component.append('step')

            if 'epoch' in state:
                self.epoch = state['epoch']
                ckpt_component.append('epoch')

            if 'score_status' in state:
                self.score_status = state['score_status']
                ckpt_component.append(f'score status: {self.score_status}')

        ckpt_component = ', '.join(i for i in ckpt_component)
        self.logger.info(f'Loaded models from: {self.cfg.base.resume}')
        self.logger.info(f'Ckpt load: {ckpt_component}')

    def train(self):
        # Reset loss status
        self.reset_loss_status()
        # Set model to training mode
        torch.cuda.empty_cache()
        self.model.train()
        # Use tqdm for progress bar
        t = tqdm(dynamic_ncols=True, total=len(self.dl['train']))
        # Train loop
        for batch_idx, batch_input in enumerate(self.dl['train']):
            # Move input to GPU if available
            batch_input = tool.tensor_gpu(batch_input)
            # Compute model output and loss
            batch_output = self.model(batch_input)
            loss = compute_loss(self.cfg, batch_input, batch_output)
            loss['total'].backward()
            # Update loss status and print current loss and average loss
            self.update_loss_status(loss=loss, batch_size=self.cfg.train.batch_size)

            # add gradient clip
            if 'grad_norm_clip' in self.cfg.train:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_norm_clip)

            # Clean previous gradients, compute gradients of all variables wrt loss
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Update step: step += 1
            self.update_step()
            # Write loss to tensorboard
            self.write_loss_to_tb(split='train')
            # Write custom info to tensorboard
            self.write_custom_info_to_tb(batch_input, batch_output, split='train')
            # Training info print
            print_str = self.tqdm_info(split='train')
            # Tqdm settings
            t.set_description(desc=print_str)
            t.update()

            if self.cfg.base.debug:
                if batch_idx == 3:
                    break

        # Close tqdm
        t.close()

    @torch.no_grad()
    def evaluate(self):
        # Set model to evaluation mode
        torch.cuda.empty_cache()
        self.model.eval()
        # Compute metrics over the dataset
        for split in ['val', 'test']:
            if split not in self.dl:
                continue
            # Initialize loss and metric statuses
            self.reset_loss_status()
            self.reset_metric_status(split)
            cur_sample_idx = 0
            for batch_idx, batch_input in enumerate(self.dl[split]):
                # Move data to GPU if available
                batch_input = tool.tensor_gpu(batch_input)
                # Compute model output
                batch_output = self.model(batch_input)
                # Get real batch size
                if 'img' in batch_input:
                    batch_size = batch_input['img'].size()[0]
                else:
                    batch_size = self.cfg.test.batch_size
                # # Compute all loss on this batch
                # loss = compute_loss(mng.cfg, batch_input, batch_output)
                # mng.update_loss_status(loss, batch_size)
                # Compute all metrics on this batch

                metric = compute_metric(self.cfg, batch_input, batch_output)
                metric = tool.tensor_gpu(metric, check_on=False)
                self.update_metric_status(metric, split, batch_size)

            # Update data to tensorboard
            self.write_metric_to_tb(split)
            # # Write custom info to tensorboard
            # mng.write_custom_info_to_tb(batch_input, batch_output, split)
            # For each epoch, update and print the metric
            self.print_metric(split, only_best=False)

    def train_and_evaluate(self):
        self.logger.info(f'Starting training for {self.cfg.train.num_epochs} epoch(s)')
        # Load weights from restore_file if specified
        if self.cfg.base.resume is not None:
            self.load_ckpt()

        for epoch in range(self.epoch, self.cfg.train.num_epochs):
            # Train one epoch
            self.train()
            # Evaluate one epoch, check if current is best, save best and latest checkpoints
            self.evaluate()
            # Update scheduler
            self.scheduler.step()
            # Update epoch: epoch += 1
            self.update_epoch()
            # Save checkpoint
            self.save_ckpt()


if __name__ == '__main__':
    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'cfg.json')
    assert os.path.isfile(json_path), f'No json configuration file found at {json_path}'
    cfg = Config(json_path).cfg

    # Update args into cfg.base
    cfg.base.update(vars(args))

    # Use GPU if available
    cfg.base.cuda = torch.cuda.is_available()
    if cfg.base.cuda:
        cfg.base.num_gpu = torch.cuda.device_count()
        torch.backends.cudnn.benchmark = True

    # Main function
    trainer = Trainer(cfg=cfg)
    trainer.train_and_evaluate()
