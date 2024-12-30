import argparse
import os
import torch
import numpy as np
import json
from tqdm import tqdm
from collections import defaultdict
from termcolor import colored

from data_loader.data_loader import fetch_dataloader
from model.model import fetch_model
from loss.loss import compute_loss, compute_metric
from common import tool
from common.manager import Manager
from common.config import Config

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", default="", type=str, help="Directory containing params.json")
parser.add_argument("--debug", "-d", action="store_true", help="Debug")
parser.add_argument("--resume", default="", type=str, help="Path of model weights")


class Tester():
    def __init__(self, cfg):
        # Config status
        self.cfg = cfg

        # Set logger
        self.logger = tool.set_logger(os.path.join(cfg.base.model_dir, "test.log"))

        # Fetch dataloader
        self.logger.info("Dataset: {}".format(cfg.data.name))
        self.dl, self.ds = fetch_dataloader(cfg)

        # Fetch model
        self.model = fetch_model(cfg)

        # Init some recorders
        self.init_status()

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
        for split in ["val", "test"]:
            self.score_status[split] = {"cur": np.inf, "best": np.inf}

    def update_loss_status(self, loss, batch_size):
        for k, v in loss.items():
            self.loss_status[k].update(val=v.item(), num=batch_size)

    def update_metric_status(self, metric, split, batch_size):
        for k, v in metric.items():
            self.metric_status[split][k].update(val=v.item(), num=batch_size)
            self.score_status[split]["cur"] = self.metric_status[split][self.cfg.metric.major_metric].avg

    def reset_loss_status(self):
        for k, v in self.loss_status.items():
            self.loss_status[k].reset()

    def reset_metric_status(self, split):
        for k, v in self.metric_status[split].items():
            self.metric_status[split][k].reset()

    def tqdm_info(self, split):
        if split == "train":
            exp_name = self.cfg.base.model_dir.split("/")[-1]
            print_str = "{} E:{:2d}, lr:{:.2E} ".format(exp_name, self.epoch, self.scheduler.get_last_lr()[0])
            print_str += "loss: {:.4f}/{:.4f}".format(self.loss_status["total"].val, self.loss_status["total"].avg)
        else:
            print_str = ""
            for k, v in self.metric_status[split].items():
                print_str += "{}: {:.4f}/{:.4f}".format(k, v.val, v.avg)
        return print_str

    def print_metric(self, split, only_best=False):
        is_best = self.score_status[split]["cur"] < self.score_status[split]["best"]
        color = "white" if split == "val" else "red"
        print_str = " | ".join("{}: {:.3f}".format(k, v.avg) for k, v in self.metric_status[split].items())
        if only_best:
            if is_best:
                self.logger.info(colored("Best Epoch: {}, {} Results: {}".format(self.epoch, split, print_str), color, attrs=["bold"]))
        else:
            self.logger.info(colored("Epoch: {}, {} Results: {}".format(self.epoch, split, print_str), color, attrs=["bold"]))

    def write_loss_to_tb(self, split):
        if self.step % self.cfg.summary.save_summary_steps == 0:
            for k, v in self.loss_status.items():
                self.loss_writter.add_scalar("{}_loss/{}".format(split, k), v.val, self.step)

    def write_metric_to_tb(self, split):
        for k, v in self.metric_status[split].items():
            self.metric_writter.add_scalar("{}_metric/{}".format(split, k), v.avg, self.epoch)

    def write_custom_info_to_tb(self, input, output, split):
        pass

    def load_ckpt(self):
        state = torch.load(self.cfg.base.resume)

        ckpt_component = []

        if "state_dict" in state and self.model is not None:
            self.model.load_state_dict(state["state_dict"])
            ckpt_component.append("net")

        if not self.cfg.base.only_weights:
            if "optimizer" in state and self.optimizer is not None:
                self.optimizer.load_state_dict(state["optimizer"])
                ckpt_component.append("opt")

            if "scheduler" in state and self.scheduler is not None:
                self.scheduler.load_state_dict(state["scheduler"])
                ckpt_component.append("sch")

            if "step" in state:
                self.step = state["step"]
                ckpt_component.append("step")

            if "epoch" in state:
                self.epoch = state["epoch"]
                ckpt_component.append("epoch")

            if "score_status" in state:
                self.score_status = state["score_status"]
                ckpt_component.append("score status: {}".format(self.score_status))

        ckpt_component = ", ".join(i for i in ckpt_component)
        self.logger.info("Loaded models from: {}".format(self.cfg.base.resume))
        self.logger.info("Ckpt load: {}".format(ckpt_component))

    @torch.no_grad()
    def test(self):
        self.logger.info("Starting test")
        # Load model weigths
        if self.cfg.base.resume is not None:
            self.load_ckpt()
        # Set model to evaluation mode
        torch.cuda.empty_cache()
        self.model.eval()
        # Compute metrics over the dataset
        for split in ["val", "test"]:
            if split not in self.dl:
                continue
            # Initialize loss and metric statuses
            self.reset_loss_status()
            self.reset_metric_status(split)
            # Use tqdm for progress bar
            t = tqdm(total=len(self.dl[split]))
            cur_sample_idx = 0
            for batch_idx, batch_input in enumerate(self.dl[split]):
                # Move data to GPU if available
                batch_input = tool.tensor_gpu(batch_input)
                # Compute model output
                batch_output = self.model(batch_input)
                # Get real batch size
                if "img" in batch_input:
                    batch_size = batch_input["img"].size()[0]
                else:
                    batch_size = self.cfg.test.batch_size
                # # Compute all loss on this batch
                # loss = compute_loss(mng.cfg, batch_input, batch_output)
                # mng.update_loss_status(loss, batch_size)
                # Compute all metrics on this batch
                if "DEX" in self.cfg.data.name:
                    # batch_output = tool.tensor_gpu(batch_output, check_on=False)
                    # batch_output = [{k: v[bid] for k, v in batch_output.items()} for bid in range(batch_size)]
                    # # evaluate
                    # metric = self.ds[split].evaluate(batch_output, cur_sample_idx)
                    # cur_sample_idx += len(batch_output)
                    metric = compute_metric(self.cfg, batch_input, batch_output)
                    self.update_metric_status(metric, split, batch_size)
                else:
                    batch_output = tool.tensor_gpu(batch_output, check_on=False)
                    batch_output = [{k: v[bid] for k, v in batch_output.items()} for bid in range(batch_size)]
                    # evaluate
                    metric = self.ds[split].evaluate(batch_output, cur_sample_idx)
                    cur_sample_idx += len(batch_output)

                # Tqdm settings
                t.set_description(desc="")
                t.update()
            if "DEX" in self.cfg.data.name:
                self.print_metric(split, only_best=False)
            else:
                self.ds[split].print_eval_result(self.epoch)
            t.close()


if __name__ == "__main__":
    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, "cfg.json")
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    cfg = Config(json_path).cfg

    # Update args into cfg.base
    cfg.base.update(vars(args))

    # Use GPU if available
    cfg.base.cuda = torch.cuda.is_available()
    if cfg.base.cuda:
        cfg.base.num_gpu = torch.cuda.device_count()
        torch.backends.cudnn.benchmark = True

    # Only load model weights
    cfg.base.only_weights = True

    # Main function
    tester = Tester(cfg=cfg)
    tester.test()
