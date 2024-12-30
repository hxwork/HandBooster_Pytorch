import random
import torch
import os
import numpy as np
from multiprocessing import cpu_count
from torch.utils.data import DataLoader

from data_loader.HO3D import HO3D
from data_loader.DEX_YCB import DEX_YCB

from data_loader.transforms import fetch_transforms


def worker_init_fn(worker_id):
    rand_seed = random.randint(0, 2**32 - 1)
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)


def fetch_dataloader(cfg):
    # Train and test transforms
    train_transforms, test_transforms = fetch_transforms(cfg)
    # Train dataset
    train_ds = eval(cfg.data.name)(cfg, train_transforms, 'train')
    # Val dataset
    if 'val' in cfg.data.eval_type:
        val_ds = eval(cfg.data.name)(cfg, test_transforms, 'val')
    elif 'train' in cfg.data.eval_type:
        val_ds = eval(cfg.data.name)(cfg, test_transforms, 'train')
    # Test dataset
    if 'test' in cfg.data.eval_type:
        test_ds = eval(cfg.data.name)(cfg, test_transforms, 'test')

    # Data loader
    train_dl = DataLoader(train_ds, batch_size=cfg.train.batch_size, num_workers=cpu_count(), pin_memory=True, shuffle=True, drop_last=True, worker_init_fn=worker_init_fn)

    if 'val' in cfg.data.eval_type:
        val_dl = DataLoader(val_ds, batch_size=cfg.test.batch_size, num_workers=cpu_count(), pin_memory=True, shuffle=False)
    elif 'train' in cfg.data.eval_type:
        val_dl = DataLoader(train_ds, batch_size=cfg.test.batch_size, num_workers=cpu_count(), pin_memory=True, shuffle=False)
    else:
        val_dl = None

    if 'test' in cfg.data.eval_type:
        test_dl = DataLoader(test_ds, batch_size=cfg.test.batch_size, num_workers=cpu_count(), pin_memory=True, shuffle=False)
    else:
        test_dl = None

    dl, ds = {}, {}
    dl['train'], ds['train'] = train_dl, train_ds
    if val_dl is not None:
        dl['val'], ds['val'] = val_dl, val_ds
    if test_dl is not None:
        dl['test'], ds['test'] = test_dl, test_ds

    return dl, ds
