import argparse
import os
import datetime
import numpy as np
from omegaconf import OmegaConf
import importlib
import shutil

import torch
import torch.nn as nn
import torch.distributed as dist
import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from lars import LARS, LARSWrapper
from utils import save_codes

def load_config(config_path):
    """
    Load the config by the given path
    """
    config = OmegaConf.load(config_path)

    return config

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

class SSOLE(pl.LightningModule):
    def __init__(self, config):
        super(SSOLE, self).__init__()
        self.config = config
        self.ffcv = 'ffcv' in config.data["target"]
        self.use_lars = getattr(config.train, 'LARS', True)
        # self.total_crop = np.sum(self.config.data.params.nmb_crops)
        encoder = instantiate_from_config(config.encoder)
        self.encoder = nn.SyncBatchNorm.convert_sync_batchnorm(encoder)
        self.loss = instantiate_from_config(config.loss)

    def forward(self, x):
        return self.encoder(x)

    def training_step(self, batch, batch_idx):
        if not self.ffcv:
            imgs = batch
        else:
            imgs = list(batch[:1] + batch[2:])
        z = self(imgs)
        loss, info_dict = self.loss(z)
        if self.use_lars:
            info_dict['lr'] = self.optimizer.optim.param_groups[0]["lr"]
        else:
            info_dict['lr'] = self.optimizer.param_groups[0]["lr"]
        self.log_dict(info_dict, prog_bar=True, on_step=True, on_epoch=False)

        # Skip this step if loss is NaN on any GPU
        loss_tensor = torch.tensor([loss], device=self.device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        if torch.isnan(loss_tensor).any():
            return None  # Skip this step if loss is NaN on any GPU

        return loss
    
    def train_dataloader(self):
        dataset = instantiate_from_config(self.config.data)
        return dataset._data_loader(self.config.train.batch_size, self.config.train.workers)

    def configure_optimizers(self):
        if self.use_lars:
            optimizer = torch.optim.SGD(
                self.encoder.parameters(),
                lr=self.config.train.base_lr,
                momentum=0.9,
                weight_decay=self.config.train.wd,
            )
            self.optimizer = LARSWrapper(optimizer=optimizer, eta=0.001, clip=False)
        else:
            param_weights = []
            param_biases = []
            for param in self.encoder.parameters():
                if param.ndim == 1:
                    param_biases.append(param)
                else:
                    param_weights.append(param)
            optimizer = torch.optim.SGD([{'params': param_biases, 'weight_decay': 0,},
                                {'params': param_weights, 'weight_decay': self.config.train.wd}],
                                lr=self.config.train.base_lr, momentum=0.9)
            self.optimizer = optimizer
        return self.optimizer

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)

    args = parser.parse_args()

    config = load_config(args.config)

    pl.seed_everything(config.train.seed)

    log_path = os.path.join('logs', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') +'_' + config.train.log_dir)

    # ckpt_path = os.path.join(log_path, "ckpt")
    # os.makedirs(ckpt_path, exist_ok=True)

    logger = TensorBoardLogger(save_dir=log_path, name="logs")

    callbacks = [ModelCheckpoint(dirpath=os.path.join(log_path, "ckpt"), monitor='loss', save_last=True, save_top_k=config.train.save_top_k),]
    
    for callback_config in config.train.callbacks:
        if callback_config is not None:
            callbacks.append(instantiate_from_config(callback_config))

    
    model = SSOLE(config)

    trainer = pl.Trainer(
        accelerator="gpu",
        precision='16-mixed',
        devices=config.train.devices,
        num_nodes=1,
        enable_progress_bar=True,
        logger=logger,
        callbacks=callbacks,
        max_epochs=config.train.epochs,
    )
    
    @rank_zero_only
    def make_dirs():
        os.makedirs(log_path, exist_ok=True)
        code_path = os.path.join(log_path, "codes")
        os.makedirs(code_path, exist_ok=True)
        save_codes(src=".", dst=code_path, cfg=args.config)
    make_dirs()

    trainer.fit(model, ckpt_path=getattr(config.train, "resume", None))
    
if __name__ == "__main__":
    train()