import math
import numpy as np
import time
from datetime import timedelta
import pandas as pd
import argparse
import logging
from logging import getLogger
import pickle
import os
import shutil
import importlib
from omegaconf import OmegaConf

import abc
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

import torch.distributed as dist
from torch.utils.data import Sampler

FALSY_STRINGS = {"off", "false", "0"}
TRUTHY_STRINGS = {"on", "true", "1"}

logger = getLogger()

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

def save_codes(src, dst, cfg):
    for root, dirs, files in os.walk(src):
        if root in [".", "./data", "./models", "./models/backbones", "./loss", "./loss_ab"]:
            new_dst = os.path.join(dst, root)
            os.makedirs(new_dst, exist_ok=True)
            for file in files:
                if file.endswith(".py"):
                    path = os.path.join(root, file)
                    shutil.copy(path, new_dst)
    shutil.copy(cfg, dst)

class LogFormatter:
    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime("%x %X"),
            timedelta(seconds=elapsed_seconds),
        )
        message = record.getMessage()
        message = message.replace("\n", "\n" + " " * (len(prefix) + 3))
        return "%s - %s" % (prefix, message) if message else ""


def create_logger(filepath, rank):
    """
    Create a logger.
    Use a different log file for each process.
    """
    # create log formatter
    log_formatter = LogFormatter()

    # create file handler and set level to debug
    if filepath is not None:
        if rank > 0:
            filepath = "%s-%i" % (filepath, rank)
        file_handler = logging.FileHandler(filepath, "a")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(log_formatter)

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # create logger and set level to debug
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if filepath is not None:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()

    logger.reset_time = reset_time

    return logger


class PD_Stats(object):
    """
    Log stuff with pandas library
    """

    def __init__(self, path, columns):
        self.path = path

        # reload path stats
        if os.path.isfile(self.path):
            self.stats = pd.read_pickle(self.path)

            # check that columns are the same
            assert list(self.stats.columns) == list(columns)

        else:
            self.stats = pd.DataFrame(columns=columns)

    def update(self, row, save=True):
        self.stats.loc[len(self.stats.index)] = row

        # save the statistics
        if save:
            self.stats.to_pickle(self.path)


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def init_distributed_mode(args):
    """
    Initialize the following variables:
        - world_size
        - rank
    """

    args.is_slurm_job = "SLURM_JOB_ID" in os.environ

    if args.is_slurm_job:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.world_size = int(os.environ["SLURM_NNODES"]) * int(
            os.environ["SLURM_TASKS_PER_NODE"][0]
        )
    else:
        # multi-GPU job (local or multi-node) - jobs started with torch.distributed.launch
        # read environment variables
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])

    # prepare distributed
    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    # set cuda device
    args.gpu_to_work_on = args.rank % torch.cuda.device_count()
    torch.cuda.set_device(args.gpu_to_work_on)
    return


def initialize_exp(params, *args, dump_params=True):
    """
    Initialize the experience:
    - dump parameters
    - create checkpoint repo
    - create a logger
    - create a panda object to keep track of the training statistics
    """

    # dump parameters
    if dump_params:
        os.makedirs(os.path.join(params.dump_path, "params"), exist_ok=True)
        pickle.dump(params, open(os.path.join(params.dump_path, "params", "params.pkl"), "wb"))

    # create repo to store checkpoints
    params.dump_checkpoints = os.path.join(params.dump_path, "checkpoints")
    if not params.rank and not os.path.isdir(params.dump_checkpoints):
        os.mkdir(params.dump_checkpoints)

    # create a panda object to log loss and acc
    os.makedirs(os.path.join(params.dump_path, "stats"), exist_ok=True)
    training_stats = PD_Stats(
        os.path.join(params.dump_path, "stats", "stats" + str(params.rank) + ".pkl"), args
    )

    # create a logger
    os.makedirs(os.path.join(params.dump_path, "logs"), exist_ok=True)
    logger = create_logger(
        os.path.join(params.dump_path, "logs", "train.log"), rank=params.rank
    )
    if dist.get_rank() == 0:
        logger.info("============ Initialized logger ============")
        logger.info(
            "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(params)).items()))
        )
        logger.info("The experiment will be stored in %s\n" % params.dump_path)
        logger.info("")
    return logger, training_stats


def restart_from_checkpoint(ckp_paths, run_variables=None, **kwargs):
    """
    Re-start from checkpoint
    """
    # look for a checkpoint in exp repository
    if isinstance(ckp_paths, list):
        for ckp_path in ckp_paths:
            if os.path.isfile(ckp_path):
                break
    else:
        ckp_path = ckp_paths

    if not os.path.isfile(ckp_path):
        return

    rank = dist.get_rank()

    if rank == 0:
        logger.info("Found checkpoint at {}".format(ckp_path))

    # open checkpoint file
    checkpoint = torch.load(
        ckp_path, map_location="cuda:" + str(rank % torch.cuda.device_count())
    )

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
            except TypeError:
                msg = value.load_state_dict(checkpoint[key])
            if rank == 0:
                logger.info("=> loaded {} from checkpoint '{}'".format(key, ckp_path))
                logger.info("=> msg: {}".format(msg))
        else:
            if rank == 0:
                logger.warning(
                    "=> failed to load {} from checkpoint '{}'".format(key, ckp_path)
                )

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class PaceAverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self, pace=100):
        self.pace = pace
        self.reset()

    def reset(self):
        self.val = 0
        self.val_queue = []
        self.avg = 0
        self.count = 0

    def update(self, val):
        self.val = val
        self.val_queue.append(val)
        self.count += 1
        if self.count < self.pace:
            self.avg = sum(self.val_queue) / self.count
        else:
            self.val_queue = self.val_queue[-self.pace:]
            self.avg = sum(self.val_queue) / self.pace


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[torch.distributed.get_rank()]
        return grad_out

class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        torch.distributed.all_reduce(all_gradients)
        return all_gradients[torch.distributed.get_rank()]

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

def cust_norm(X, ep=1e-6, dim=-1):
    X = X.sign()*(X.abs().clamp(min=ep))
    return nn.functional.normalize(X, dim=dim)

class DecayLRCosWarmUp(pl.Callback):
    def __init__(self, base_lr, final_lr, warmup_eps):
        super().__init__()
        self.base_lr = base_lr
        self.final_lr = final_lr
        self.warmup_eps = warmup_eps

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        warmup_steps = trainer.num_training_batches * self.warmup_eps
        if trainer.global_step < warmup_steps:
            lr = self.base_lr * (trainer.global_step + 1) / warmup_steps
        else:
            steps = trainer.global_step - warmup_steps
            max_steps = pl_module.config.train.epochs * trainer.num_training_batches - warmup_steps
            q = 0.5 * (1 + math.cos(math.pi * (steps + 1) / max_steps))
            lr = self.base_lr * q + self.final_lr * (1 - q)

        for g in pl_module.optimizer.param_groups:
            g['lr'] = lr

class DecayMomentumCos(pl.Callback):
    def __init__(self, base_m):
        super().__init__()
        self.base_m = base_m

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        pl_module.encoder.m = 1. - 0.5 * (1. + math.cos(math.pi * trainer.global_step / (pl_module.config.train.epochs * trainer.num_training_batches))) * (1. - self.base_m)

class DecayLRCos(pl.Callback):
    def __init__(self, base_lr, final_lr):
        super().__init__()
        self.base_lr = base_lr
        self.final_lr = final_lr

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        t = 0.5 * (self.base_lr - self.final_lr) * (1 + math.cos(math.pi * trainer.global_step / (pl_module.config.train.epochs * trainer.num_training_batches))) + self.final_lr
        for g in pl_module.optimizer.param_groups:
            g['lr'] = t