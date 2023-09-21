from pathlib import Path
import datetime

import torch
import wandb
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, ModelSummary
from pytorch_lightning.loggers import WandbLogger

from utils.filelogger import FilesystemLogger


def generate_exp_name(cfg):
    return f"{cfg.exp_name}_{datetime.datetime.now().strftime('%m%d%H%M')}"

def create_trainer(cfg):
    cfg.exp_name = generate_exp_name(cfg)
    if cfg.val_check_interval > 1:
        cfg.val_check_interval = int(cfg.val_check_interval)
    seed_everything(cfg.seed, workers=True)

    # save code files
    filesystem_logger = FilesystemLogger(cfg)
    if cfg.logger == 'wandb':
        logger = WandbLogger(
            project=cfg.project,
            entity=cfg.entity,
            group=cfg.group,
            name=cfg.exp_name,
            job_type=cfg.job_type,
            tags=cfg.tags,
            notes=cfg.notes,
            id=cfg.exp_name, 
            # settings=wandb.Settings(start_method='thread')
        )
    else:
        logger = False

    checkpoint_callback = ModelCheckpoint(dirpath=(Path(cfg.log_path) / cfg.exp_name / "checkpoints"),
                                          monitor=cfg.monitor,
                                          save_top_k=3,
                                          save_last=True,
                                          mode='max',
                                          )
    callbacks = [LearningRateMonitor("step"),  checkpoint_callback, ModelSummary(max_depth=2)] if logger else []
    gpu_count = torch.cuda.device_count()
    if cfg.job_type == 'debug':
        cfg.train_percent = 15
        cfg.val_percent = 5
        cfg.test_percent = 5
        cfg.val_check_interval = 1

    kwargs = {
        'logger': logger,
        'accelerator': 'gpu',
        'devices': gpu_count,
        'num_sanity_val_steps': 1,
        'max_steps': cfg.max_steps,
        'max_epochs': cfg.max_epochs,
        'limit_train_batches': cfg.train_percent,
        'limit_val_batches': cfg.val_percent,
        'limit_test_batches': cfg.test_percent,
        'val_check_interval': float(min(cfg.val_check_interval, 1)),
        'check_val_every_n_epoch': max(1, cfg.val_check_interval),
        'callbacks': callbacks,
        'gradient_clip_val': cfg.grad_clip if cfg.grad_clip > 0 else None,
        'precision': cfg.precision,
        'profiler': cfg.profiler,
        'benchmark': cfg.benchmark,
        'deterministic': cfg.deterministic,
    }
    if gpu_count > 1:
        kwargs['strategy'] = 'ddp'
    trainer = Trainer(**kwargs)
    return trainer

