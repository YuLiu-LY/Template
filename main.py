import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary
from pytorch_lightning.profiler import PyTorchProfiler

import os
import sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

from models.dinosaur import Dinosaur
from methods.method import SlotAttentionMethod
from methods.utils import ImageLogCallback, set_random_seed, state_dict_ckpt

import argparse
import json
import yaml

from data.birds_dataset import BirdsDataModule
from data.ptr_dataset import PTRDataModule
from data.clevr_dataset import CLEVRDataModule
from data.voc_dataset import VOCDataModule


datamodules = {
    'birds': BirdsDataModule,
    'ptr': PTRDataModule,
    'clevr': CLEVRDataModule,
    'voc': VOCDataModule,
}

monitors = {
    'iou': 'avg_IoU',
    'ari': 'avg_ARI_FG',
    'mbo': 'avg_mBO',
    'ap': 'avg_AP@05',
}

parser = argparse.ArgumentParser()

# Path arguments
parser.add_argument('--log_path', default='')
parser.add_argument('--data_root', default='')
parser.add_argument('--ckpt_path', default='')
parser.add_argument('--config', default='')
# Wandb arguments
parser.add_argument('--entity', default='bigai_vision', help='wandb entity')
parser.add_argument('--project', default='debug', help='wandb project name')
parser.add_argument('--name', default='debug', help='wandb run name')
parser.add_argument('--group', default=None, type=str, help='wandb group name')
parser.add_argument('--tags', nargs='+', default=None, help='wandb tags')
parser.add_argument('--notes', default='', help='notes for this run')
parser.add_argument('--job_type', default='train', help='train, test, debug')
parser.add_argument('--save_code', default=False, action='store_true')
# Data arguments
parser.add_argument('--dataset', default='')
parser.add_argument('--split_name', type=str, default='image', help='split for dataset')
# Training arguments
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--evaluator', type=str, default='iou', help='ari, iou, mbo')
parser.add_argument('--monitor', type=str, default='avg_IoU', help='avg_ARI_FG or avg_IoU or avg_mBO')

parser.add_argument('--devices', type=str, default='0', help='gpu ids, e.g. 0,1,2,3')
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--num_sanity_val_steps', type=int, default=1)
parser.add_argument('--check_val_every_n_epoch', type=int, default=0, help='0 to disable')
parser.add_argument('--check_val_every_n_step', type=int, default=0, help='0 to disable')
parser.add_argument('--log_every_n_steps', type=int, default=50, help='log frequency (steps)')

parser.add_argument('--precision', type=int, default=32, help='16 or 32')
parser.add_argument('--resolution', nargs='+', type=int, default=[224, 224])
parser.add_argument('--n_samples', type=int, default=16, help='number of samples for visualization')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--grad_clip', type=float, default=None, help='None to disable')
parser.add_argument('--grad_clip_algorithm', type=str, default='norm', help='norm or value')

parser.add_argument('--enable_logger', default=False, action='store_true')
parser.add_argument('--load_from_ckpt', default=False, action='store_true')
parser.add_argument('--enable_profiler', default=False, help='enable profiler to trace the time of operations')
# Optimizer arguments
parser.add_argument('--lr', type=float, default=4e-4)
parser.add_argument('--warmup_steps', type=int, default=10000)
parser.add_argument('--decay_steps', type=int, default=100000)
parser.add_argument('--max_steps', type=int, default=250000)
parser.add_argument('--max_epochs', type=int, default=100000)
# Model arguments
parser.add_argument('--num_dec_blocks', type=int, default=4)
parser.add_argument('--vocab_size', type=int, default=4096)
parser.add_argument('--num_dec_heads', type=int, default=4)
parser.add_argument('--num_heads', type=int, default=4)
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--truncate',  type=str, default='bi-level', help='bi-level or fixed-point or none')

parser.add_argument('--num_iter', type=int, default=3)
parser.add_argument('--num_slots', type=int, default=2)
parser.add_argument('--slot_size', type=int, default=256)

parser.add_argument('--init_method', default='embedding', help='embedding, shared_gaussian')

parser.add_argument('--sigma_steps', type=int, default=30000)
parser.add_argument('--sigma_final', type=float, default=0)
parser.add_argument('--sigma_start', type=float, default=1)
parser.add_argument('--use_rescale', default=False, action='store_true')


def main(args):
    print(args)
    set_random_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices

    args.monitor = monitors[args.evaluator]
    datamodule = datamodules[args.dataset](args)
    model = Dinosaur(args)
    if args.job_type == 'test':
        model.load_state_dict(state_dict_ckpt(args.ckpt_path))
        print(f'Load from checkpoint: {args.ckpt_path}')
    method = SlotAttentionMethod(model=model, datamodule=datamodule, args=args)

    if args.enable_logger:
        logger = pl_loggers.WandbLogger(
            project=args.project,
            entity=args.entity,
            group=args.group,
            name=args.name,
            job_type=args.job_type,
            tags=args.tags,
            notes=args.notes,
            save_dir=args.log_path,
            save_code=args.save_code,
        ) 
        method.save_hyperparameters(args)
        callbacks = [
            LearningRateMonitor("step"), 
            ImageLogCallback(), 
            ModelCheckpoint(monitor=args.monitor, save_top_k=1, save_last=True, mode='max'),
            ModelSummary(max_depth=2),
        ]
    else:
        logger = False
        callbacks = []

    profiler = None
    if args.enable_profiler:
        profiler = PyTorchProfiler(filename='profiler')

    kwargs = {
        'resume_from_checkpoint': args.ckpt_path if args.load_from_ckpt else None,
        'logger': logger,
        'default_root_dir': args.log_path,
        'num_sanity_val_steps': args.num_sanity_val_steps,
        'accelerator': 'gpu',
        'devices': -1,
        'strategy': 'ddp' if len(args.devices.split(',')) > 1 else None,
        'max_steps': args.max_steps,
        'max_epochs': args.max_epochs,
        'log_every_n_steps': args.log_every_n_steps,
        'callbacks': callbacks,
        'gradient_clip_val': args.grad_clip,
        'gradient_clip_algorithm': args.grad_clip_algorithm,
        'precision': args.precision,
        'profiler': profiler,
    }
    if args.check_val_every_n_step > 0:
        kwargs['val_check_interval'] = args.check_val_every_n_step
    if args.check_val_every_n_epoch > 0:
        kwargs['check_val_every_n_epoch'] = args.check_val_every_n_epoch
    if args.job_type == 'debug':
        kwargs['fast_dev_run'] = True

    trainer = Trainer(**kwargs)
    if args.job_type == 'test':
        trainer.test(method, datamodule=datamodule)
    else:
        trainer.fit(method, datamodule=datamodule)

if __name__ == "__main__":
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    parser.set_defaults(**config)
    args = parser.parse_args()

    paths = json.load(open('./path.json', 'r'))
    data_paths = paths['data_paths']
    args.data_root = data_paths[args.dataset]
    args.log_path = os.path.join(paths['log_path'], args.dataset)
    main(args)

# salloc -N 1 -t 30:00 --gres=gpu:1 --qos=debug --cpus-per-task=32 --job-name=debug --partition=gpu
# salloc -N 1 -t 24:00:00 --gres=gpu:1 --qos=gpu --cpus-per-task=32 --partition=gpu