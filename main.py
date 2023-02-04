import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import os
import sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

from models.dinosaur import Dinosaur
from methods.method import SlotAttentionMethod
from methods.utils import ImageLogCallback, set_random_seed, state_dict_ckpt

import tensorboard
import argparse
import json

from data.birds_dataset import BirdsDataModule
from data.shapestacks_dataset import ShapeStacksDataModule
from data.flowers_dataset import FlowersDataModule
from data.ptr_dataset import PTRDataModule
from data.clevrtex_dataset import CLEVRTEXDataModule
from data.dogs_dataset import DogsDataModule
from data.cars_dataset import CarsDataModule
from data.ycb_dataset import YCBDataModule
from data.scannet_dataset import ScanNetDataModule
from data.coco_dataset import COCODataModule
from data.objectsroom_dataset import ObjectsRoomDataModule
from data.clevr_dataset import CLEVRDataModule
from data.voc_dataset import VOCDataModule


datamodules = {
    'shapestacks': ShapeStacksDataModule,
    'birds': BirdsDataModule,
    'dogs': DogsDataModule,
    'cars': CarsDataModule,
    'clevrtex': CLEVRTEXDataModule,
    'ptr': PTRDataModule,
    'flowers': FlowersDataModule,
    'objectsroom': ObjectsRoomDataModule,
    'ycb': YCBDataModule,
    'scannet': ScanNetDataModule,
    'coco': COCODataModule,
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

parser.add_argument('--dataset', default='')
parser.add_argument('--data_root', default='')
parser.add_argument('--split_name', type=str, default='image', help='split for CLEVRTEX')
parser.add_argument('--project_name', default='')
parser.add_argument('--log_name', default='test')
parser.add_argument('--log_path', default='')
parser.add_argument('--ckpt_path', default='.ckpt')
parser.add_argument('--test_ckpt_path', default='ckpt.pt.tar')
parser.add_argument('--max_n_objects', default=3, type=int)

parser.add_argument('--evaluate', type=str, default='iou', help='ari, iou, mbo')
parser.add_argument('--monitor', type=str, default='avg_IoU', help='avg_ARI_FG or avg_IoU or avg_mBO')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--num_sanity_val_steps', type=int, default=1)
parser.add_argument('--check_val_every_n_epoch', type=int, default=0)
parser.add_argument('--check_val_every_n_step', type=int, default=0)
parser.add_argument('--n_samples', type=int, default=16)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--gpus', type=int, default=0)
parser.add_argument('--device', type=str, default='0')

parser.add_argument('--grad_clip', type=float, default=1.0)
parser.add_argument('--resolution', nargs='+', type=int, default=[224, 224])

parser.add_argument('--is_logger_enabled', default=False, action='store_true')
parser.add_argument('--load_from_ckpt', default=False, action='store_true')
parser.add_argument('--use_rescale', default=True, action='store_false')

parser.add_argument('--drop_path', type=float, default=0)
parser.add_argument('--dvae_kernel_size', type=int, default=3)
parser.add_argument('--truncate',  type=str, default='bi-level', help='bi-level or fixed-point or none')

parser.add_argument('--lr_main', type=float, default=4e-4)
parser.add_argument('--warmup_steps', type=int, default=10000)
parser.add_argument('--max_steps', type=int, default=250000)
parser.add_argument('--decay_steps', type=int, default=100000)
parser.add_argument('--max_epochs', type=int, default=100000)

parser.add_argument('--dino_type', type=str, default='vitb16')
parser.add_argument('--loss', type=str, default='mse', help='mse or pearson')
parser.add_argument('--num_dec_blocks', type=int, default=4)
parser.add_argument('--vocab_size', type=int, default=4096)
parser.add_argument('--num_dec_heads', type=int, default=4)
parser.add_argument('--num_heads', type=int, default=4)
parser.add_argument('--dropout', type=float, default=0)

parser.add_argument('--num_iter', type=int, default=3)
parser.add_argument('--num_slots', type=int, default=2)
parser.add_argument('--slot_size', type=int, default=256)

parser.add_argument('--init_method', default='embedding', help='embedding, shared_gaussian')

parser.add_argument('--sigma_steps', type=int, default=30000)
parser.add_argument('--sigma_final', type=float, default=0)
parser.add_argument('--sigma_start', type=float, default=1)

parser.add_argument('--test',  default=False, action='store_true')


def main(args):
    print(args)
    set_random_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    args.monitor = monitors[args.evaluate]
    datamodule = datamodules[args.dataset](args)
    model = Dinosaur(args)
    if args.test:
        ckpt = state_dict_ckpt(args.test_ckpt_path)
        model.load_state_dict(ckpt)
    method = SlotAttentionMethod(model=model, datamodule=datamodule, args=args)
    method.hparams = args

    if args.is_logger_enabled:
        logger = pl_loggers.TensorBoardLogger(args.log_path, name=args.log_name) 
        arg_str_list = ['{}={}'.format(k, v) for k, v in vars(args).items()]
        arg_str = '__'.join(arg_str_list)
        log_dir = os.path.join(args.log_path, args.log_name)
        print(log_dir)
        logger.experiment.add_text('hparams', arg_str)
        callbacks = [LearningRateMonitor("step"), ImageLogCallback(), ModelCheckpoint(monitor=args.monitor, save_top_k=1, save_last=True, mode='max')]
    else:
        logger = False
        callbacks = []
    kwargs = {
        'resume_from_checkpoint': args.ckpt_path if args.load_from_ckpt else None,
        'logger': logger,
        'default_root_dir': args.log_path,
        'accelerator': "ddp" if args.gpus > 1 else None,
        'num_sanity_val_steps': args.num_sanity_val_steps,
        'gpus': args.gpus,
        'max_steps': args.max_steps,
        'max_epochs': args.max_epochs,
        'log_every_n_steps': 50,
        'callbacks': callbacks,
        'gradient_clip_val': args.grad_clip,
    }
    if args.check_val_every_n_step > 0:
        kwargs['val_check_interval'] = args.check_val_every_n_step
    if args.check_val_every_n_epoch > 0:
        kwargs['check_val_every_n_epoch'] = args.check_val_every_n_epoch

    trainer = Trainer(**kwargs)
    if args.test:
        trainer.test(method)
        # visualize
        from torchvision import transforms
        for i in range(1):
            method.eval()
            images = method.sample_images()
            images = transforms.ToPILImage()(images)
            images.save(f'{args.dataset}_test.png')
    else:
        trainer.fit(method)

if __name__ == "__main__":
    args = parser.parse_args()
    # args.dataset = 'voc'
    # args.evaluate = 'mbo'
    # args.gpus = 1 
    # args.device = '7' 
    # args.batch_size = 32 
    # args.num_slots = 6
    # args.init_method = 'text'
    # args.test_ckpt_path = '/home/yuliu/Projects/LG-SLOT/results/birds/test_BOQSA/version_1/checkpoints/epoch=499-step=156499.ckpt' 
    # args.dino_type = 'vits16' 
    # args.test = True
    paths = json.load(open('./path.json', 'r'))
    data_paths = paths['data_paths']
    args.log_path = paths['log_path']
    args.data_root = data_paths[args.dataset]
    args.log_path += args.dataset
    main(args)

# salloc -N 1 -t 30:00 --gres=gpu:1 --qos=debug --cpus-per-task=32 --job-name=debug --partition=gpu
# salloc -N 1 -t 24:00:00 --gres=gpu:1 --qos=gpu --cpus-per-task=32 --partition=gpu