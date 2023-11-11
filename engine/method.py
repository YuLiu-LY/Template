import wandb
import torch
from tabulate import tabulate
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from torchvision.utils import make_grid
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

import os
import sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

from model import make_model
from dataset import make_dataset


class Method(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.model = make_model(cfg.model)
        self.train_set, self.val_set, self.test_set = make_dataset(cfg.dataset)
        self.cfg = cfg.training

        self.vis_stride = len(self.val_set) // self.cfg.num_vis

    def training_step(self, batch, batch_idx):
        batch_img = batch['image']
        recon = self.model(batch_img)
        loss = F.mse_loss(recon, batch_img, reduction='mean')
        self.log("train/loss", loss, on_step=True, on_epoch=False, 
                 prog_bar=True, logger=True, sync_dist=True)
        return loss

    @rank_zero_only
    def visualize(self, out):
        r'''
        out: a dict containing the following keys:
            img: [B, 3, H, W]
            recon: [B, 3, H, W]
            mask: [B, 1, H, W]
            mask_gt: [B, 1, H, W]
        '''
        img = out['img']
        B, _, H, W  = img.shape
        recon = out['recon']
        stack = torch.stack([img, recon], dim=1).reshape(-1, *img.shape[1:])
        grid = make_grid(stack, nrow=2, padding=3, pad_value=0)
        if self.logger is not None:
            self.logger.log_image(key=f"Recon", images=[grid])
        
            # mask = torch.randint(0, 2, size=[B, H, W]).numpy()
            # mask_gt = torch.randint(0, 2, size=[B, H, W]).numpy()
            # table = wandb.Table(columns=["img", "recon", "seg", 
            #                          ])
            # for i in range(len(img)):
            #     table.add_data(
            #         wandb.Image(img[i]), 
            #         wandb.Image(recon[i]), 
            #         wandb.Image(img[i], masks={
            #                 "prediction" : {
            #                     "mask_data" : mask[i],
            #                     # "class_labels" : class_labels
            #                     },
            #                 "ground_truth" : {
            #                     "mask_data" : mask_gt[i],
            #                     # "class_labels" : class_labels
            #                 }
            #             })
            #     )
            # self.logger.experiment.log({f"Seg": table})

    def evaluate(self, batch):
        batch_img = batch['image']
        recon = self.model(batch_img)
        loss = F.mse_loss(recon, batch_img, reduction='mean')
        psnr = 10 * torch.log10(1 / loss)
        self.metrics['psnr'].append(psnr)
        self.outputs['img'].append(batch_img)
        self.outputs['recon'].append(recon)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch)

    def on_validation_epoch_end(self):
        metrics = self.metrics
        keys = metrics.keys()
        logs = {}
        for k in keys:
            v = torch.stack(metrics[k]).mean()
            logs['val/' + k] = v
        self.log_dict(logs, sync_dist=True)
        table = [keys, ]
        table.append(tuple([logs['val/' + key] for key in table[0]]))
        print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))
        outputs = {}
        for k, v in self.outputs.items():
            outputs[k] = torch.cat(v)[::self.vis_stride]
        # self.visualize(outputs)

    def test_step(self, batch, batch_idx):
        self.evaluate(batch)

    def on_test_epoch_end(self):
        metrics = self.metrics
        keys = metrics.keys()
        logs = {}
        for k in keys:
            v = torch.stack(metrics[k]).mean()
            logs['test/' + k] = v
        self.log_dict(logs, sync_dist=True)
        table = [keys, ]
        table.append(tuple([logs['test/' + key] for key in table[0]]))
        print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))
        outputs = {}
        for k, v in self.outputs.items():
            outputs[k] = torch.cat(v[self.visualize_indices])
        self.visualize(outputs)

    def configure_optimizers(self):
        warmup_steps = self.cfg.warmup_steps
        decay_steps = self.cfg.decay_steps
        min_lr_factor = self.cfg.min_lr_factor
        def lr_warmup_exp_decay(step: int):
            factor = min(1, step / (warmup_steps + 1e-6))
            decay_factor = 0.5 ** (step / decay_steps)
            return factor * decay_factor * (1 - min_lr_factor) + min_lr_factor
        def lr_exp_decay(step: int):
            decay_factor = 0.5 ** (step / decay_steps)
            return decay_factor * (1 - min_lr_factor) + min_lr_factor
        
        params_enc = []
        params_dec = []
        params_other = []
        for name, p in self.model.named_parameters():
            if 'encoder' in name:
                params_enc.append(p)
            elif 'decoder' in name:
                params_dec.append(p)
            else:
                params_other.append(p)
        
        params = [
            {'params': params_other, 'lr': self.cfg.lr, 'weight_decay': self.cfg.weight_decay},
            ]
        lr_lambda_list = [lr_exp_decay]
        if not self.cfg.freeze_enc:
            params.append({'params': params_enc, 'lr': self.cfg.lr * 0.1, 'weight_decay': self.cfg.weight_decay})
            lr_lambda_list.append(lr_warmup_exp_decay)
        if not self.cfg.freeze_dec:
            params.append({'params': params_dec, 'lr': self.cfg.lr * 0.1, 'weight_decay': self.cfg.weight_decay})
            lr_lambda_list.append(lr_warmup_exp_decay)

        opt = optim.AdamW(params)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer=opt, lr_lambda=lr_lambda_list)
        return (
            [opt],
            [{"scheduler": scheduler, "interval": "step",}],
        )
    
    def on_train_epoch_start(self):
        torch.cuda.empty_cache()
        
    def on_validation_epoch_start(self):
        torch.cuda.empty_cache()
        self.metrics = {
            'psnr': []
        }
        self.outputs = {
            'img': [],
            'recon': [],
        }

    def on_test_epoch_start(self):
        torch.cuda.empty_cache()
        self.metrics = {
            'psnr': []
        }
        self.outputs = {
            'img': [],
            'recon': [],
        }

    def train_dataloader(self):
        shuffle = False if self.cfg.job_type == 'debug' else True
        return DataLoader(self.train_set, self.cfg.batch_size, shuffle=shuffle, pin_memory=True, num_workers=self.cfg.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, self.cfg.batch_size, shuffle=False, pin_memory=True, num_workers=self.cfg.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_set, self.cfg.batch_size, shuffle=False, pin_memory=True, num_workers=self.cfg.num_workers)
