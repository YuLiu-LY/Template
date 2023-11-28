import torch
import wandb
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
        self.num_vis = self.cfg.num_vis
        
        self.loss = torch.nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        batch_img = batch['image']
        label = batch['label'] # [B]
        logits = self.model(batch_img) # [B, C]
        loss = self.loss(logits, label)
        self.log("train/loss", loss, on_step=True, on_epoch=False, 
                 prog_bar=True, logger=True, sync_dist=True)
        return loss

    @rank_zero_only
    def visualize(self, out):
        r'''
        out: a dict containing the following keys:
            img: [B, 3, H, W]
            pred: [B, C] (probabilities)
        '''
        img = out['img']
        pred = out['pred']
        
        if self.logger is not None:
            grid = make_grid(img, nrow=1, normalize=True, scale_each=True)
            self.logger.experiment.log({"Image": wandb.Image(grid)})
            for i in range(len(img)):
                prob_table = wandb.Table(columns=["Class", "Prob"], data=[[f"{i}", f"{p.item():.2f}"] for i, p in enumerate(pred[i])])
                self.logger.experiment.log({f"Image/{i:04d}": wandb.Image(img[i]),
                                            f"Prediction/{i:04d}": wandb.plot.bar(prob_table, "Class", "Prob", title=f"Prediction/{i:04d}"),
                                            })
        
    def evaluate(self, batch):
        batch_img = batch['image']
        logits = self.model(batch_img) # [B, C]
        loss = self.loss(logits, batch['label'])
        pred = torch.argmax(logits, dim=1)
        acc = (pred == batch['label']).float().mean()
        self.metrics['acc'].append(acc)
        self.metrics['loss'].append(loss)
        self.outputs['img'].append(batch_img)
        self.outputs['pred'].append(F.softmax(logits, dim=-1))

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
            outputs[k] = torch.cat(v)[::self.vis_stride][:self.num_vis]
        self.visualize(outputs)

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
            outputs[k] = torch.cat(v)[::self.vis_stride][:self.num_vis]
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
        
        # separate params
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
            'loss': [],
            'acc': [],
        }
        self.outputs = {
            'img': [],
            'pred': [],
        }

    def on_test_epoch_start(self):
        torch.cuda.empty_cache()
        self.metrics = {
            'loss': [],
            'acc': [],
        }
        self.outputs = {
            'img': [],
            'pred': [],
        }

    def train_dataloader(self):
        shuffle = False if self.cfg.job_type == 'debug' else True
        return DataLoader(self.train_set, self.cfg.batch_size, shuffle=shuffle, pin_memory=True, num_workers=self.cfg.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, self.cfg.batch_size, shuffle=False, pin_memory=True, num_workers=self.cfg.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_set, self.cfg.batch_size, shuffle=False, pin_memory=True, num_workers=self.cfg.num_workers)
