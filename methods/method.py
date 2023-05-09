import math
import torch
import pytorch_lightning as pl
from torch import optim
from torchvision import utils as vutils
from torch.nn import functional as F

import os
import sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

from methods.utils import to_rgb_from_tensor, Evaluator
from methods.seg_metrics import Segmentation_Metrics_Calculator
from models.model import make_model


class Method(pl.LightningModule):
    def __init__(self, model, datamodule: pl.LightningDataModule, args):
        super().__init__()
        self.model = model
        self.datamodule = datamodule
        self.args = args
        self.val_iter = iter(self.datamodule.val_dataloader())
        self.sample_num = 0
        self.empty_cache = True
        self.evaluator = Evaluator(args.evaluator) if args.evaluator != None else None
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def forward(self, input, **kwargs):
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx):
        batch_img = batch['image']
        text_token = batch.get('text_token', None)
        out = self.model.forward(batch_img, text_token=text_token)
        loss_dict = out['loss']
        loss = 0
        logs = {}
        for k, v in loss_dict.items():
            loss += v
            logs[k] = v.item()
        self.log_dict(logs, sync_dist=True)
        return {'loss': loss}

    def visualize(self):
        if self.sample_num % (len(self.val_iter) - 1) == 0:
            self.val_iter = iter(self.datamodule.val_dataloader())
        self.sample_num += 1

        batch = next(self.val_iter)
        batch_img = batch['image'][:self.args.n_samples].to(self.device)
        text_token = batch.get('text_token', None)
        if text_token is not None:
            text_token = text_token[:self.args.n_samples].to(self.device)
        out = self.model.forward(batch_img, text_token=text_token, visualize=True)
        recon = out['recon']
        if self.args.img_normalize:
            batch_img = to_rgb_from_tensor(batch_img, self.mean.to(self.device), self.std.to(self.device))
            recon = to_rgb_from_tensor(recon, self.mean.to(self.device), self.std.to(self.device))
        batch_img = batch_img.cpu().clamp(0, 1)
        recon = recon.cpu().clamp(0, 1)
        mask = out['masks'].argmax(dim=1).squeeze().cpu().numpy()     
        return {
            'img': batch_img,
            'recon': recon,
            'mask': mask,
            # 'mask_gt': batch['mask'][:self.args.n_samples].squeeze().cpu().numpy(),
        }

    def validation_step(self, batch, batch_idx):
        if self.empty_cache:
            torch.cuda.empty_cache()
            self.empty_cache = False

        batch_img = batch['image']
        text_token = batch.get('text_token', None)
        out = self.model.forward(batch_img, text_token=text_token)
        loss_dict = out['loss']
        masks = out['masks'] 
        output = {}
        for k, v in loss_dict.items():
            output[k] = v
        if self.evaluator != None:
            masks_gt = batch['mask'] # [B, 1, H, W] 
            metrics = self.evaluator(masks, masks_gt)
            output.update(metrics)
        return output

    def validation_epoch_end(self, outputs):
        self.empty_cache = True
        keys = outputs[0].keys()
        logs = {}
        for k in keys:
            v = torch.stack([x[k] for x in outputs]).mean()
            logs['val_' + k] = v
        self.log_dict(logs, sync_dist=True)
        print("; ".join([f"{k}: {v.item():.6f}" for k, v in logs.items()]))
        

    def test_step(self, batch, batch_idx):
        if self.empty_cache:
            torch.cuda.empty_cache()
            self.empty_cache = False

        batch_img = batch['image']
        text_token = batch.get('text_token', None)
        out = self.model.forward(batch_img, text_token=text_token)
        loss_dict = out['loss']
        masks = out['masks'] 
        output = {}
        for k, v in loss_dict.items():
            output[k] = v
        if self.evaluator != None:
            masks_gt = batch['mask'] # [B, 1, H, W] 
            metrics = self.evaluator(masks, masks_gt)
            output.update(metrics)
        return output

    def test_epoch_end(self, outputs):
        self.empty_cache = True
        keys = outputs[0].keys()
        logs = {}
        for k in keys:
            v = torch.stack([x[k] for x in outputs]).mean()
            logs['test_' + k] = v
        self.log_dict(logs, sync_dist=True)
        print("; ".join([f"{k}: {v.item():.6f}" for k, v in logs.items()]))

    def configure_optimizers(self):
        params_other = []
        params_img_enc = []
        params_text_enc = []
        for name, p in self.model.named_parameters():
            if 'backbone' in name:
                params_img_enc.append(p)
            elif 'text_enc' in name:
                params_text_enc.append(p)
            else:
                params_other.append(p)
        
        params = [
            {'params': params_other, 'lr': self.args.lr},
            ]
        if not self.args.freeze_img_enc:
            params.append({'params': params_img_enc, 'lr': self.args.lr * 0.001})
        if not self.args.freeze_text_enc:
            params.append({'params': params_text_enc, 'lr': self.args.lr * 0.001})

        optimizer = optim.AdamW(params, weight_decay=self.args.weight_decay)
        
        warmup_steps = self.args.warmup_steps
        decay_steps = self.args.decay_steps

        def lr_scheduler_main(step: int):
            if step < warmup_steps:
                factor = step / warmup_steps
            else:
                factor = 1
            factor *= 0.5 ** (step / decay_steps)
            return factor

        scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_scheduler_main)
        return (
            [optimizer],
            [{"scheduler": scheduler, "interval": "step",}],
        )

    def cosine_anneal(self, step, final_step, start_step=0, start_value=1.0, final_value=0.1):
    
        if start_value <= final_value or start_step >= final_step:
            return final_value
        
        if step < start_step:
            value = start_value
        elif step >= final_step:
            value = final_value
        else:
            a = 0.5 * (start_value - final_value)
            b = 0.5 * (start_value + final_value)
            progress = (step - start_step) / (final_step - start_step)
            value = a * math.cos(math.pi * progress) + b
        return value
