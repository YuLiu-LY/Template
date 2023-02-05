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

from methods.utils import to_rgb_from_tensor, average_ari, iou_and_dice, average_segcover, mean_best_overlap
from methods.seg_metrics import Segmentation_Metrics_Calculator


class SlotAttentionMethod(pl.LightningModule):
    def __init__(self, model, datamodule: pl.LightningDataModule, args):
        super().__init__()
        self.model = model
        self.datamodule = datamodule
        self.args = args
        self.val_iter = iter(self.datamodule.val_dataloader())
        self.sample_num = 0
        self.empty_cache = True
        self.sigma = 0
        self.evaluator = args.evaluator
        self.init_method = args.init_method
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        if self.evaluator == "ap":
            self.seg_metric_logger = Segmentation_Metrics_Calculator(max_ins_num=self.args.num_slots)

    def forward(self, input, **kwargs):
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx):
        batch_img = batch['image']
        self.sigma = self.cosine_anneal(self.global_step, self.args.sigma_steps, start_value=self.args.sigma_start, final_value=self.args.sigma_final)
        if self.init_method == 'text':
            labels = batch['label']
            loss_dict = self.model.forward(batch_img, sigma=self.sigma, labels=labels)['loss']
        else:
            loss_dict = self.model.forward(batch_img, sigma=self.sigma)['loss']
        loss = 0
        logs = {'sigma': self.sigma}
        for k, v in loss_dict.items():
            loss += v
            logs[k] = v.item()
        self.log_dict(logs, sync_dist=True)
        return {'loss': loss}

    def sample_images(self):
        if self.sample_num % (len(self.val_iter) - 1) == 0:
            self.val_iter = iter(self.datamodule.val_dataloader())
        self.sample_num += 1

        batch = next(self.val_iter)
        ls = torch.arange(self.args.n_samples)
        batch_img = batch['image'][ls].to(self.device)
        # mask_gt = batch['mask'][ls]
        if self.init_method == 'text':
            labels = batch['label'][ls].to(self.device)
            out = self.model.forward(batch_img, sigma=0, labels=labels)
        else:
            out = self.model.forward(batch_img, sigma=0)
        m1 = out['attns']
        m2 = out['cross_attns']

        if self.args.use_rescale:
            batch_img = to_rgb_from_tensor(batch_img, self.mean.to(self.device), self.std.to(self.device))
        out = torch.cat(
                [
                    batch_img.unsqueeze(1),  # original images
                ],
                dim=1,
            ).cpu().clamp(0, 1)

        m_i1 = (m1 * batch_img.unsqueeze(1) + 1 - m1).cpu() # [B, K, C, H, W]
        m_i2 = (m2 * batch_img.unsqueeze(1) + 1 - m2).cpu() # [B, K, C, H, W]
        out = torch.cat([out, m_i1, m_i2], dim=1) # add masks
        # out = torch.cat([out, mask_gt.unsqueeze(1).expand(-1, -1, 3, -1, -1).cpu()], dim=1) # add gt masks

        # shape of out: [N, K, C, H, W]
        batch_size, C, H, W = batch_img.shape
        images = vutils.make_grid(
            out.reshape(out.shape[0] * out.shape[1], C, H, W), normalize=False, nrow=out.shape[1],
            padding=3, pad_value=0,
        )

        return images

    def validation_step(self, batch, batch_idx):
        if self.empty_cache:
            torch.cuda.empty_cache()
            self.empty_cache = False

        batch_img = batch['image']
        masks_gt = batch['mask'] # [B, 1, H, W]
        if self.init_method == 'text':
            labels = batch['label']
            out = self.model.forward(batch_img, sigma=self.sigma, labels=labels)
        else:
            out = self.model.forward(batch_img, sigma=self.sigma)
        loss_dict = out['loss']
        masks = out['attns'] 
        output = {}
        for k, v in loss_dict.items():
            output[k] = v

        if self.evaluator == 'ari':
            m = masks.detach().argmax(dim=1)
            ari, _ = average_ari(m, masks_gt)
            ari_fg, _ = average_ari(m, masks_gt, True)
            msc_fg, _ = average_segcover(masks_gt, m, True)
            output['ARI'] = ari.to(self.device)
            output['ARI_FG'] = ari_fg.to(self.device)
            output['MSC_FG'] = msc_fg.to(self.device)
        elif self.evaluator == 'iou':
            K = self.args.num_slots
            m = F.one_hot(masks.argmax(dim=1), K).permute(0, 4, 1, 2, 3) # (B, K, 1, H, W)
            iou, dice = iou_and_dice(m[:, 0], masks_gt)
            for i in range(1, K):
                iou1, dice1 = iou_and_dice(m[:, i], masks_gt)
                iou = torch.max(iou, iou1)
                dice = torch.max(dice, dice1)
            output['IoU'] = iou.mean()
            output['Dice'] = dice.mean()
        elif self.evaluator == "ap":
            masks = masks.detach().cpu()[:, :, 0, :, :]           # (B, K, 1, H, W) -> (B, K, H, W)
            pred_mask_conf, _ = masks.max(dim=1)
            confidence_mask = (pred_mask_conf > 0.5)
            pred_mask_oh = masks.argmax(dim=1)                    # (B, K, H, W) -> (B, H, W)
            gt_mask_oh = masks_gt[:, 0, :, :].detach().cpu()               # (B, 1, H, W) -> (B, H, W)                   
            gt_fg_batch = (gt_mask_oh != 0)
            self.seg_metric_logger.update_new_batch(
                pred_mask_batch=pred_mask_oh,
                gt_mask_batch=gt_mask_oh,
                valid_pred_batch=confidence_mask,
                gt_fg_batch=gt_fg_batch,
                pred_conf_mask_batch=pred_mask_conf
            )
        elif self.evaluator == 'mbo':
            K = self.args.num_slots
            m = F.one_hot(masks.argmax(dim=1), K).permute(0, 4, 1, 2, 3).squeeze(2) # (B, K, H, W)
            mBO, mOR = mean_best_overlap(m, masks_gt)
            output['mBO'] = mBO.mean()
            output['mOR'] = mOR.mean()
        return output

    def validation_epoch_end(self, outputs):
        self.empty_cache = True
        if self.evaluator != "ap":
            keys = outputs[0].keys()
            logs = {}
            for k in keys:
                v = torch.stack([x[k] for x in outputs]).mean()
                logs['avg_' + k] = v
            self.log_dict(logs, sync_dist=True)
            print("; ".join([f"{k}: {v.item():.6f}" for k, v in logs.items()]))
        else:
            outputs = self.seg_metric_logger.calculate_score_summary()
            keys = outputs.keys()
            logs = {}
            for k in keys:
                if k not in ['AP@05','PQ','F1','precision','recall']:
                    continue
                v = outputs[k]
                logs['avg_' + k] = v
            self.log_dict(logs, sync_dist=True)
            print("; ".join([f"{k}: {v:.6f}" for k, v in logs.items()]))
            self.seg_metric_logger.reset()

    def test_step(self, batch, batch_idx):
        if self.empty_cache:
            torch.cuda.empty_cache()
            self.empty_cache = False

        batch_img = batch['image']
        masks_gt = batch['mask']
        if self.init_method == 'text':
            labels = batch['label']
            out = self.model.forward(batch_img, sigma=0, labels=labels)
        else:
            out = self.model.forward(batch_img, sigma=0)
        loss_dict = out['loss']
        masks = out['attns'] 
        # masks = out['cross_attns'] 
        output = {}
        for k, v in loss_dict.items():
            output[k] = v

        if self.evaluator == 'ari':
            m = masks.detach().argmax(dim=1)
            ari, _ = average_ari(m, masks_gt)
            ari_fg, _ = average_ari(m, masks_gt, True)
            msc_fg, _ = average_segcover(masks_gt, m, True)
            output['ARI'] = ari.to(self.device)
            output['ARI_FG'] = ari_fg.to(self.device)
            output['MSC_FG'] = msc_fg.to(self.device)
        elif self.evaluator == 'iou':
            K = self.args.num_slots
            m = F.one_hot(masks.argmax(dim=1), K).permute(0, 4, 1, 2, 3)
            iou, dice = iou_and_dice(m[:, 0], masks_gt)
            for i in range(1, K):
                iou1, dice1 = iou_and_dice(m[:, i], masks_gt)
                iou = torch.max(iou, iou1)
                dice = torch.max(dice, dice1)
            output['IoU'] = iou.mean()
            output['Dice'] = dice.mean()
        elif self.evaluator == "ap":
            masks = masks.detach()[:, :, 0, :, :]           # (B, K, 1, H, W) -> (B, K, H, W)
            pred_mask_conf, _ = masks.max(dim=1)
            confidence_mask = (pred_mask_conf > 0.5)
            pred_mask_oh = masks.argmax(dim=1)                    # (B, K, H, W) -> (B, H, W)
            gt_mask_oh = masks_gt[:, 0, :, :].detach()     # (B, 1, H, W) -> (B, H, W)                   
            gt_fg_batch = (gt_mask_oh != 0)
            self.seg_metric_logger.update_new_batch(
                pred_mask_batch=pred_mask_oh,
                gt_mask_batch=gt_mask_oh,
                valid_pred_batch=confidence_mask,
                gt_fg_batch=gt_fg_batch,
                pred_conf_mask_batch=pred_mask_conf
            )
        elif self.evaluator == 'mbo':
            K = self.args.num_slots
            m = F.one_hot(masks.argmax(dim=1), K).permute(0, 4, 1, 2, 3).squeeze(2) # (B, K, H, W)
            mBO, mOR = mean_best_overlap(m, masks_gt)
            output['mBO'] = mBO.mean()
            output['mOR'] = mOR.mean()
        return output

    def test_epoch_end(self, outputs):
        self.empty_cache = True
        if self.evaluator != "ap":
            keys = outputs[0].keys()
            logs = {}
            for k in keys:
                v = torch.stack([x[k] for x in outputs]).mean()
                logs['avg_' + k] = v
            self.log_dict(logs, sync_dist=True)
            print("; ".join([f"{k}: {v.item():.6f}" for k, v in logs.items()]))
        else:
            outputs = self.seg_metric_logger.calculate_score_summary()
            keys = outputs.keys()
            logs = {}
            for k in keys:
                if k not in ['AP@05','PQ','F1','precision','recall']:
                    continue
                v = outputs[k]
                logs['avg_' + k] = v
            self.log_dict(logs, sync_dist=True)
            print("outcomes:")
            for k in ['AP@05','PQ','F1','precision','recall']:
                v = logs['avg_' + k] * 100
                print(f'{v:.2f}')
            self.seg_metric_logger.reset()


    def configure_optimizers(self):
        params = [
        {'params': (x[1] for x in self.model.named_parameters() if 'backbone' not in x[0] and 'clip' not in x[0]), 'lr': self.args.lr},
        ]
        optimizer = optim.Adam(params)
        
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
