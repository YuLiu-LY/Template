import torch
import wandb
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import Trainer
from pytorch_lightning import Callback
from sklearn.metrics import adjusted_rand_score


def build_grid(resolution):
    ranges = [torch.linspace(0.0, 1.0, steps=res) for res in resolution]
    grid = torch.meshgrid(*ranges)
    grid = torch.stack(grid, dim=-1)
    grid = torch.reshape(grid, [resolution[0], resolution[1], -1])
    grid = grid.unsqueeze(0)
    return torch.cat([grid, 1.0 - grid], dim=-1)


class PositionEmbed(nn.Module):
    def __init__(self, hidden_size: int, resolution):
        super().__init__()
        self.dense = nn.Linear(4, out_features=hidden_size)
        self.register_buffer("grid", build_grid(resolution))

    def forward(self, inputs):
        emb_proj = self.dense(self.grid).permute(0, 3, 1, 2)

        return inputs + emb_proj


class ImageLogCallback(Callback):
    def on_validation_epoch_end(self, trainer: Trainer, pl_module):
        """Called when the test epoch ends."""
        if trainer.logger:
            with torch.no_grad():
                pl_module.eval()
                out = pl_module.visualize()
                self.log_img(out, trainer)
                
    def on_test_epoch_end(self, trainer: Trainer, pl_module):
        """Called when the test epoch ends."""

        if trainer.logger:
            with torch.no_grad():
                pl_module.eval()
                out = pl_module.visualize()
                self.log_img(out, trainer)

    def log_img(self, out, trainer):
        img = out['img']
        recon = out['recon']
        mask = out['mask']
        # mask_gt = out['mask_gt']
        table = wandb.Table(columns=[
            "img", 
            "recon", 
            "mask", 
            # "mask_gt",
        ])
        for i in range(len(img)):
            table.add_data(wandb.Image(img[i]), 
                wandb.Image(recon[i]), 
                wandb.Image(img[i], masks={
                    "prediction" : {
                        "mask_data" : mask[i],
                        # "class_labels" : class_labels
                        },
                    }
                ), 
                # wandb.Image(img[i], masks={
                #     "prediction" : {
                #         "mask_data" : mask_gt[i],
                #         # "class_labels" : class_labels
                #         },
                #     }
                # ), 
            )
        trainer.logger.experiment.log({"Table": table})


def to_rgb_from_tensor(x, mean=[0, 0, 0], std=[1, 1, 1]):
    return x * std + mean
    

def state_dict_ckpt(path, device='cpu'):
    if device == 'cpu':
        ckpt = torch.load(path, map_location='cpu')
    else:
        ckpt = torch.load(path)    
    model_state_dict = ckpt["state_dict"]
    dict = model_state_dict.copy()
    for s in dict:
        x = s[6:]
        model_state_dict[x] = model_state_dict.pop(s)
    return model_state_dict


def iou_and_dice(mask, mask_gt):
    B, _, H, W = mask.shape

    pred = mask > 0.5
    gt = mask_gt > 0.5
    eps = 1e-8 # to prevent NaN
    insertion = (pred * gt).view(B, -1).sum(dim=-1)
    union = ((pred + gt) > 0).view(B, -1).sum(dim=-1) + eps
    pred_plus_gt = pred.view(B, -1).sum(dim=-1) + gt.view(B, -1).sum(dim=-1) + eps
    
    iou =  (insertion + eps) / union
    dice = (2 * insertion + eps) / pred_plus_gt
        
    return iou, dice


def iou(mask, mask_gt):
    eps = 1e-8 # to prevent NaN
    insertion = (mask * mask_gt).view(-1).sum()
    union = ((mask + mask_gt) > 0).view(-1).sum() 
    iou =  (insertion + eps) / (union + eps)
    return iou


def imask2bmask(imasks, ignore_index=None):
    """Convert index mask to binary mask.
    Args:
        imask: index mask, shape (B, 1, H, W)
    """
    B, _, H, W = imasks.shape
    bmasks = []
    for i in range(B):
        imask = imasks[i] # (1, H, W)
        classes = imask.unique().tolist()
        if ignore_index in classes:
            classes.remove(ignore_index)
        bmask = [imask == c for c in classes]
        bmask = torch.cat(bmask, dim=0) # (K, H, W)
        bmasks.append(bmask)
    # can't use torch.stack because of different K
    return bmasks # a list of (K, H, W), len = B
        

def mean_best_overlap(masks, masks_gt, fg_only=True):
    """Compute the best overlap between predicted and ground truth masks.
    Args:
        masks: predicted masks, shape (B, K, H, W), binary
        masks_gt: ground truth masks, shape (B, 1, H, W), index
    """
    B, _, _, _ = masks.shape
    ignore_index = None
    if fg_only:
        ignore_index = 0
    bmasks_gt = imask2bmask(masks_gt, ignore_index=ignore_index)
    mean_best_overlap = []
    mOR = []
    for i in range(B):
        mask = masks[i].unsqueeze(0) > 0.5 # (1, K, H, W)
        mask_gt = bmasks_gt[i].unsqueeze(1) > 0.5 # (K_gt, 1, H, W)
        # Compute IOU
        eps = 1e-8
        intersection = (mask * mask_gt).sum((2, 3))
        union = (mask + mask_gt).sum((2, 3))
        iou = (intersection + eps) / (union + eps) # (K_gt, K)
        # Compute best overlap
        best_overlap, _ = torch.max(iou, dim=1)
        # Compute mean best overlap
        mean_best_overlap.append(best_overlap.mean())
        mOR.append((best_overlap > 0.5).float().mean())
    return torch.stack(mean_best_overlap), torch.stack(mOR)


def iou_binary(mask_A, mask_B, debug=False):
    if debug:
        assert mask_A.shape == mask_B.shape
        assert mask_A.dtype == torch.bool
        assert mask_B.dtype == torch.bool
    intersection = (mask_A * mask_B).sum((1, 2, 3))
    union = (mask_A + mask_B).sum((1, 2, 3))
    # Return -100 if union is zero, else return IOU
    return torch.where(union == 0, torch.tensor(-100.0).to(mask_A.device),
                       intersection.float() / union.float())


def average_ari(masks, masks_gt, foreground_only=False):
    ari = []
    assert masks.shape[0] == masks_gt.shape[0], f"The number of masks is not equal to the number of masks_gt"
    # Loop over elements in batch
    for i in range(masks.shape[0]):
        m = masks[i].cpu().numpy().flatten()
        m_gt = masks_gt[i].cpu().numpy().flatten()
        if foreground_only:
            m = m[np.where(m_gt > 0)]
            m_gt = m_gt[np.where(m_gt > 0)]
        score = adjusted_rand_score(m, m_gt)
        ari.append(score)
    return torch.Tensor(ari).mean(), ari


def average_segcover(segA, segB, ignore_background=False):
    """
    Covering of segA by segB
    segA.shape = [batch size, 1, img_dim1, img_dim2]  ground truth
    segB.shape = [batch size, 1, img_dim1, img_dim2]

    scale: If true, take weighted mean over IOU values proportional to the
           the number of pixels of the mask being covered.

    Assumes labels in segA and segB are non-negative integers.
    Negative labels will be ignored.
    """

    assert segA.shape == segB.shape, f"{segA.shape} - {segB.shape}"
    assert segA.shape[1] == 1 and segB.shape[1] == 1
    bsz = segA.shape[0]
    nonignore = (segA >= 0)

    mean_scores = torch.tensor(bsz*[0.0]).to(segA.device)
    N = torch.tensor(bsz*[0]).to(segA.device)
    scaled_scores = torch.tensor(bsz*[0.0]).to(segA.device)
    scaling_sum = torch.tensor(bsz*[0]).to(segA.device)

    # Find unique label indices to iterate over
    if ignore_background:
        iter_segA = torch.unique(segA[segA > 0]).tolist()
    else:
        iter_segA = torch.unique(segA[segA >= 0]).tolist()
    iter_segB = torch.unique(segB[segB >= 0]).tolist()
    # Loop over segA
    for i in iter_segA:
        binaryA = segA == i
        if not binaryA.any():
            continue
        max_iou = torch.tensor(bsz*[0.0]).to(segA.device)
        # Loop over segB to find max IOU
        for j in iter_segB:
            # Do not penalise pixels that are in ignore regions
            binaryB = (segB == j) * nonignore
            if not binaryB.any():
                continue
            iou = iou_binary(binaryA, binaryB)
            max_iou = torch.where(iou > max_iou, iou, max_iou)
        # Accumulate scores
        mean_scores += max_iou
        N = torch.where(binaryA.sum((1, 2, 3)) > 0, N+1, N)
        scaled_scores += binaryA.sum((1, 2, 3)).float() * max_iou
        scaling_sum += binaryA.sum((1, 2, 3))

    # Compute coverage
    mean_sc = mean_scores / torch.max(N, torch.tensor(1)).float()
    scaled_sc = scaled_scores / torch.max(scaling_sum, torch.tensor(1)).float()

    # Return mean over batch dimension 
    return mean_sc.mean(0), scaled_sc.mean(0)


def gumbel_max(logits, dim=-1):
    
    eps = torch.finfo(logits.dtype).tiny
    
    gumbels = -(torch.empty_like(logits).exponential_() + eps).log()
    gumbels = logits + gumbels
    
    return gumbels.argmax(dim)


def gumbel_softmax(logits, tau=1., hard=False, dim=-1):
    
    eps = torch.finfo(logits.dtype).tiny
    
    gumbels = -(torch.empty_like(logits).exponential_() + eps).log()
    gumbels = (logits + gumbels) / tau
    
    y_soft = F.softmax(gumbels, dim)
    
    if hard:
        index = y_soft.argmax(dim, keepdim=True)
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.)
        return y_hard - y_soft.detach() + y_soft
    else:
        return y_soft


def conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0,
           dilation=1, groups=1, bias=True, padding_mode='zeros',
           weight_init='xavier'):
    
    m = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                  dilation, groups, bias, padding_mode)
    
    if weight_init == 'kaiming':
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
    else:
        nn.init.xavier_uniform_(m.weight)
    
    if bias:
        nn.init.zeros_(m.bias)
    
    return m


class Conv2dBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, use_norm=True):
        super().__init__()
        self.use_norm = use_norm
        self.m = conv2d(in_channels, out_channels, kernel_size, stride, padding,
                        bias=False, weight_init='kaiming')
        if self.use_norm:
            self.weight = nn.Parameter(torch.ones(out_channels))
            self.bias = nn.Parameter(torch.zeros(out_channels))
    
    
    def forward(self, x):
        x = self.m(x)
        if self.use_norm:
            return F.relu(F.group_norm(x, 1, self.weight, self.bias))
        else:
            return F.relu(x)



def linear(in_features, out_features, bias=True, weight_init='xavier', gain=1.):
    
    m = nn.Linear(in_features, out_features, bias)
    
    if weight_init == 'kaiming':
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
    else:
        nn.init.xavier_uniform_(m.weight, gain)
    
    if bias:
        nn.init.zeros_(m.bias)
    
    return m


def gru_cell(input_size, hidden_size, bias=True):
    
    m = nn.GRUCell(input_size, hidden_size, bias)
    
    nn.init.xavier_uniform_(m.weight_ih)
    nn.init.orthogonal_(m.weight_hh)
    
    if bias:
        nn.init.zeros_(m.bias_ih)
        nn.init.zeros_(m.bias_hh)
    
    return m


class Evaluator(object):
    
    def __init__(self, evaluator):
        
        self.evaluator = evaluator
        self.reset()
    
    
    def reset(self):
        
        self.loss = 0.
        self.acc = 0.
        self.n = 0
    
    
    def evaluate(self):
        
        self.reset()
        
        self.model.eval()
        
        with torch.no_grad():
            for batch in self.dataloader:
                batch = tuple(t.to(self.device) for t in batch)
                loss, acc = self.model(*batch)
                self.loss += loss.item()
                self.acc += acc.item()
                self.n += 1
        
        self.loss /= self.n
        self.acc /= self.n
        
        return self.loss, self.acc