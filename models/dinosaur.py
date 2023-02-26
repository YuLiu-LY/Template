import os
import sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

import torch.nn.functional as F
import clip
from methods.utils import *
from models.slot_attn import SlotAttentionEncoder
from models.transformer import TransformerDecoder
from models.dino_hubconf import dino_models
from models.models import register


CLASSES = ['background', \
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', \
    'bus', 'car', 'cat', 'chair', 'cow', \
    'diningtable', 'dog', 'horse', 'motorbike', 'person', \
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


class DINOEncoder(nn.Module):
    def __init__(self, dino_type='vits8'):
        super().__init__()
        self.dino = dino_models[dino_type](pretrained=True)
        self.dino_type = dino_type
        self.emb_dim = self.dino.embed_dim
        self.freeze()

    def forward(self, x):
        x = self.dino.prepare_tokens(x)
        for blk in self.dino.blocks:
            x = blk(x)
        x = self.dino.norm(x)
        return x[:, 1:]
    
    def freeze(self):
        for param in self.dino.parameters():
            param.requires_grad = False


class TextEncoder(nn.Module):
    def __init__(self, slot_size):
        super().__init__()
        self.clip = clip.load('ViT-B/16')[0]
        emb_dim = self.clip.text_projection.shape[1]
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.GELU(),
            nn.Linear(4 * emb_dim, slot_size)
        )
        self.freeze()
        self.embs = self.get_text_embs(self.clip.text_projection.device).float()

    def forward(self, labels):
        # labels: [B, K]
        embs = self.mlp(self.embs.clone().to(labels.device)) # [C + 1, slot_size]
        # add NULL token at last
        # embs = torch.cat([embs, torch.zeros(1, embs.shape[1]).to(embs.device)], dim=0)
        return embs[labels] # [B, K, slot_size]

    def freeze(self):
        for param in self.clip.parameters():
            param.requires_grad = False
        
    def get_text_embs(self, device):
        text = [CLASSES[0]]
        text += [f'a photo of {CLASSES[l]}' for l in range(1, len(CLASSES))]
        text += ['null']
        tokens = clip.tokenize(text)
        embs = self.clip.encode_text(tokens.to(device))
        return embs


class DinosaurDecoder(nn.Module):
    def __init__(self, args, d_model):
        super().__init__()
        N_tokens = (args.resolution[0] // 4) * (args.resolution[1] // 4)
        self.bos = nn.Parameter(torch.Tensor(1, 1, d_model))
        nn.init.xavier_uniform_(self.bos)
        self.tf_dec = TransformerDecoder(
            args.num_dec_blocks, N_tokens, d_model, args.num_dec_heads, args.dropout)


@register('dinosaur')
class Dinosaur(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.resolution = args.resolution
        self.backbone = DINOEncoder('vits8')
        d_model =  self.backbone.emb_dim
        if args.init_method == 'text':
            self.text_enc = TextEncoder(args.slot_size)
        
        enc_stride = int("".join(list(filter(str.isdigit, 'vits8'))))
        feature_resolution = args.resolution[0] // enc_stride, args.resolution[1] // enc_stride
        self.H = feature_resolution[0]
        self.W = feature_resolution[1]

        self.slot_attn = SlotAttentionEncoder(
            args.num_iter, args.num_slots, 
            d_model, args.slot_size, 
            args.truncate, 
            args.init_method, 0)
        
        self.slot_proj = linear(args.slot_size, d_model, bias=False)
        self.dinosaur_dec = DinosaurDecoder(args, d_model)

        self.loss = 'mse'
        if self.loss == 'mse':
            self.criterion = nn.MSELoss()
        elif self.loss == 'pearson':
            self.criterion = PearsonLoss()


    def forward(self, image, sigma=0, text_token=None, tau=0.1, labels=None):
        """
        image: batch_size x img_channels x H x W
        """
        out = {}
        loss = {}

        B, C, H, W = image.size()

        # apply slot attention
        features = self.backbone(image) #[B, N, D]
        if labels != None:
            slots_init = self.text_enc(labels)
            slot_attn_out = self.slot_attn(features, sigma=sigma, slots_init=slots_init, labels=labels)
        else:
            slot_attn_out = self.slot_attn(features, sigma=sigma)
        slots = slot_attn_out['slots']
        attns = slot_attn_out['attn']
        
        # add BOS token
        z_emb = torch.cat([self.dinosaur_dec.bos.expand(B, -1, -1), features[:, :-1]], dim=1) # B, 1 + H_z*W_z, d_model

        slots = self.slot_proj(slots) # [B, num_slots * num_block, d_model]
        y, cross_attn = self.dinosaur_dec.tf_dec(z_emb, slots)

        attns = attns.reshape(B, -1, self.H, self.W)
        cross_attn = cross_attn.reshape(B, -1, self.H, self.W)

        if self.H != H or self.W != W:
            attns = F.interpolate(attns, size=(H, W), mode='bilinear', align_corners=False).unsqueeze(2)
            cross_attn = F.interpolate(cross_attn, size=(H, W), mode='bilinear', align_corners=False).unsqueeze(2)
        
        loss[self.loss] = self.criterion(y, features)
        out['attns'] = attns
        out['cross_attns'] = cross_attn
        out['loss'] = loss
        return out
    

class PearsonLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, z, y):
        """
        z: [B, N, D]
        y: [B, N, D]
        """
        z = z - z.mean(dim=-1, keepdim=True)
        y = y - y.mean(dim=-1, keepdim=True)
        loss = 1 - F.cosine_similarity(z, y, dim=-1).mean()
        return loss

