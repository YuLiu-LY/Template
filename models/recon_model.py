import os
import sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

import torch.nn as nn
import torch.nn.functional as F
from models.slot_attn import SlotAttentionEncoder
from models.model import register, make_model


@register('recon')
class ReconModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.backbone = make_model(args.img_enc, args)
        img_emb_dim = self.backbone.emb_dim
        
        self.P = self.backbone.P

        self.slot_attn = SlotAttentionEncoder(
            args.num_iter, args.num_slots, 
            img_emb_dim, args.slot_size, 
            args.drop_path,
            args.use_feats_mlp, args.use_pe,
            args.init_method,
            hard_assign=args.hard_assign,
        )
        self.dec = args.decoder
        self.decoder = make_model(args.decoder, args)

    def forward(self, batch, sigma=0, tau=0.1, visualize=False, is_Train=False):
        """
        feats: [B, L+1, D], L patchs 1 cls token
        """
        out = {}
        loss = {}
        imgs = batch['image']
        B, _, H, W = imgs.shape
        feats = self.backbone.encode_img(imgs)
        slot_attn_out = self.slot_attn(feats[:, 1:], sigma=sigma)
        slots = slot_attn_out['slots']
        attns = slot_attn_out['attn']
        H1 = H // self.P
        W1 = W // self.P
        attns = attns.reshape(B, -1, H1, H1)
        if H1 != H or W1 != W:
            attns = F.interpolate(attns, size=(H, W), mode='bilinear', align_corners=False).unsqueeze(2)
        # decoding
        if self.dec == 'feat_dec':
            decoder_out = self.decoder(feats, slots)
        elif self.dec == 'img_dec':
            decoder_out = self.decoder(imgs, slots)
        else:
            target = batch['code']
            decoder_out = self.decoder(target, slots)
        loss['recon_loss'] = decoder_out['loss']
        out['attns'] = attns
        out['loss'] = loss
        
        if visualize:
            out['recon'] = imgs
            cross_attn = decoder_out['attn']
            cross_attn = cross_attn.reshape(B, -1, H1, H1)
            if H1 != H or W1 != W:
                cross_attn = F.interpolate(cross_attn, size=(H, W), mode='bilinear', align_corners=False).unsqueeze(2)
            out['loss_attns'] = cross_attn # [B, K, L]

        return out
    
    def inference(self, imgs):
        feats = self.backbone(imgs)
        out = self.forward(imgs, feats, visualize=True)
        return out

    
