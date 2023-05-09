import os
import sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

from methods.utils import *
from models.transformer import TransformerEncoderBlock, CrossAttnBlock, Attention, Block
from timm.models.layers import DropPath


class SlotAttention(nn.Module):
    def __init__(
        self,
        feature_size,
        slot_size, 
        epsilon=1e-8,
        drop_path=0.2,
        hard_assign=False,
        init_method='boqsa',
    ):
        super().__init__()
        self.slot_size = slot_size 
        self.epsilon = epsilon
        self.hard_assign = hard_assign
        self.init_method = init_method

        self.norm_feature = nn.LayerNorm(feature_size)
        self.norm_mlp = nn.LayerNorm(slot_size)
        self.norm_slots = nn.LayerNorm(slot_size)
        
        self.project_k = linear(feature_size, slot_size, bias=False)
        self.project_v = linear(feature_size, slot_size, bias=False)
        self.project_q = linear(slot_size, slot_size, bias=False)

        self.gru = gru_cell(slot_size, slot_size)

        self.mlp = nn.Sequential(
            linear(slot_size, slot_size * 4, weight_init='kaiming'),
            nn.ReLU(),
            linear(slot_size * 4, slot_size),
        )

        # self.attn = Attention(slot_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, features, slots_init, num_iter=3):
        # `feature` has shape [batch_size, num_feature, inputs_size].
        features = self.norm_feature(features)  
        k = self.project_k(features)  # Shape: [B, num_features, slot_size]
        v = self.project_v(features)  # Shape: [B, num_features, slot_size]

        slots = slots_init
        # Multiple rounds of attention.
        for i in range(num_iter - 1):
            slots, attn = self.iter(slots, k, v)
        if self.init_method == 'boqsa':
            slots = slots.detach() + slots_init - slots_init.detach()
        slots, attn = self.iter(slots, k, v, hard_assign=self.hard_assign)
        return slots, attn

    def iter(self, slots, k, v, hard_assign=False):
        B, K, D = slots.shape
        slots_prev = slots
        # slots = slots + self.drop_path(self.attn(slots))
        slots = self.norm_slots(slots)
        q = self.project_q(slots)
        # Attention
        scale = D ** -0.5
        attn_logits= torch.einsum('bid,bjd->bij', q, k) * scale # [B, K, N]
        if hard_assign:
            attn = gumbel_softmax(attn_logits, dim=1, tau=1, hard=True)
        else:
            attn = F.softmax(attn_logits, dim=1)

        # Weighted mean
        attn_sum = torch.sum(attn, dim=-1, keepdim=True) + self.epsilon
        attn_wm = attn / attn_sum 
        updates = torch.einsum('bij, bjd->bid', attn_wm, v)            

        # Update slots
        slots = self.gru(
            updates.reshape(-1, D),
            slots_prev.reshape(-1, D)
        )
        slots = slots.reshape(B, -1, D)
        slots = slots + self.drop_path(self.mlp(self.norm_mlp(slots)))
        return slots, attn


class SlotAttentionEncoder(nn.Module):
    def __init__(
        self, num_iter, num_slots, feature_size, 
        slot_size, drop_path=0.2,
        use_feats_mlp=False, use_pe=False,
        init_method='boqsa',
        hard_assign=False,
        ):
        super().__init__()
        
        self.num_iter = num_iter
        self.num_slots = num_slots
        self.slot_size = slot_size

        self.mlp = nn.Sequential(
            nn.Linear(feature_size, feature_size),
            nn.ReLU(),
            nn.Linear(feature_size, slot_size)
        )
        self.pe = PositionEmbed(feature_size, [224 // 16, 224 // 16]) if use_pe else nn.Identity()

        self.slot_attention = SlotAttention(slot_size, slot_size, drop_path=drop_path, hard_assign=hard_assign, init_method=init_method)

        num_blocks = 3
        dpr = [x.item() for x in torch.linspace(0, drop_path, num_blocks)]  # stochastic depth decay rule
        self.self_attn = nn.Sequential(*[
            Block(
                dim=slot_size,
                num_heads=8,
                qkv_bias=True,
                drop_path=dpr[i],
            ) for i in range(num_blocks)
        ])
        self.init_method = init_method
        if init_method == 'boqsa':
            self.slots_init = nn.Parameter(torch.zeros(1, num_slots, slot_size))
            nn.init.xavier_uniform_(self.slots_init)
        elif init_method == 'sa':
            self.slot_mu = nn.Parameter(torch.zeros(1, 1, slot_size))
            self.slot_log_sigma = nn.Parameter(torch.zeros(1, 1, slot_size))
            nn.init.xavier_uniform_(self.slot_mu)
            nn.init.xavier_uniform_(self.slot_log_sigma)
    
    def forward(self, f, sigma, slots_init=None):
        B = f.shape[0]
        f = self.pe(f)
        f = self.mlp(f)
        if slots_init == None:
            # initialize slots.
            if self.init_method == 'boqsa':
                mu = self.slots_init.expand(B, -1, -1)
                z = torch.randn_like(mu).type_as(f)
                slots_init = mu + z * sigma * mu.detach()
            elif self.init_method == 'sa':
                slots_init = torch.randn(B, self.num_slots, self.slot_size).type_as(f) * torch.exp(self.slot_log_sigma) + self.slot_mu
        slots, attn  = self.slot_attention(f, slots_init, self.num_iter)
        slots = self.self_attn(slots)
        
        return {
            'slots': slots,
            'slots_init': slots_init,
            'attn': attn,
        }


