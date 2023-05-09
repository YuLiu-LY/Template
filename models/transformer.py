import math
from methods.utils import *
from timm.models.vision_transformer import LayerScale, Mlp, DropPath


class MultiHeadAttention(nn.Module):
    
    def __init__(self, d_model, num_heads, dropout=0., gain=1.):
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        
        self.attn_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        
        self.proj_q = linear(d_model, d_model, bias=False)
        self.proj_k = linear(d_model, d_model, bias=False)
        self.proj_v = linear(d_model, d_model, bias=False)
        self.proj_o = linear(d_model, d_model, bias=False, gain=gain)
    
    
    def forward(self, q, k, v, attn_mask=None, return_attn=False):
        """
        q: batch_size x target_len x d_model
        k: batch_size x source_len x d_model
        v: batch_size x source_len x d_model
        attn_mask: target_len x source_len
        return: batch_size x target_len x d_model
        """
        B, T, _ = q.shape
        _, S, _ = k.shape
        
        q = self.proj_q(q).view(B, T, self.num_heads, -1).transpose(1, 2)
        k = self.proj_k(k).view(B, S, self.num_heads, -1).transpose(1, 2)
        v = self.proj_v(v).view(B, S, self.num_heads, -1).transpose(1, 2)
        
        q = q * (q.shape[-1] ** (-0.5))
        attn = torch.matmul(q, k.transpose(-1, -2))
        
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn_d = self.attn_dropout(attn)
        
        output = torch.matmul(attn_d, v).transpose(1, 2).reshape(B, T, -1)
        output = self.proj_o(output)
        output = self.output_dropout(output)
        if return_attn:
            return output, attn.mean(1).permute(0, 2, 1)
        else:
            return output


class TransformerEncoderBlock(nn.Module):
    
    def __init__(self, d_model, num_heads, dropout=0., gain=1., is_first=False):
        super().__init__()
        
        self.is_first = is_first
        
        self.attn_layer_norm = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout, gain)
        
        self.ffn_layer_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            linear(d_model, 4 * d_model, weight_init='kaiming'),
            nn.ReLU(),
            linear(4 * d_model, d_model, gain=gain),
            nn.Dropout(dropout))
    
    
    def forward(self, input):
        """
        input: batch_size x source_len x d_model
        return: batch_size x source_len x d_model
        """
        if self.is_first:
            input = self.attn_layer_norm(input)
            x = self.attn(input, input, input)
            input = input + x
        else:
            x = self.attn_layer_norm(input)
            x = self.attn(x, x, x)
            input = input + x
        
        x = self.ffn_layer_norm(input)
        x = self.ffn(x)
        return input + x


class TransformerEncoder(nn.Module):
    
    def __init__(self, num_blocks, d_model, num_heads, dropout=0.):
        super().__init__()
        
        if num_blocks > 0:
            gain = (2 * num_blocks) ** (-0.5)
            self.blocks = nn.ModuleList(
                [TransformerEncoderBlock(d_model, num_heads, dropout, gain, is_first=True)] +
                [TransformerEncoderBlock(d_model, num_heads, dropout, gain, is_first=False)
                 for _ in range(num_blocks - 1)])
        else:
            self.blocks = nn.ModuleList()
        
        self.layer_norm = nn.LayerNorm(d_model)
    
    
    def forward(self, input):
        """
        input: batch_size x source_len x d_model
        return: batch_size x source_len x d_model
        """
        for block in self.blocks:
            input = block(input)
        
        return self.layer_norm(input)


class TransformerDecoderBlock(nn.Module):
    
    def __init__(self, max_len, d_model, num_heads, dropout=0., gain=1., is_first=False, causal_attn=True):
        super().__init__()
        
        self.is_first = is_first
        
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, gain)
        
        if causal_attn:
            mask = torch.triu(torch.ones((max_len, max_len), dtype=torch.bool), diagonal=1)
        else:
            mask = torch.zeros((max_len, max_len), dtype=torch.bool)
        self.register_buffer('self_attn_mask', mask)
        
        self.encoder_decoder_attn_layer_norm = nn.LayerNorm(d_model)
        self.encoder_decoder_attn = MultiHeadAttention(d_model, num_heads, dropout, gain)
        
        self.ffn_layer_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            linear(d_model, 4 * d_model, weight_init='kaiming'),
            nn.ReLU(),
            linear(4 * d_model, d_model, gain=gain),
            nn.Dropout(dropout))
    
    
    def forward(self, input, encoder_output, return_attn=False):
        """
        input: batch_size x target_len x d_model
        encoder_output: batch_size x source_len x d_model
        return: batch_size x target_len x d_model
        """
        T = input.shape[1]
        
        if self.is_first:
            input = self.self_attn_layer_norm(input)
            x = self.self_attn(input, input, input, self.self_attn_mask[:T, :T])
            input = input + x
        else:
            x = self.self_attn_layer_norm(input)
            x = self.self_attn(x, x, x, self.self_attn_mask[:T, :T])
            input = input + x
        
        x = self.encoder_decoder_attn_layer_norm(input)
        if return_attn:
            x, attn = self.encoder_decoder_attn(x, encoder_output, encoder_output, return_attn=True)
            input = input + x
            x = self.ffn_layer_norm(input)
            x = self.ffn(x)
            return input + x, attn
        else:
            x = self.encoder_decoder_attn(x, encoder_output, encoder_output)
            input = input + x
            x = self.ffn_layer_norm(input)
            x = self.ffn(x)
            return input + x
        

class TransformerDecoder(nn.Module):
    
    def __init__(self, num_blocks, max_len, d_model, num_heads, dropout=0., causal_attn=True):
        super().__init__()
        
        if num_blocks > 0:
            gain = (3 * num_blocks) ** (-0.5)
            self.blocks = nn.ModuleList(
                [TransformerDecoderBlock(max_len, d_model, num_heads, dropout, gain, is_first=True, causal_attn=causal_attn)] +
                [TransformerDecoderBlock(max_len, d_model, num_heads, dropout, gain, is_first=False, causal_attn=causal_attn)
                 for _ in range(num_blocks - 1)])
        else:
            self.blocks = nn.ModuleList()
        
        self.layer_norm = nn.LayerNorm(d_model)
    
    
    def forward(self, input, encoder_output):
        """
        input: batch_size x target_len x d_model
        encoder_output: batch_size x source_len x d_model
        return: batch_size x target_len x d_model
        """
        for i, block in enumerate(self.blocks):
            if i == len(self.blocks) - 1:
                input, attn = block(input, encoder_output, return_attn=True)
            else:
                input = block(input, encoder_output)
        
        return self.layer_norm(input), attn


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, xq, xkv=None, attn_mask=None, return_attn=False):
        B, N, C = xq.shape
        q = self.proj_q(xq).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if xkv is None:
            k = self.proj_k(xq).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            v = self.proj_v(xq).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        else:
            k = self.proj_k(xkv).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            v = self.proj_v(xkv).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, float('-inf'))
        attn = attn.softmax(dim=-1)
        attn_d = self.attn_drop(attn)

        x = (attn_d @ v).transpose(1, 2).reshape(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if return_attn:
            return x, attn.mean(1).permute(0, 2, 1)
        else:
            return x


class Block(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class DecoderBlock(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.self_attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.cross_attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm3 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls3 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, encoder_output, self_attn_mask=None, return_attn=False):
        x = x + self.drop_path1(self.ls1(self.self_attn(self.norm1(x), attn_mask=self_attn_mask)))
        h, cross_attn = self.cross_attn(self.norm2(x), encoder_output, return_attn=True)
        x = x + self.drop_path2(self.ls2(h))
        x = x + self.drop_path3(self.ls3(self.mlp(self.norm3(x))))
        if return_attn:
            return x, cross_attn
        else:
            return x
    

class ViTDecoder(nn.Module):
    def __init__(
            self,
            num_blocks,
            max_len,
            dim,
            num_heads,
            causal_attn=True,
            mlp_ratio=4.,
            qkv_bias=True,
            drop_rate=0.,
            attn_drop_rate=0.,
            init_values=0.1,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm
    ):
        super().__init__()

        if causal_attn:
            mask = torch.triu(torch.ones((max_len, max_len), dtype=torch.bool), diagonal=1)
            self.register_buffer('self_attn_mask', mask)
        else:
            self.self_attn_mask = None
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_blocks)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            DecoderBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                init_values=init_values,
                drop_path=dpr[i],
                norm_layer=norm_layer
            ) for i in range(num_blocks)
        ])

        self.norm = norm_layer(dim)
        self.apply(self._init_weights)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.self_attn.proj.weight.data, layer_id + 1)
            rescale(layer.cross_attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)
    
    def _init_weights(self, m):
        std = 0.02
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=std, a=-std, b=std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, input, encoder_output):
        L = input.shape[1]
        self_attn_mask = self.self_attn_mask[:L, :L]
        for i, block in enumerate(self.blocks):
            if i == len(self.blocks) - 1:
                input, attn = block(input, encoder_output, self_attn_mask, return_attn=True)
            else:
                input = block(input, encoder_output, self_attn_mask)
        
        return self.norm(input), attn


class CrossAttnBlock(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 post_norm=False):
        super().__init__()
        if post_norm:
            self.norm_post = norm_layer(dim)
            self.norm_q = nn.Identity()
            self.norm_k = nn.Identity()
        else:
            self.norm_q = norm_layer(dim)
            self.norm_k = norm_layer(dim)
            self.norm_post = nn.Identity()
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, query, key, *, mask=None):
        x = query
        x = x + self.drop_path(self.attn(self.norm_q(query), self.norm_k(key), attn_mask=mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = self.norm_post(x)
        return x