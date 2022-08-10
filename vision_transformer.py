#!/usr/bin/env python

from collections import OrderedDict
import numpy as np
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

'''
QuickGELU and LayerNorm w/ fp16 from official CLIP repo
(https://github.com/openai/CLIP/blob/3b473b0e682c091a9e53623eebc1ca1657385717/clip/model.py)
'''
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class Attention(nn.Module):
    '''
    A generalized attention module with more flexibility.
    '''

    def __init__(
        self, q_in_dim: int, k_in_dim: int, v_in_dim: int,
        qk_proj_dim: int, v_proj_dim: int, num_heads: int, out_dim: int,
        return_all_features: bool = False,
    ):
        super().__init__()

        self.q_proj = nn.Linear(q_in_dim, qk_proj_dim)
        self.k_proj = nn.Linear(k_in_dim, qk_proj_dim)
        self.v_proj = nn.Linear(v_in_dim, v_proj_dim)
        self.out_proj = nn.Linear(v_proj_dim, out_dim)

        self.num_heads = num_heads
        self.return_all_features = return_all_features
        assert qk_proj_dim % num_heads == 0 and v_proj_dim % num_heads == 0

        self._initialize_weights()


    def _initialize_weights(self):
        for m in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.)


    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        assert q.ndim == 3 and k.ndim == 3 and v.ndim == 3
        N = q.size(0); assert k.size(0) == N and v.size(0) == N
        Lq, Lkv = q.size(1), k.size(1); assert v.size(1) == Lkv

        q, k, v = self.q_proj(q), self.k_proj(k), self.v_proj(v)
        
        H = self.num_heads
        Cqk, Cv = q.size(-1) // H, v.size(-1) // H

        q = q.view(N, Lq, H, Cqk)
        k = k.view(N, Lkv, H, Cqk)
        v = v.view(N, Lkv, H, Cv)

        aff = torch.einsum('nqhc,nkhc->nqkh', q / (Cqk ** 0.5), k)
        aff = aff.softmax(dim=-2)
        mix = torch.einsum('nqlh,nlhc->nqhc', aff, v)

        out = self.out_proj(mix.flatten(-2))

        if self.return_all_features:
            return dict(q=q, k=k, v=v, aff=aff, out=out)
        else:
            return out


class PatchEmbed2D(nn.Module):

    def __init__(
        self,
        patch_size: Tuple[int, int] = (16, 16),
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.in_channels = in_channels

        self.proj = nn.Linear(np.prod(patch_size) * in_channels, embed_dim)


    def _initialize_weights(self, x):
        nn.init.kaiming_normal_(self.proj.weight, 0.)
        nn.init.constant_(self.proj.bias, 0.)


    def forward(self, x: torch.Tensor):
        B, C, H, W = x.size()
        pH, pW = self.patch_size

        assert C == self.in_channels and H % pH == 0 and W % pW == 0

        x = x.view(B, C, H // pH, pH, W // pW, pW).permute(0, 2, 4, 1, 3, 5).flatten(3).flatten(1, 2)
        x = self.proj(x)
        
        return x

class TransformerEncoderLayer(nn.Module):

    def __init__(
        self,
        in_feature_dim: int = 768,
        qkv_dim: int = 768,
        num_heads: int = 12,
        mlp_factor: float = 4.0,
        mlp_dropout: float = 0.0,
        act: nn.Module = QuickGELU,
        return_all_features: bool = False,
    ):
        super().__init__()

        self.return_all_features = return_all_features

        self.attn = Attention(
            q_in_dim=in_feature_dim, k_in_dim=in_feature_dim, v_in_dim=in_feature_dim,
            qk_proj_dim=qkv_dim, v_proj_dim=qkv_dim, num_heads=num_heads, out_dim=in_feature_dim,
            return_all_features=return_all_features,
        )

        mlp_dim = round(mlp_factor * in_feature_dim)
        self.mlp = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_feature_dim, mlp_dim)),
            ('act', act()),
            ('dropout', nn.Dropout(mlp_dropout)),
            ('fc2', nn.Linear(mlp_dim, in_feature_dim)),
        ]))

        self.norm1 = LayerNorm(in_feature_dim)
        self.norm2 = LayerNorm(in_feature_dim)

        self._initialize_weights()


    def _initialize_weights(self):
        for m in (self.mlp[0], self.mlp[-1]):
            nn.init.xavier_uniform_(m.weight)
            nn.init.normal_(m.bias, std=1e-6)


    def forward(self, x: torch.Tensor):
        if self.return_all_features:
            ret_dict = {}
            
            x_norm = self.norm1(x)
            attn_out = self.attn(x_norm, x_norm, x_norm)
            ret_dict['q'] = attn_out['q']
            ret_dict['k'] = attn_out['k']
            ret_dict['v'] = attn_out['v']
            ret_dict['attn_out'] = attn_out['out']
            x = x + attn_out['out']

            x = x + self.mlp(self.norm2(x))
            ret_dict['out'] = x

            return ret_dict
        
        else:
            x_norm = self.norm1(x)
            x = x + self.attn(x_norm, x_norm, x_norm)
            x = x + self.mlp(self.norm2(x))

            return x


class TransformerDecoderLayer(nn.Module):

    def __init__(
        self,
        in_feature_dim: int = 768,
        qkv_dim: int = 768,
        num_heads: int = 12,
        mlp_factor: float = 4.0,
        mlp_dropout: float = 0.0,
        act: nn.Module = QuickGELU,
    ):
        super().__init__()

        self.attn = Attention(
            q_in_dim=in_feature_dim, k_in_dim=in_feature_dim, v_in_dim=in_feature_dim,
            qk_proj_dim=qkv_dim, v_proj_dim=qkv_dim, num_heads=num_heads, out_dim=in_feature_dim,
        )

        mlp_dim = round(mlp_factor * in_feature_dim)
        self.mlp = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_feature_dim, mlp_dim)),
            ('act', act()),
            ('dropout', nn.Dropout(mlp_dropout)),
            ('fc2', nn.Linear(mlp_dim, in_feature_dim)),
        ]))

        self.norm1 = LayerNorm(in_feature_dim)
        self.norm2 = LayerNorm(in_feature_dim)
        self.norm3 = LayerNorm(in_feature_dim)

        self._initialize_weights()


    def _initialize_weights(self):
        for m in (self.mlp[0], self.mlp[-1]):
            nn.init.xavier_uniform_(m.weight)
            nn.init.normal_(m.bias, std=1e-6)


    def forward(self, x: torch.Tensor, y: torch.Tensor):
        y_norm = self.norm3(y)
        x = x + self.attn(self.norm1(x), y_norm, y_norm)
        x = x + self.mlp(self.norm2(x))

        return x


class VisionTransformer2D(nn.Module):

    def __init__(
        self,
        feature_dim: int = 768,
        input_size: Tuple[int, int] = (224, 224),
        patch_size: Tuple[int, int] = (16, 16),
        num_heads: int = 12,
        num_layers: int = 12,
        mlp_factor: float = 4.0,
        act: nn.Module = QuickGELU,
        return_all_features: bool = False,
        ln_pre: bool = False,
    ):
        super().__init__()

        self.return_all_features = return_all_features
        
        self.patch_embed = PatchEmbed2D(patch_size=patch_size, embed_dim=feature_dim)
        self.num_patches = np.prod([x // y for x, y in zip(input_size, patch_size)]) + 1

        self.cls_token = nn.Parameter(torch.zeros([feature_dim]))
        self.pos_embed = nn.Parameter(torch.zeros([self.num_patches, feature_dim]))

        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(
                in_feature_dim=feature_dim, qkv_dim=feature_dim, num_heads=num_heads, mlp_factor=mlp_factor, act=act,
                return_all_features=return_all_features,
            ) for _ in range(num_layers)
        ])

        if ln_pre:
            self.ln_pre = LayerNorm(feature_dim)
        else:
            self.ln_pre = nn.Identity()

        self._initialize_weights()


    def _initialize_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor):
        dtype = self.patch_embed.proj.weight.dtype
        x = x.to(dtype)

        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.view(1, 1, -1).repeat(x.size(0), 1, 1), x], dim=1)
        x = x + self.pos_embed

        x = self.ln_pre(x)

        if self.return_all_features:
            all_features = []
            for blk in self.blocks:
                x = blk(x)
                all_features.append(x)
                x = x['out']
            return all_features
        
        else:
            for blk in self.blocks:
                x = blk(x)
            return x


def model_to_fp16(model: VisionTransformer2D):
    def _module_to_fp16(m: nn.Module):
        if isinstance(m, (nn.Linear,)):
            m.half()
    model.apply(_module_to_fp16)

    model.pos_embed.data = model.pos_embed.data.half()
    model.cls_token.data = model.cls_token.data.half()


vit_presets = {
    'ViT-B/16-lnpre': dict(
        feature_dim=768,
        input_size=(224, 224),
        patch_size=(16, 16),
        num_heads=12,
        num_layers=12,
        mlp_factor=4.0,
        ln_pre=True,
    ),
    'ViT-L/14-lnpre': dict(
        feature_dim=1024,
        input_size=(224, 224),
        patch_size=(14, 14),
        num_heads=16,
        num_layers=24,
        mlp_factor=4.0,
        ln_pre=True,
    ),
}