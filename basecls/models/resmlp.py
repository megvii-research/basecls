#!/usr/bin/env python3
# Copyright (c) 2015-present, Facebook, Inc.
# This file has been modified by Megvii ("Megvii Modifications").
# All Megvii Modifications are Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
"""ResMLP Series

ResMLP: `"ResMLP: Feedforward networks for image classification with data-efficient training"
<https://arxiv.org/abs/2105.03404>`_

References:
    https://github.com/facebookresearch/deit/blob/main/resmlp_models.py
"""
import megengine as mge
import megengine.functional as F
import megengine.hub as hub
import megengine.module as M

from basecls.layers import DropPath, init_vit_weights
from basecls.utils import recursive_update, registers

from .vit import FFN, PatchEmbed

__all__ = ["Affine", "ResMLPBlock", "ResMLP"]


class Affine(M.Module):
    """ResMLP Affine Layer."""

    def __init__(self, dim: int):
        super().__init__()
        self.alpha = mge.Parameter(F.ones(dim))
        self.beta = mge.Parameter(F.zeros(dim))

    def forward(self, x):
        return self.alpha * x + self.beta


class ResMLPBlock(M.Module):
    """ResMLP block.

    Args:
        dim: Number of input channels.
        drop: Dropout ratio.
        drop_path: Stochastic depth rate.
        num_patches: Number of patches.
        init_scale: Initial value for LayerScale.
        ffn_ratio: Ratio of ffn hidden dim to embedding dim.
        act_name: activation function.
    """

    def __init__(
        self,
        dim: int,
        drop: float,
        drop_path: float,
        num_patches: int,
        init_scale: float,
        ffn_ratio: float,
        act_name: str,
    ):
        super().__init__()
        self.norm1 = Affine(dim)
        self.attn = M.Linear(num_patches, num_patches)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else None
        self.norm2 = Affine(dim)
        self.ffn = FFN(
            in_features=dim, hidden_features=int(ffn_ratio * dim), act_name=act_name, drop=drop
        )
        self.gamma1 = mge.Parameter(init_scale * F.ones((dim)))
        self.gamma2 = mge.Parameter(init_scale * F.ones((dim)))

    def forward(self, x):
        if self.drop_path:
            x = x + self.drop_path(
                self.gamma1 * self.attn(self.norm1(x).transpose(0, 2, 1)).transpose(0, 2, 1)
            )
            x = x + self.drop_path(self.gamma2 * self.ffn(self.norm2(x)))
        else:
            x = x + self.gamma1 * self.attn(self.norm1(x).transpose(0, 2, 1)).transpose(0, 2, 1)
            x = x + self.gamma2 * self.ffn(self.norm2(x))
        return x


@registers.models.register()
class ResMLP(M.Module):
    """ResMLP model.

    Args:
        img_size: Input image size. Default: ``224``
        patch_size: Patch token size. Default: ``16``
        in_chans: Number of input image channels. Default: ``3``
        embed_dim: Number of linear projection output channels. Default: ``768``
        depth: Depth of Transformer Encoder layer. Default: ``12``
        drop_rate: Dropout rate. Default: ``0.0``
        drop_path_rate: Stochastic depth rate. Default: ``0.0``
        embed_layer: Patch embedding layer. Default: :py:class:`PatchEmbed`
        init_scale: Initial value for LayerScale. Default: ``1e-4``
        ffn_ratio: Ratio of ffn hidden dim to embedding dim. Default: ``4.0``
        act_name: Activation function. Default: ``"gelu"``
        num_classes: Number of classes. Default: ``1000``
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        embed_layer: M.Module = PatchEmbed,
        init_scale: float = 1e-4,
        ffn_ratio: float = 4.0,
        act_name: str = "gelu",
        num_classes: int = 1000,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = embed_layer(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        dpr = [drop_path_rate for _ in range(depth)]

        self.blocks = [
            ResMLPBlock(
                dim=embed_dim,
                drop=drop_rate,
                drop_path=dpr[i],
                num_patches=num_patches,
                init_scale=init_scale,
                ffn_ratio=ffn_ratio,
                act_name=act_name,
            )
            for i in range(depth)
        ]
        self.norm = Affine(embed_dim)
        self.head = M.Linear(embed_dim, num_classes) if num_classes > 0 else None
        self.apply(init_vit_weights)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x.mean(axis=1).reshape(B, 1, -1)
        x = x[:, 0]
        if self.head:
            x = self.head(x)
        return x


def _build_resmlp(**kwargs):
    model_args = dict(embed_dim=384, drop_path_rate=0.05)
    recursive_update(model_args, kwargs)
    return ResMLP(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/resmlp/resmlp_s12/resmlp_s12.pkl"
)
def resmlp_s12(**kwargs):
    model_args = dict(depth=12, init_scale=0.1)
    recursive_update(model_args, kwargs)
    return _build_resmlp(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/resmlp/resmlp_s24/resmlp_s24.pkl"
)
def resmlp_s24(**kwargs):
    model_args = dict(depth=24, init_scale=1e-5)
    recursive_update(model_args, kwargs)
    return _build_resmlp(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/resmlp/resmlp_s36/resmlp_s36.pkl"
)
def resmlp_s36(**kwargs):
    model_args = dict(depth=36, init_scale=1e-6)
    recursive_update(model_args, kwargs)
    return _build_resmlp(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/resmlp/resmlp_b24/resmlp_b24.pkl"
)
def resmlp_b24(**kwargs):
    model_args = dict(patch_size=8, embed_dim=768, depth=24, init_scale=1e-6)
    recursive_update(model_args, kwargs)
    return _build_resmlp(**model_args)
