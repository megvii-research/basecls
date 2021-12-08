#!/usr/bin/env python3
# Hacked together by / Copyright 2021 Ross Wightman
# This file has been modified by Megvii ("Megvii Modifications").
# All Megvii Modifications are Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
"""Vision Transformer (ViT)

ViT: `"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
<https://arxiv.org/abs/2010.11929>`_

DeiT: `"Training data-efficient image transformers & distillation through attention"
<https://arxiv.org/abs/2012.12877>`_
"""
import math
from collections import OrderedDict
from typing import Callable, Optional, Union

import cv2
import megengine as mge
import megengine.functional as F
import megengine.hub as hub
import megengine.module as M
import numpy as np
from loguru import logger
from megengine.utils.tuple_function import _pair as to_2tuple

from basecls.layers import DropPath, activation, init_vit_weights, norm2d, trunc_normal_
from basecls.utils import recursive_update, registers

__all__ = ["PatchEmbed", "Attention", "FFN", "EncoderBlock", "ViT"]


class PatchEmbed(M.Module):
    """Image to Patch Embedding

    Args:
        img_size: Image size.  Default: ``224``
        patch_size: Patch token size. Default: ``16``
        in_chans: Number of input image channels. Default: ``3``
        embed_dim: Number of linear projection output channels. Default: ``768``
        flatten: Flatten embedding. Default: ``True``
        norm_name: Normalization layer. Default: ``None``
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        flatten: bool = True,
        norm_name: str = None,
        **kwargs,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = M.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm2d(norm_name, embed_dim) if norm_name else None

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], (
            f"Input image size ({H}*{W}) doesn't match model "
            f"({self.img_size[0]}*{self.img_size[1]})."
        )
        x = self.proj(x)
        if self.flatten:
            x = F.flatten(x, 2).transpose(0, 2, 1)
        if self.norm:
            x = self.norm(x)
        return x


class Attention(M.Module):
    """Self-Attention block.

    Args:
        dim: input Number of input channels.
        num_heads: Number of attention heads. Default: ``8``
        qkv_bias: If True, add a learnable bias to query, key, value. Default: ``False``
        qk_scale: Override default qk scale of ``head_dim ** -0.5`` if set.
        attn_drop: Dropout ratio of attention weight. Default: ``0.0``
        proj_drop: Dropout ratio of output. Default: ``0.0``
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = M.Linear(dim, dim * 3, bias=qkv_bias)
        self.softmax = M.Softmax(axis=-1)
        self.attn_drop = M.Dropout(attn_drop)
        self.proj = M.Linear(dim, dim)
        self.proj_drop = M.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .transpose(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = F.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = F.matmul(attn, v).transpose(0, 2, 1, 3).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FFN(M.Module):
    """FFN for ViT

    Args:
        in_features: Number of input features.
        hidden_features: Number of input features. Default: ``None``
        out_features: Number of output features. Default: ``None``
        drop: Dropout ratio. Default: ``0.0``
        act_name: activation function. Default: ``"gelu"``
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        drop: float = 0.0,
        act_name: str = "gelu",
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = M.Linear(in_features, hidden_features)
        self.act = activation(act_name)
        self.fc2 = M.Linear(hidden_features, out_features)
        self.drop = M.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class EncoderBlock(M.Module):
    """Transformer Encoder block.

    Args:
        dim: Number of input channels.
        num_heads: Number of attention heads.
        ffn_ratio: Ratio of ffn hidden dim to embedding dim. Default: ``4.0``
        qkv_bias: If True, add a learnable bias to query, key, value. Default: ``False``
        qk_scale: Override default qk scale of ``head_dim ** -0.5`` if set.
        drop: Dropout ratio of non-attention weight. Default: ``0.0``
        attn_drop: Dropout ratio of attention weight. Default: ``0.0``
        drop_path: Stochastic depth rate. Default: ``0.0``
        norm_name: Normalization layer. Default: ``"LN"``
        act_name: Activation layer. Default: ``"gelu"``
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float = None,
        attn_drop: float = 0.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
        norm_name: str = "LN",
        act_name: str = "gelu",
        **kwargs,
    ):
        super().__init__()
        self.norm1 = norm2d(norm_name, dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else None
        self.norm2 = norm2d(norm_name, dim)
        ffn_hidden_dim = int(dim * ffn_ratio)
        self.ffn = FFN(
            in_features=dim, hidden_features=ffn_hidden_dim, drop=drop, act_name=act_name
        )

    def forward(self, x):
        if self.drop_path:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.ffn(self.norm2(x)))
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.ffn(self.norm2(x))
        return x


@registers.models.register()
class ViT(M.Module):
    """ViT model.

    Args:
        img_size: Input image size. Default: ``224``
        patch_size: Patch token size. Default: ``16``
        in_chans: Number of input image channels. Default: ``3``
        embed_dim: Number of linear projection output channels. Default: ``768``
        depth: Depth of Transformer Encoder layer. Default: ``12``
        num_heads: Number of attention heads. Default: ``12``
        ffn_ratio: Ratio of ffn hidden dim to embedding dim. Default: ``4.0``
        qkv_bias: If True, add a learnable bias to query, key, value. Default: ``True``
        qk_scale: Override default qk scale of head_dim ** -0.5 if set. Default: ``None``
        representation_size: Size of representation layer (pre-logits). Default: ``None``
        distilled: Includes a distillation token and head. Default: ``False``
        drop_rate: Dropout rate. Default: ``0.0``
        attn_drop_rate: Attention dropout rate. Default: ``0.0``
        drop_path_rate: Stochastic depth rate. Default: ``0.0``
        embed_layer: Patch embedding layer. Default: :py:class:`PatchEmbed`
        norm_name: Normalization layer. Default: ``"LN"``
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
        num_heads: int = 12,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float = None,
        representation_size: int = None,
        distilled: bool = False,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        embed_layer: M.Module = PatchEmbed,
        norm_name: str = "LN",
        act_name: str = "gelu",
        num_classes: int = 1000,
        **kwargs,
    ):
        super().__init__()
        # Patch Embedding
        self.embed_dim = embed_dim
        self.patch_embed = embed_layer(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        # CLS & DST Tokens
        self.cls_token = mge.Parameter(F.zeros([1, 1, embed_dim]))
        self.dist_token = mge.Parameter(F.zeros([1, 1, embed_dim])) if distilled else None
        self.num_tokens = 2 if distilled else 1
        # Pos Embedding
        self.pos_embed = mge.Parameter(F.zeros([1, num_patches + self.num_tokens, embed_dim]))
        self.pos_drop = M.Dropout(drop_rate)
        # Blocks
        dpr = [
            x.item() for x in F.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = M.Sequential(
            *[
                EncoderBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    ffn_ratio=ffn_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_name=norm_name,
                    act_name=act_name,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm2d(norm_name, embed_dim)
        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = M.Sequential(
                OrderedDict(
                    [("fc", M.Linear(embed_dim, representation_size)), ("act", activation("tanh"))]
                )
            )
        else:
            self.pre_logits = None
        # Classifier head(s)
        self.head = M.Linear(self.embed_dim, num_classes) if num_classes > 0 else None
        self.head_dist = None
        if distilled:
            self.head_dist = M.Linear(self.embed_dim, num_classes) if num_classes > 0 else None
        # Init
        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(init_vit_weights)

    def forward(self, x):
        x = self.patch_embed(x)
        cls_token = F.broadcast_to(self.cls_token, (x.shape[0], 1, self.cls_token.shape[-1]))
        if self.dist_token is None:
            x = F.concat((cls_token, x), axis=1)
        else:
            dist_token = F.broadcast_to(self.dist_token, (x.shape[0], 1, self.dist_token.shape[-1]))
            x = F.concat((cls_token, dist_token, x), axis=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            x = x[:, 0]
            if self.pre_logits:
                x = self.pre_logits(x)
        else:
            x = x[:, 0], x[:, 1]
        if self.head_dist is not None:
            x_cls, x_dist = x
            if self.head:
                x_cls = self.head(x_cls)
            if self.head_dist:
                x_dist = self.head_dist(x_dist)
            if self.training:
                # during inference, return the average of both classifier predictions
                return x_cls, x_dist
            else:
                return (x_cls + x_dist) / 2
        elif self.head:
            x = self.head(x)
        return x

    def load_state_dict(
        self,
        state_dict: Union[dict, Callable[[str, mge.Tensor], Optional[np.ndarray]]],
        strict=True,
    ):
        if "pos_embed" in state_dict:
            old_pos_embed = state_dict["pos_embed"]
            old_n_patches = old_pos_embed.shape[1] - self.num_tokens
            old_gs = int(math.sqrt(old_n_patches + 0.5))
            new_n_patches = self.pos_embed.shape[1] - self.num_tokens
            new_gs = int(math.sqrt(new_n_patches + 0.5))
            logger.info("Position embedding grid-size from {} to {}", [old_gs] * 2, [new_gs] * 2)
            logger.info(
                "Resized position embedding: {} to {}", old_pos_embed.shape, self.pos_embed.shape
            )
            if isinstance(old_pos_embed, mge.Tensor):
                old_pos_embed = old_pos_embed.numpy()
            pos_emb_tok, old_pos_emb_grid = np.split(old_pos_embed, [self.num_tokens], axis=1)
            old_pos_emb_grid = old_pos_emb_grid.reshape(old_gs, old_gs, -1).transpose(2, 0, 1)
            new_pos_embed_grid = (
                np.stack(
                    [
                        cv2.resize(c, (new_gs, new_gs), interpolation=cv2.INTER_CUBIC)
                        for c in old_pos_emb_grid
                    ]
                )
                .transpose(1, 2, 0)
                .reshape(1, new_gs ** 2, -1)
            )
            new_pos_embed = np.concatenate([pos_emb_tok, new_pos_embed_grid], axis=1)
            if isinstance(old_pos_embed, mge.Tensor):
                new_pos_embed = mge.Parameter(new_pos_embed)
            state_dict["pos_embed"] = new_pos_embed
        super().load_state_dict(state_dict, strict)


def _build_vit(**kwargs):
    model_args = dict(depth=12, drop_path_rate=0.1)
    recursive_update(model_args, kwargs)
    return ViT(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/"
    "vit/vit_tiny_patch16_224/vit_tiny_patch16_224.pkl"
)
def vit_tiny_patch16_224(**kwargs):
    model_args = dict(patch_size=16, embed_dim=192, num_heads=3)
    recursive_update(model_args, kwargs)
    return _build_vit(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/"
    "vit/vit_tiny_patch16_384/vit_tiny_patch16_384.pkl"
)
def vit_tiny_patch16_384(**kwargs):
    model_args = dict(img_size=384)
    recursive_update(model_args, kwargs)
    return vit_tiny_patch16_224(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/"
    "vit/vit_small_patch16_224/vit_small_patch16_224.pkl"
)
def vit_small_patch16_224(**kwargs):
    model_args = dict(patch_size=16, embed_dim=384, num_heads=6)
    recursive_update(model_args, kwargs)
    return _build_vit(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/"
    "vit/vit_small_patch16_384/vit_small_patch16_384.pkl"
)
def vit_small_patch16_384(**kwargs):
    model_args = dict(img_size=384)
    recursive_update(model_args, kwargs)
    return vit_small_patch16_224(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/"
    "vit/vit_small_patch32_224/vit_small_patch32_224.pkl"
)
def vit_small_patch32_224(**kwargs):
    model_args = dict(patch_size=32, embed_dim=384, num_heads=6)
    recursive_update(model_args, kwargs)
    return _build_vit(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/"
    "vit/vit_small_patch32_384/vit_small_patch32_384.pkl"
)
def vit_small_patch32_384(**kwargs):
    model_args = dict(img_size=384)
    recursive_update(model_args, kwargs)
    return vit_small_patch32_224(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/"
    "vit/vit_base_patch16_224/vit_base_patch16_224.pkl"
)
def vit_base_patch16_224(**kwargs):
    model_args = dict(patch_size=16, embed_dim=768, num_heads=12)
    recursive_update(model_args, kwargs)
    return _build_vit(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/"
    "vit/vit_base_patch16_384/vit_base_patch16_384.pkl"
)
def vit_base_patch16_384(**kwargs):
    model_args = dict(img_size=384)
    recursive_update(model_args, kwargs)
    return vit_base_patch16_224(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/"
    "vit/vit_base_patch32_224/vit_base_patch32_224.pkl"
)
def vit_base_patch32_224(**kwargs):
    model_args = dict(patch_size=32, embed_dim=768, num_heads=12)
    recursive_update(model_args, kwargs)
    return _build_vit(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/"
    "vit/vit_base_patch32_384/vit_base_patch32_384.pkl"
)
def vit_base_patch32_384(**kwargs):
    model_args = dict(img_size=384)
    recursive_update(model_args, kwargs)
    return vit_base_patch32_224(**model_args)
