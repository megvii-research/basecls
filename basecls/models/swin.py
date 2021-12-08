#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
"""
Swin Transformer Series

Swin Transformer : `"Hierarchical Vision Transformer using Shifted Windows"
<https://arxiv.org/abs/2103.14030>`_
"""
from typing import Sequence, Tuple

import megengine as mge
import megengine.functional as F
import megengine.module as M
import numpy as np
from megengine.utils.tuple_function import _pair as to_2tuple

from basecls.layers import DropPath, norm2d, trunc_normal_
from basecls.utils import recursive_update, registers

from .vit import FFN, PatchEmbed

__all__ = [
    "window_partition",
    "window_reverse",
    "WindowAttention",
    "PatchMerging",
    "SwinBlock",
    "SwinBasicLayer",
    "SwinTransformer",
]


def window_partition(x, window_size: int):
    """
    Args:
        x: (B, H, W, C)
        window_size: window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.reshape(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.transpose(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size: Window size
        H: Height of image
        W: Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.transpose(0, 1, 3, 2, 4, 5).reshape(B, H, W, -1)
    return x


class WindowAttention(M.Module):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim: Number of input channels.
        window_size: The height and width of the window.
        num_heads: Number of attention heads.
        qkv_bias: If True, add a learnable bias to query, key, value. Default: ``True``
        qk_scale: Override default qk scale of ``head_dim ** -0.5`` if set.
        attn_drop: Dropout ratio of attention weight. Default: ``0.0``
        proj_drop: Dropout ratio of output. Default: ``0.0``
    """

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: float = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.rel_pos_bias_table = mge.Parameter(
            F.zeros([(2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads])
        )  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = np.arange(self.window_size[0])
        coords_w = np.arange(self.window_size[1])
        coords = np.stack(np.meshgrid(coords_h, coords_w))  # 2, Wh, Ww
        coords_flatten = np.reshape(coords, (coords.shape[0], -1))  # 2, Wh*Ww
        rel_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        rel_coords = rel_coords.transpose(1, 2, 0)  # Wh*Ww, Wh*Ww, 2
        rel_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        rel_coords[:, :, 1] += self.window_size[1] - 1
        rel_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        rel_pos_index = rel_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.rel_pos_index = mge.Tensor(rel_pos_index)

        self.qkv = M.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = M.Dropout(attn_drop)
        self.proj = M.Linear(dim, dim)
        self.proj_drop = M.Dropout(proj_drop)

        trunc_normal_(self.rel_pos_bias_table, std=0.02)
        self.softmax = M.Softmax(axis=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .transpose(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = F.matmul(q, k.transpose(0, 1, 3, 2))

        rel_pos_bias = self.rel_pos_bias_table[self.rel_pos_index.reshape(-1)].reshape(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )  # Wh*Ww,Wh*Ww,nH
        rel_pos_bias = rel_pos_bias.transpose(2, 0, 1)  # nH, Wh*Ww, Wh*Ww
        attn = attn + F.expand_dims(rel_pos_bias, 0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.reshape(B_ // nW, nW, self.num_heads, N, N) + F.expand_dims(mask, [0, 2])
            attn = attn.reshape(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = F.matmul(attn, v).transpose(0, 2, 1, 3).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def _module_info_string(self) -> str:
        return f"dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}"


class PatchMerging(M.Module):
    r"""Patch Merging Layer.

    Args:
        dim: Number of input channels.
        input_resolution: Resolution of input feature.
        norm_name: Normalization layer. Default: ``"LN"``
    """

    def __init__(self, dim: int, input_resolution: Tuple[int, int], norm_name: str = "LN"):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.reduction = M.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm2d(norm_name, 4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.reshape(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = F.concat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.reshape(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def _module_info_string(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"


class SwinBlock(M.Module):
    r"""Swin Transformer Block.

    Args:
        dim: Number of input channels.
        input_resolution: Input resulotion.
        num_heads: Number of attention heads.
        window_size: Window size. Default: ``7``
        shift_size: Shift size for SW-MSA. Default: ``0``
        ffn_ratio: Ratio of ffn hidden dim to embedding dim. Default: ``4.0``
        qkv_bias: If True, add a learnable bias to query, key, value. Default: ``True``
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
        input_resolution: Tuple[int, int],
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_name: str = "LN",
        act_name: str = "gelu",
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.ffn_ratio = ffn_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm2d(norm_name, dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else None
        self.norm2 = norm2d(norm_name, dim)
        ffn_hidden_dim = int(dim * ffn_ratio)
        self.ffn = FFN(
            in_features=dim, hidden_features=ffn_hidden_dim, drop=drop, act_name=act_name
        )

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = F.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(
                img_mask, self.window_size
            )  # nW, window_size, window_size, 1
            mask_windows = mask_windows.reshape(-1, self.window_size * self.window_size)
            attn_mask = F.expand_dims(mask_windows, 1) - F.expand_dims(mask_windows, 2)
            attn_mask[attn_mask != 0] = -100.0
            attn_mask[attn_mask == 0] = 0.0
        else:
            attn_mask = None

        self.attn_mask = mge.Tensor(attn_mask) if attn_mask is not None else None

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.reshape(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = F.roll(x, shift=(-self.shift_size, -self.shift_size), axis=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size
        )  # nW*B, window_size, window_size, C
        x_windows = x_windows.reshape(
            -1, self.window_size * self.window_size, C
        )  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.reshape(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = F.roll(shifted_x, shift=(self.shift_size, self.shift_size), axis=(1, 2))
        else:
            x = shifted_x
        x = x.reshape(B, H * W, C)

        # FFN
        if self.drop_path:
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.ffn(self.norm2(x)))
        else:
            x = shortcut + x
            x = x + self.ffn(self.norm2(x))

        return x

    def _module_info_string(self) -> str:
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, "
            f"num_heads={self.num_heads}, window_size={self.window_size}, "
            f"shift_size={self.shift_size}, ffn_ratio={self.ffn_ratio}"
        )


class SwinBasicLayer(M.Module):
    """A basic Swin Transformer layer for one stage.

    Args:
        dim: Number of input channels.
        input_resolution: Input resolution.
        depth: Number of blocks.
        num_heads: Number of attention heads.
        window_size: Local window size.
        ffn_ratio: Ratio of ffn hidden dim to embedding dim.
        qkv_bias: If True, add a learnable bias to query, key, value. Default: ``True``
        qk_scale: Override default qk scale of ``head_dim ** -0.5`` if set.
        drop: Dropout rate. Default: ``0.0``
        attn_drop: Attention dropout rate. Default: ``0.0``
        drop_path: Stochastic depth rate. Default: ``0.0``
        norm_name: Normalization layer. Default: ``"LN"``
        act_name: Activation layer. Default: ``"gelu"``
        downsample: Downsample layer at the end of the layer. Default: ``None``
    """

    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        downsample: M.Module = None,
        norm_name: str = "LN",
        act_name: str = "gelu",
    ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # build blocks
        self.blocks = [
            SwinBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                ffn_ratio=ffn_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_name=norm_name,
                act_name=act_name,
            )
            for i in range(depth)
        ]

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim, input_resolution, norm_name)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample:
            x = self.downsample(x)
        return x

    def _module_info_string(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


@registers.models.register()
class SwinTransformer(M.Module):
    r"""Swin Transformer
        A PyTorch impl of :
            `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
            https://arxiv.org/pdf/2103.14030

    Args:
        img_size: Input image size. Default: ``224``
        patch_size: Patch size. Default: ``4``
        in_chans: Number of input image channels. Default: ``3``
        embed_dim: Patch embedding dimension. Default: ``96``
        depths: Depth of each Swin Transformer layer.
        num_heads: Number of attention heads in different layers.
        window_size: Window size. Default: ``7``
        ffn_ratio: Ratio of ffn hidden dim to embedding dim. Default: ``4.0``
        qkv_bias: If True, add a learnable bias to query, key, value. Default: ``True``
        qk_scale: Override default qk scale of head_dim ** -0.5 if set. Default: ``None``
        ape: If True, add absolute position embedding to the patch embedding. Default: ``False``
        patch_norm: If True, add normalization after patch embedding. Default: ``True``
        drop_rate: Dropout rate. Default: ``0``
        attn_drop_rate: Attention dropout rate. Default: ``0``
        drop_path_rate: Stochastic depth rate. Default: ``0.1``
        embed_layer: Patch embedding layer. Default: :py:class:`PatchEmbed`
        norm_name: Normalization layer. Default: ``"LN"``
        act_name: Activation layer. Default: ``"gelu"``
        num_classes: Number of classes for classification head. Default: ``1000``
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        depths: Sequence[int] = [2, 2, 6, 2],
        num_heads: Sequence[int] = [3, 6, 12, 24],
        window_size: int = 7,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float = None,
        ape: bool = False,
        patch_norm: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        embed_layer: M.Module = PatchEmbed,
        norm_name: str = "LN",
        act_name: str = "gelu",
        num_classes: int = 1000,
        **kwargs,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.ffn_ratio = ffn_ratio

        # split image into non-overlapping patches
        self.patch_embed = embed_layer(
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            norm_name=norm_name if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        grid_size = self.patch_embed.grid_size
        self.grid_size = grid_size

        # absolute position embedding
        if self.ape:
            self.abs_pos_embed = mge.Parameter(F.zeros([1, num_patches, embed_dim]))
            trunc_normal_(self.abs_pos_embed, std=0.02)

        self.pos_drop = M.Dropout(drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in np.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = []
        for i_layer in range(self.num_layers):
            layer = SwinBasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(
                    grid_size[0] // (2 ** i_layer),
                    grid_size[1] // (2 ** i_layer),
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                ffn_ratio=self.ffn_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                norm_name=norm_name,
                act_name=act_name,
            )
            self.layers.append(layer)

        self.norm = norm2d(norm_name, self.num_features)
        self.head = M.Linear(self.num_features, num_classes) if num_classes > 0 else None

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, M.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, M.Linear) and m.bias is not None:
                M.init.zeros_(m.bias)
        elif isinstance(m, M.LayerNorm):
            M.init.zeros_(m.bias)
            M.init.ones_(m.weight)

    def forward(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.abs_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        x = x.transpose(0, 2, 1).mean(-1)  # B C 1
        x = F.flatten(x, 1)
        if self.head:
            x = self.head(x)
        return x


@registers.models.register()
def swin_tiny_patch4_window7_224(**kwargs):
    model_args = dict(
        patch_size=4,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        drop_path_rate=0.2,
    )
    recursive_update(model_args, kwargs)
    return SwinTransformer(**model_args)


@registers.models.register()
def swin_small_patch4_window7_224(**kwargs):
    model_args = dict(
        patch_size=4,
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        drop_path_rate=0.3,
    )
    recursive_update(model_args, kwargs)
    return SwinTransformer(**model_args)


@registers.models.register()
def swin_base_patch4_window7_224(**kwargs):
    model_args = dict(
        patch_size=4,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        drop_path_rate=0.5,
    )
    recursive_update(model_args, kwargs)
    return SwinTransformer(**model_args)


@registers.models.register()
def swin_base_patch4_window12_384(**kwargs):
    model_args = dict(img_size=384, window_size=12)
    recursive_update(model_args, kwargs)
    return swin_base_patch4_window7_224(**model_args)


@registers.models.register()
def swin_large_patch4_window7_224(**kwargs):
    model_args = dict(
        patch_size=4,
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7,
        drop_path_rate=0.5,
    )
    recursive_update(model_args, kwargs)
    return SwinTransformer(**model_args)


@registers.models.register()
def swin_large_patch4_window12_384(**kwargs):
    model_args = dict(img_size=384, window_size=12)
    recursive_update(model_args, kwargs)
    return swin_large_patch4_window7_224(**model_args)
