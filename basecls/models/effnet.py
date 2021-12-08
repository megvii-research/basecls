#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2020 Ross Wightman
# This file has been modified by Megvii ("Megvii Modifications").
# All Megvii Modifications are Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
"""EfficientNet Series

EfficientNet: `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"
<https://arxiv.org/abs/1905.11946>`_

References:
    https://github.com/facebookresearch/pycls/blob/main/pycls/models/effnet.py
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/efficientnet.py
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/mobilenetv3.py
"""
import math
from numbers import Real
from typing import Any, Callable, Mapping, Sequence, Union

import megengine.hub as hub
import megengine.module as M

from basecls.layers import (
    SE,
    DropPath,
    activation,
    build_head,
    conv2d,
    init_weights,
    make_divisible,
    norm2d,
)
from basecls.utils import recursive_update, registers

from .mbnet import MBConv
from .resnet import AnyStage, SimpleStem

__all__ = ["FuseMBConv", "EffNet"]


class FuseMBConv(M.Module):
    """Fusing the proj conv1x1 and depthwise conv into a conv2d.

    Args:
        w_in: input width.
        w_out: output width.
        stride: stride of conv.
        kernel: kernel of conv.
        exp_r: expansion ratio.
        se_r: SE ratio.
        has_skip: whether apply skip connection.
        drop_path_prob: drop path probability.
        norm_name: normalization function.
        act_name: activation function.
    """

    def __init__(
        self,
        w_in: int,
        w_out: int,
        stride: int,
        kernel: int,
        exp_r: float,
        se_r: float,
        has_skip: bool,
        drop_path_prob: float,
        norm_name: str,
        act_name: str,
        **kwargs,
    ):
        super().__init__()
        # Expansion
        w_mid = w_in
        w_exp = int(w_in * exp_r)
        if exp_r != 1.0:
            self.exp = conv2d(w_in, w_exp, kernel, stride=stride)
            self.exp_bn = norm2d(norm_name, w_exp)
            self.exp_act = activation(act_name)
            w_mid = w_exp
        # SE
        if se_r > 0.0:
            w_se = int(w_in * se_r)
            self.se = SE(w_mid, w_se, act_name)
        # PWConv
        self.proj = conv2d(
            w_mid, w_out, 1 if exp_r != 1.0 else kernel, stride=1 if exp_r != 1.0 else stride
        )
        self.proj_bn = norm2d(norm_name, w_out)
        self.has_proj_act = exp_r == 1.0
        if self.has_proj_act:
            self.proj_act = activation(act_name)
        # Skip
        self.has_skip = has_skip and w_in == w_out and stride == 1
        if self.has_skip:
            self.drop_path = DropPath(drop_path_prob)

    def forward(self, x):
        x_p = x
        if getattr(self, "exp", None) is not None:
            x = self.exp(x)
            x = self.exp_bn(x)
            x = self.exp_act(x)
        if getattr(self, "se", None) is not None:
            x = self.se(x)
        x = self.proj(x)
        x = self.proj_bn(x)
        if self.has_proj_act:
            x = self.proj_act(x)
        if self.has_skip:
            x = self.drop_path(x)
            x = x + x_p
        return x


@registers.models.register()
class EffNet(M.Module):
    """EfficientNet model.

    Args:
        stem_w: stem width.
        block_name: block name.
        depths: depth for each stage (number of blocks in the stage).
        widths: width for each stage (width of each block in the stage).
        strides: strides for each stage (applies to the first block of each stage).
        kernels: kernel sizes for each stage.
        exp_rs: expansion ratios for MBConv blocks in each stage.
        se_r: Squeeze-and-Excitation (SE) ratio. Default: ``0.25``
        drop_path_prob: drop path probability. Default: ``0.0``
        depth_mult: depth multiplier. Default: ``1.0``
        width_mult: width multiplier. Default: ``1.0``
        omit_mult: omit multiplier for stem width, head width, the first stage depth and
            the last stage depth, enabled in EfficientNet-Lite. Default: ``False``
        norm_name: normalization function. Default: ``"BN"``
        act_name: activation function. Default: ``"silu"``
        head: head args. Default: ``None``
    """

    def __init__(
        self,
        stem_w: int,
        block_name: Union[Union[str, Callable], Sequence[Union[str, Callable]]],
        depths: Sequence[int],
        widths: Sequence[int],
        strides: Sequence[int],
        kernels: Sequence[int],
        exp_rs: Union[float, Sequence[Union[float, Sequence[float]]]] = 1.0,
        se_rs: Union[float, Sequence[Union[float, Sequence[float]]]] = 0.0,
        drop_path_prob: float = 0.0,
        depth_mult: float = 1.0,
        width_mult: float = 1.0,
        omit_mult: bool = False,
        norm_name: str = "BN",
        act_name: str = "silu",
        head: Mapping[str, Any] = None,
    ):
        super().__init__()
        depths = [
            d if omit_mult and i in (0, len(depths) - 1) else math.ceil(d * depth_mult)
            for i, d in enumerate(depths)
        ]
        self.depths = depths

        stem_w = stem_w if omit_mult else make_divisible(stem_w * width_mult, round_limit=0.9)
        self.stem = SimpleStem(3, stem_w, norm_name, act_name)

        if isinstance(block_name, (str, Callable)):
            block_name = [block_name] * len(depths)
        block_func = [self.get_block_func(bn) for bn in block_name]
        widths = [make_divisible(w * width_mult, round_limit=0.9) for w in widths]
        if isinstance(exp_rs, Real):
            exp_rs = [exp_rs] * len(depths)
        if isinstance(se_rs, Real):
            se_rs = [se_rs] * len(depths)
        drop_path_prob_iter = (i / sum(depths) * drop_path_prob for i in range(sum(depths)))
        drop_path_probs = [[next(drop_path_prob_iter) for _ in range(d)] for d in depths]
        model_args = [depths, widths, strides, block_func, kernels, exp_rs, se_rs, drop_path_probs]
        prev_w = stem_w
        for i, (d, w, s, bf, k, exp_r, se_r, dp_p) in enumerate(zip(*model_args)):
            stage = AnyStage(
                prev_w,
                w,
                s,
                d,
                bf,
                kernel=k,
                exp_r=exp_r,
                se_r=se_r,
                se_from_exp=False,
                se_act_name=act_name,
                se_approx=False,
                se_rd_fn=int,
                has_proj_act=False,
                has_skip=True,
                drop_path_prob=dp_p,
                norm_name=norm_name,
                act_name=act_name,
            )
            setattr(self, f"s{i + 1}", stage)
            prev_w = w

        if head:
            if head.get("width", 0) > 0 and not omit_mult:
                head["width"] = make_divisible(head["width"] * width_mult, round_limit=0.9)
            self.head = build_head(prev_w, head, norm_name, act_name)

        self.apply(init_weights)

    def forward(self, x):
        x = self.stem(x)
        for i in range(len(self.depths)):
            stage = getattr(self, f"s{i + 1}")
            x = stage(x)
        if getattr(self, "head", None) is not None:
            x = self.head(x)
        return x

    @staticmethod
    def get_block_func(name: Union[str, Callable]):
        """Retrieves the block function by name."""
        if callable(name):
            return name
        if isinstance(name, str):
            block_funcs = {
                "FuseMBConv": FuseMBConv,
                "MBConv": MBConv,
            }
            if name in block_funcs.keys():
                return block_funcs[name]
        raise ValueError(f"Block '{name}' not supported")


def _build_effnet(**kwargs):
    model_args = dict(
        stem_w=32,
        block_name=MBConv,
        depths=[1, 2, 2, 3, 3, 4, 1],
        widths=[16, 24, 40, 80, 112, 192, 320],
        strides=[1, 2, 2, 2, 1, 2, 1],
        kernels=[3, 3, 5, 3, 5, 5, 3],
        exp_rs=[1, 6, 6, 6, 6, 6, 6],
        se_rs=0.25,
        drop_path_prob=0.2,
        head=dict(name="ClsHead", width=1280, dropout_prob=0.2),
    )
    recursive_update(model_args, kwargs)
    return EffNet(**model_args)


def _build_effnet_lite(**kwargs):
    model_args = dict(se_rs=0.0, omit_mult=True, act_name="relu6")
    recursive_update(model_args, kwargs)
    return _build_effnet(**model_args)


def _build_effnetv2(**kwargs):
    model_args = dict(
        stem_w=32,
        block_name=[FuseMBConv, FuseMBConv, FuseMBConv, MBConv, MBConv, MBConv],
        depths=[1, 2, 2, 3, 5, 8],
        widths=[16, 32, 48, 96, 112, 192],
        strides=[1, 2, 2, 2, 1, 2],
        kernels=[3, 3, 3, 3, 3, 3],
        exp_rs=[1, 4, 4, 4, 6, 6],
        se_rs=[0, 0, 0, 0.25, 0.25, 0.25],
    )
    recursive_update(model_args, kwargs)
    return _build_effnet(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/effnet/effnet_b0/effnet_b0.pkl"
)
def effnet_b0(**kwargs):
    model_args = dict(depth_mult=1.0, width_mult=1.0)
    recursive_update(model_args, kwargs)
    return _build_effnet(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/effnet/effnet_b1/effnet_b1.pkl"
)
def effnet_b1(**kwargs):
    model_args = dict(depth_mult=1.1, width_mult=1.0)
    recursive_update(model_args, kwargs)
    return _build_effnet(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/effnet/effnet_b2/effnet_b2.pkl"
)
def effnet_b2(**kwargs):
    model_args = dict(depth_mult=1.2, width_mult=1.1, head=dict(dropout_prob=0.3))
    recursive_update(model_args, kwargs)
    return _build_effnet(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/effnet/effnet_b3/effnet_b3.pkl"
)
def effnet_b3(**kwargs):
    model_args = dict(depth_mult=1.4, width_mult=1.2, head=dict(dropout_prob=0.3))
    recursive_update(model_args, kwargs)
    return _build_effnet(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/effnet/effnet_b4/effnet_b4.pkl"
)
def effnet_b4(**kwargs):
    model_args = dict(depth_mult=1.8, width_mult=1.4, head=dict(dropout_prob=0.4))
    recursive_update(model_args, kwargs)
    return _build_effnet(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/effnet/effnet_b5/effnet_b5.pkl"
)
def effnet_b5(**kwargs):
    model_args = dict(depth_mult=2.2, width_mult=1.6, head=dict(dropout_prob=0.4))
    recursive_update(model_args, kwargs)
    return _build_effnet(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/effnet/effnet_b6/effnet_b6.pkl"
)
def effnet_b6(**kwargs):
    model_args = dict(depth_mult=2.6, width_mult=1.8, head=dict(dropout_prob=0.5))
    recursive_update(model_args, kwargs)
    return _build_effnet(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/effnet/effnet_b7/effnet_b7.pkl"
)
def effnet_b7(**kwargs):
    model_args = dict(depth_mult=3.1, width_mult=2.0, head=dict(dropout_prob=0.5))
    recursive_update(model_args, kwargs)
    return _build_effnet(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/effnet/effnet_b8/effnet_b8.pkl"
)
def effnet_b8(**kwargs):
    model_args = dict(depth_mult=3.6, width_mult=2.2, head=dict(dropout_prob=0.5))
    recursive_update(model_args, kwargs)
    return _build_effnet(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/effnet/effnet_l2/effnet_l2.pkl"
)
def effnet_l2(**kwargs):
    model_args = dict(depth_mult=5.3, width_mult=4.3, head=dict(dropout_prob=0.5))
    recursive_update(model_args, kwargs)
    return _build_effnet(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/effnet/effnet_b0_lite/effnet_b0_lite.pkl"
)
def effnet_b0_lite(**kwargs):
    model_args = dict(depth_mult=1.0, width_mult=1.0)
    recursive_update(model_args, kwargs)
    return _build_effnet_lite(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/effnet/effnet_b1_lite/effnet_b1_lite.pkl"
)
def effnet_b1_lite(**kwargs):
    model_args = dict(depth_mult=1.1, width_mult=1.0)
    recursive_update(model_args, kwargs)
    return _build_effnet_lite(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/effnet/effnet_b2_lite/effnet_b2_lite.pkl"
)
def effnet_b2_lite(**kwargs):
    model_args = dict(depth_mult=1.2, width_mult=1.1, head=dict(dropout_prob=0.3))
    recursive_update(model_args, kwargs)
    return _build_effnet_lite(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/effnet/effnet_b3_lite/effnet_b3_lite.pkl"
)
def effnet_b3_lite(**kwargs):
    model_args = dict(depth_mult=1.4, width_mult=1.2, head=dict(dropout_prob=0.3))
    recursive_update(model_args, kwargs)
    return _build_effnet_lite(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/effnet/effnet_b4_lite/effnet_b4_lite.pkl"
)
def effnet_b4_lite(**kwargs):
    model_args = dict(depth_mult=1.8, width_mult=1.4, head=dict(dropout_prob=0.3))
    recursive_update(model_args, kwargs)
    return _build_effnet_lite(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/effnet/effnetv2_b0/effnetv2_b0.pkl"
)
def effnetv2_b0(**kwargs):
    model_args = dict(depth_mult=1.0, width_mult=1.0)
    recursive_update(model_args, kwargs)
    return _build_effnetv2(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/effnet/effnetv2_b1/effnetv2_b1.pkl"
)
def effnetv2_b1(**kwargs):
    model_args = dict(depth_mult=1.1, width_mult=1.0)
    recursive_update(model_args, kwargs)
    return _build_effnetv2(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/effnet/effnetv2_b2/effnetv2_b2.pkl"
)
def effnetv2_b2(**kwargs):
    model_args = dict(depth_mult=1.2, width_mult=1.1, head=dict(dropout_prob=0.3))
    recursive_update(model_args, kwargs)
    return _build_effnetv2(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/effnet/effnetv2_b3/effnetv2_b3.pkl"
)
def effnetv2_b3(**kwargs):
    model_args = dict(depth_mult=1.4, width_mult=1.2, head=dict(dropout_prob=0.3))
    recursive_update(model_args, kwargs)
    return _build_effnetv2(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/effnet/effnetv2_s/effnetv2_s.pkl"
)
def effnetv2_s(**kwargs):
    model_args = dict(stem_w=24, depths=[2, 4, 4, 6, 9, 15], widths=[24, 48, 64, 128, 160, 256])
    recursive_update(model_args, kwargs)
    return _build_effnetv2(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/effnet/effnetv2_m/effnetv2_m.pkl"
)
def effnetv2_m(**kwargs):
    model_args = dict(
        stem_w=24,
        block_name=[FuseMBConv, FuseMBConv, FuseMBConv, MBConv, MBConv, MBConv, MBConv],
        depths=[3, 5, 5, 7, 14, 18, 5],
        widths=[24, 48, 80, 160, 176, 304, 512],
        strides=[1, 2, 2, 2, 1, 2, 1],
        kernels=[3, 3, 3, 3, 3, 3, 3],
        exp_rs=[1, 4, 4, 4, 6, 6, 6],
        se_rs=[0, 0, 0, 0.25, 0.25, 0.25, 0.25],
        head=dict(dropout_prob=0.3),
    )
    recursive_update(model_args, kwargs)
    return _build_effnetv2(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/effnet/effnetv2_l/effnetv2_l.pkl"
)
def effnetv2_l(**kwargs):
    model_args = dict(
        stem_w=32,
        block_name=[FuseMBConv, FuseMBConv, FuseMBConv, MBConv, MBConv, MBConv, MBConv],
        depths=[4, 7, 7, 10, 19, 25, 7],
        widths=[32, 64, 96, 192, 224, 384, 640],
        strides=[1, 2, 2, 2, 1, 2, 1],
        kernels=[3, 3, 3, 3, 3, 3, 3],
        exp_rs=[1, 4, 4, 4, 6, 6, 6],
        se_rs=[0, 0, 0, 0.25, 0.25, 0.25, 0.25],
        head=dict(dropout_prob=0.4),
    )
    recursive_update(model_args, kwargs)
    return _build_effnetv2(**model_args)
