#!/usr/bin/env python3
# Copyright (c) 2020 Ross Wightman
# This file has been modified by Megvii ("Megvii Modifications").
# All Megvii Modifications are Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
"""MobileNet Series

MobileNetV1: `"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
<https://arxiv.org/abs/1704.04861>`_

MobileNetV2: `"MobileNetV2: Inverted Residuals and Linear Bottlenecks"
<https://arxiv.org/abs/1801.04381>`_

MobileNetV3: `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_

References:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/efficientnet.py
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/mobilenetv3.py
"""
from functools import partial
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

from .resnet import SimpleStem

__all__ = ["MBConv", "MBStage", "MBNet"]


class MBConv(M.Module):
    """Mobile inverted bottleneck block with SE.

    ======= ================ =================== ==== ============== ======= ======
    Version    Expansion           DWConv         SE      PWConv     OutAct   Skip
    ======= ================ =================== ==== ============== ======= ======
    basic   [EXP, BN,    AF] [kxk_DW, BN,    AF] [SE] [1x1_Conv, BN] [   AF] [Skip]
    V1                       [3x3_DW, BN, ReLU6]      [1x1_Conv, BN] [ReLU6]
    V2      [EXP, BN, ReLU6] [3x3_DW, BN, ReLU6]      [1x1_Conv, BN]         [Skip]
    V3      [EXP, BN,    AF] [kxk_DW, BN,    AF] [SE] [1x1_Conv, BN]         [Skip]
    ======= ================ =================== ==== ============== ======= ======

    Args:
        w_in: input width.
        w_out: output width.
        stride: stride of depthwise conv.
        kernel: kernel of depthwise conv.
        exp_r: expansion ratio.
        se_r: SE ratio.
        se_from_exp: calculate SE channels from expanded (mid) channels.
        se_act_name: activation function for SE.
        se_approx: whether approximated sigmoid function (HSigmoid).
        se_rd_fn: SE round channel function.
        has_proj_act: whether apply activation to output.
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
        se_from_exp: bool,
        se_act_name: str,
        se_approx: bool,
        se_rd_fn: Callable,
        has_proj_act: bool,
        has_skip: bool,
        drop_path_prob: float,
        norm_name: str,
        act_name: str,
    ):
        super().__init__()
        # Expansion
        w_mid = w_in
        w_exp = make_divisible(w_in * exp_r, round_limit=0.9)
        if exp_r != 1.0:
            self.exp = conv2d(w_in, w_exp, 1)
            self.exp_bn = norm2d(norm_name, w_exp)
            self.exp_act = activation(act_name)
            w_mid = w_exp
        # DWConv
        self.dwise = conv2d(w_mid, w_mid, kernel, stride=stride, groups=w_mid)
        self.dwise_bn = norm2d(norm_name, w_mid)
        self.dwise_act = activation(act_name)
        # SE
        if se_r > 0.0:
            se_rd_fn = se_rd_fn or int
            w_se = se_rd_fn((w_mid if se_from_exp else w_in) * se_r)
            self.se = SE(w_mid, w_se, se_act_name, se_approx)
        # PWConv
        self.proj = conv2d(w_mid, w_out, 1)
        self.proj_bn = norm2d(norm_name, w_out)
        self.has_proj_act = has_proj_act
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
        x = self.dwise(x)
        x = self.dwise_bn(x)
        x = self.dwise_act(x)
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


class MBStage(M.Module):
    """MBNet stage (sequence of blocks w/ the same output shape)."""

    def __init__(
        self,
        w_in: int,
        w_out: int,
        stride: int,
        depth: int,
        exp_r: Union[float, Sequence[float]],
        drop_path_prob: Sequence[float],
        **kwargs,
    ):
        super().__init__()
        self.depth = depth
        if isinstance(exp_r, Real):
            exp_r = [exp_r] * depth
        for i in range(depth):
            block = MBConv(
                w_in, w_out, stride, exp_r=exp_r[i], drop_path_prob=drop_path_prob[i], **kwargs
            )
            setattr(self, f"b{i + 1}", block)
            stride, w_in = 1, w_out

    def __len__(self):
        return self.depth

    def forward(self, x):
        for i in range(self.depth):
            block = getattr(self, f"b{i + 1}")
            x = block(x)
        return x


@registers.models.register()
class MBNet(M.Module):
    """MobileNet model.

    Args:
        stem_w: stem width.
        depths: depth for each stage (number of blocks in the stage).
        widths: width for each stage (width of each block in the stage).
        strides: strides for each stage (applies to the first block of each stage).
        kernels: kernel sizes for each stage.
        exp_rs: expansion ratios for MobileNet basic blocks in each stage. Default: ``1.0``
        se_rs: Squeeze-and-Excitation (SE) ratios. Default: ``0.0``
        stage_act_names: activation function for stages. Default: ``None``
        has_proj_act: whether apply activation to output. Default: ``False``
        has_skip: whether apply skip connection. Default: ``True``
        drop_path_prob: drop path probability. Default: ``0.0``
        width_mult: width multiplier. Default: ``1.0``
        norm_name: normalization function. Default: ``"BN"``
        act_name: activation function. Default: ``"relu6"``
        head: head args. Default: ``None``
    """

    def __init__(
        self,
        stem_w: int,
        depths: Sequence[int],
        widths: Sequence[int],
        strides: Sequence[int],
        kernels: Sequence[int],
        exp_rs: Union[float, Sequence[Union[float, Sequence[float]]]] = 1.0,
        se_rs: Union[float, Sequence[Union[float, Sequence[float]]]] = 0.0,
        stage_act_names: Sequence[str] = None,
        has_proj_act: bool = False,
        has_skip: bool = True,
        drop_path_prob: float = 0.0,
        width_mult: float = 1.0,
        norm_name: str = "BN",
        act_name: str = "relu6",
        head: Mapping[str, Any] = None,
    ):
        super().__init__()
        self.depths = depths

        stem_w = make_divisible(stem_w * width_mult, round_limit=0.9)
        self.stem = SimpleStem(3, stem_w, norm_name, act_name)

        widths = [make_divisible(w * width_mult, round_limit=0.9) for w in widths]
        if isinstance(exp_rs, Real):
            exp_rs = [exp_rs] * len(depths)
        if isinstance(se_rs, Real):
            se_rs = [se_rs] * len(depths)
        drop_path_prob_iter = (i / sum(depths) * drop_path_prob for i in range(sum(depths)))
        drop_path_probs = [[next(drop_path_prob_iter) for _ in range(d)] for d in depths]
        if stage_act_names is None:
            stage_act_names = [act_name] * len(depths)
        model_args = (
            depths,
            widths,
            strides,
            kernels,
            exp_rs,
            se_rs,
            drop_path_probs,
            stage_act_names,
        )
        prev_w = stem_w
        for i, (d, w, s, k, exp_r, se_r, dp_p, act) in enumerate(zip(*model_args)):
            stage = MBStage(
                prev_w,
                w,
                s,
                d,
                kernel=k,
                exp_r=exp_r,
                se_r=se_r,
                se_from_exp=True,
                se_act_name="relu",
                se_approx=True,
                se_rd_fn=partial(make_divisible, round_limit=0.9),
                has_proj_act=has_proj_act,
                has_skip=has_skip,
                drop_path_prob=dp_p,
                norm_name=norm_name,
                act_name=act,
            )
            setattr(self, f"s{i + 1}", stage)
            prev_w = w

        if head:
            if head.get("width", 0) > 0:
                head["width"] = make_divisible(head["width"] * max(1, width_mult), round_limit=0.9)
            if "w_h" in head:  # MBV3Head
                head["w_h"] = make_divisible(head["w_h"] * width_mult, round_limit=0.9)
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


def _build_mbnetv1(**kwargs):
    model_args = dict(
        stem_w=32,
        depths=[1, 2, 2, 6, 2],
        widths=[64, 128, 256, 512, 1024],
        strides=[1, 2, 2, 2, 2],
        kernels=[3, 3, 3, 3, 3],
        has_proj_act=True,
        has_skip=False,
        head=dict(name="ClsHead", dropout_prob=0.2),
    )
    recursive_update(model_args, kwargs)
    return MBNet(**model_args)


def _build_mbnetv2(**kwargs):
    model_args = dict(
        stem_w=32,
        depths=[1, 2, 3, 4, 3, 3, 1],
        widths=[16, 24, 32, 64, 96, 160, 320],
        strides=[1, 2, 2, 2, 1, 2, 1],
        kernels=[3, 3, 3, 3, 3, 3, 3],
        exp_rs=[1, 6, 6, 6, 6, 6, 6],
        head=dict(name="ClsHead", width=1280, dropout_prob=0.2),
    )
    recursive_update(model_args, kwargs)
    return MBNet(**model_args)


def _build_mbnetv3_small(**kwargs):
    model_args = dict(
        stem_w=16,
        depths=[1, 2, 3, 2, 3],
        widths=[16, 24, 40, 48, 96],
        strides=[2, 2, 2, 1, 2],
        kernels=[3, 3, 5, 5, 5],
        exp_rs=[1, [4.5, 3.67], [4, 6, 6], 3, 6],
        se_rs=[0.25, 0, 0.25, 0.25, 0.25],
        stage_act_names=["relu", "relu", "hswish", "hswish", "hswish"],
        act_name="hswish",
        head=dict(name="MBV3Head", width=576, w_h=1024, dropout_prob=0.2),
    )
    recursive_update(model_args, kwargs)
    return MBNet(**model_args)


def _build_mbnetv3_large(**kwargs):
    model_args = dict(
        stem_w=16,
        depths=[1, 2, 3, 4, 2, 3],
        widths=[16, 24, 40, 80, 112, 160],
        strides=[1, 2, 2, 2, 1, 2],
        kernels=[3, 3, 5, 3, 3, 5],
        exp_rs=[1, [4, 3], 3, [6, 2.5, 2.3, 2.3], 6, 6],
        se_rs=[0, 0, 0.25, 0, 0.25, 0.25],
        stage_act_names=["relu", "relu", "relu", "hswish", "hswish", "hswish"],
        act_name="hswish",
        head=dict(name="MBV3Head", width=960, w_h=1280, dropout_prob=0.2),
    )
    recursive_update(model_args, kwargs)
    return MBNet(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/mbnet/mbnetv1_x025/mbnetv1_x025.pkl"
)
def mbnetv1_x025(**kwargs):
    model_args = dict(width_mult=0.25)
    recursive_update(model_args, kwargs)
    return _build_mbnetv1(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/mbnet/mbnetv1_x050/mbnetv1_x050.pkl"
)
def mbnetv1_x050(**kwargs):
    model_args = dict(width_mult=0.5)
    recursive_update(model_args, kwargs)
    return _build_mbnetv1(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/mbnet/mbnetv1_x075/mbnetv1_x075.pkl"
)
def mbnetv1_x075(**kwargs):
    model_args = dict(width_mult=0.75)
    recursive_update(model_args, kwargs)
    return _build_mbnetv1(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/mbnet/mbnetv1_x100/mbnetv1_x100.pkl"
)
def mbnetv1_x100(**kwargs):
    model_args = dict(width_mult=1.0)
    recursive_update(model_args, kwargs)
    return _build_mbnetv1(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/mbnet/mbnetv2_x035/mbnetv2_x035.pkl"
)
def mbnetv2_x035(**kwargs):
    model_args = dict(width_mult=0.35)
    recursive_update(model_args, kwargs)
    return _build_mbnetv2(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/mbnet/mbnetv2_x050/mbnetv2_x050.pkl"
)
def mbnetv2_x050(**kwargs):
    model_args = dict(width_mult=0.5)
    recursive_update(model_args, kwargs)
    return _build_mbnetv2(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/mbnet/mbnetv2_x075/mbnetv2_x075.pkl"
)
def mbnetv2_x075(**kwargs):
    model_args = dict(width_mult=0.75)
    recursive_update(model_args, kwargs)
    return _build_mbnetv2(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/mbnet/mbnetv2_x100/mbnetv2_x100.pkl"
)
def mbnetv2_x100(**kwargs):
    model_args = dict(width_mult=1.0)
    recursive_update(model_args, kwargs)
    return _build_mbnetv2(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/mbnet/mbnetv2_x140/mbnetv2_x140.pkl"
)
def mbnetv2_x140(**kwargs):
    model_args = dict(width_mult=1.4)
    recursive_update(model_args, kwargs)
    return _build_mbnetv2(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/"
    "mbnet/mbnetv3_small_x075/mbnetv3_small_x075.pkl"
)
def mbnetv3_small_x075(**kwargs):
    model_args = dict(width_mult=0.75)
    recursive_update(model_args, kwargs)
    return _build_mbnetv3_small(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/"
    "mbnet/mbnetv3_small_x100/mbnetv3_small_x100.pkl"
)
def mbnetv3_small_x100(**kwargs):
    model_args = dict(width_mult=1.0)
    recursive_update(model_args, kwargs)
    return _build_mbnetv3_small(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/"
    "mbnet/mbnetv3_large_x075/mbnetv3_large_x075.pkl"
)
def mbnetv3_large_x075(**kwargs):
    model_args = dict(width_mult=0.75)
    recursive_update(model_args, kwargs)
    return _build_mbnetv3_large(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/"
    "mbnet/mbnetv3_large_x100/mbnetv3_large_x100.pkl"
)
def mbnetv3_large_x100(**kwargs):
    model_args = dict(width_mult=1.0)
    recursive_update(model_args, kwargs)
    return _build_mbnetv3_large(**model_args)
