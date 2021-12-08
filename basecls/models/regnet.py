#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
"""RegNet Series

RegNet X/Y: `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_
"""
from typing import Any, Callable, Mapping, Tuple, Union

import megengine.hub as hub
import megengine.module as M
import numpy as np

from basecls.layers import SE, activation, adjust_block_compatibility, conv2d, norm2d
from basecls.utils import recursive_update, registers

from .resnet import ResNet, SimpleStem

__all__ = ["RegBottleneckBlock", "RegNet"]


class RegBottleneckBlock(M.Module):
    """Residual bottleneck block for RegNet: x + f(x), f = 1x1, 3x3 [+SE], 1x1."""

    def __init__(
        self,
        w_in: int,
        w_out: int,
        stride: int,
        bot_mul: float,
        group_w: int,
        se_r: float,
        norm_name: str,
        act_name: str,
        **kwargs,
    ):
        super().__init__()
        if w_in != w_out or stride > 1:
            self.proj = conv2d(w_in, w_out, 1, stride=stride)
            self.bn = norm2d(norm_name, w_out)

        w_b = int(round(w_out * bot_mul))
        w_se = int(round(w_in * se_r))
        groups = w_b // group_w
        self.a = conv2d(w_in, w_b, 1)
        self.a_bn = norm2d(norm_name, w_b)
        self.a_act = activation(act_name)
        self.b = conv2d(w_b, w_b, 3, stride=stride, groups=groups)
        self.b_bn = norm2d(norm_name, w_b)
        self.b_act = activation(act_name)
        if w_se > 0:
            self.se = SE(w_b, w_se, act_name)
        self.c = conv2d(w_b, w_out, 1)
        self.c_bn = norm2d(norm_name, w_out)
        self.c_bn.final_bn = True
        self.act = activation(act_name)

    def forward(self, x):
        x_p = x
        if getattr(self, "proj", None) is not None:
            x_p = self.proj(x_p)
            x_p = self.bn(x_p)

        x = self.a(x)
        x = self.a_bn(x)
        x = self.a_act(x)
        x = self.b(x)
        x = self.b_bn(x)
        x = self.b_act(x)
        if getattr(self, "se", None) is not None:
            x = self.se(x)
        x = self.c(x)
        x = self.c_bn(x)
        x += x_p
        x = self.act(x)
        return x


def generate_regnet(w_a: float, w_0: int, w_m: float, d: int, q: int = 8) -> Tuple:
    """Generates per stage widths and depths from RegNet parameters."""
    assert w_a >= 0 and w_0 > 0 and w_m > 1 and w_0 % q == 0
    # Generate continuous per-block ws
    ws_cont = np.arange(d) * w_a + w_0
    # Generate quantized per-block ws
    ks = np.round(np.log(ws_cont / w_0) / np.log(w_m))
    ws_all = w_0 * np.power(w_m, ks)
    ws_all = np.round(np.divide(ws_all, q)).astype(int) * q
    # Generate per stage ws and ds (assumes ws_all are sorted)
    ws, ds = np.unique(ws_all, return_counts=True)
    # Compute number of actual stages and total possible stages
    num_stages, total_stages = len(ws), ks.max() + 1
    # Convert numpy arrays to lists and return
    ws, ds, ws_all, ws_cont = (x.tolist() for x in (ws, ds, ws_all, ws_cont))
    return ws, ds, num_stages, total_stages, ws_all, ws_cont


@registers.models.register()
class RegNet(ResNet):
    """RegNet model.

    Args:
        stem_name: stem name.
        stem_w: stem width.
        block_name: block name.
        depth: depth.
        w0: initial width.
        wa: slope.
        wm: quantization.
        group_w: group width for each stage (applies to bottleneck block).
        stride: stride for each stage (applies to the first block of each stage). Default: ``2``
        bot_mul: bottleneck multiplier for each stage (applies to bottleneck block).
            Default: ``1.0``
        se_r: Squeeze-and-Excitation (SE) ratio. Default: ``0.0``
        drop_path_prob: drop path probability. Default: ``0.0``
        zero_init_final_gamma: enable zero-initialize or not. Default: ``False``
        norm_name: normalization function. Default: ``"BN"``
        act_name: activation function. Default: ``"relu"``
        head: head args. Default: ``None``
    """

    def __init__(
        self,
        stem_name: Union[str, Callable],
        stem_w: int,
        block_name: Union[str, Callable],
        depth: int,
        w0: int,
        wa: float,
        wm: float,
        group_w: int,
        stride: int = 2,
        bot_mul: float = 1.0,
        se_r: float = 0.0,
        drop_path_prob: float = 0.0,
        zero_init_final_gamma: bool = False,
        norm_name: str = "BN",
        act_name: str = "relu",
        head: Mapping[str, Any] = None,
    ):
        # Generates per stage widths, depths, strides, bot_muls and group_ws from RegNet parameters
        widths, depths = generate_regnet(wa, w0, wm, depth)[0:2]
        strides = [stride] * len(widths)
        bot_muls = [bot_mul] * len(widths)
        group_ws = [group_w] * len(widths)
        widths, bot_muls, group_ws = adjust_block_compatibility(widths, bot_muls, group_ws)

        super().__init__(
            stem_name,
            stem_w,
            block_name,
            depths,
            widths,
            strides,
            bot_muls,
            group_ws,
            se_r,
            False,
            drop_path_prob,
            zero_init_final_gamma,
            norm_name,
            act_name,
            head,
        )


def _build_regnetx(**kwargs):
    model_args = dict(
        stem_name=SimpleStem,
        stem_w=32,
        block_name=RegBottleneckBlock,
        head=dict(name="ClsHead"),
    )
    recursive_update(model_args, kwargs)
    return RegNet(**model_args)


def _build_regnety(**kwargs):
    model_args = dict(se_r=0.25)
    recursive_update(model_args, kwargs)
    return _build_regnetx(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/regnet/regnetx_002/regnetx_002.pkl"
)
def regnetx_002(**kwargs):
    model_args = dict(depth=13, w0=24, wa=36.44, wm=2.49, group_w=8)
    recursive_update(model_args, kwargs)
    return _build_regnetx(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/regnet/regnetx_004/regnetx_004.pkl"
)
def regnetx_004(**kwargs):
    model_args = dict(depth=22, w0=24, wa=24.48, wm=2.54, group_w=16)
    recursive_update(model_args, kwargs)
    return _build_regnetx(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/regnet/regnetx_006/regnetx_006.pkl"
)
def regnetx_006(**kwargs):
    model_args = dict(depth=16, w0=48, wa=36.97, wm=2.24, group_w=24)
    recursive_update(model_args, kwargs)
    return _build_regnetx(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/regnet/regnetx_008/regnetx_008.pkl"
)
def regnetx_008(**kwargs):
    model_args = dict(depth=16, w0=56, wa=35.73, wm=2.28, group_w=16)
    recursive_update(model_args, kwargs)
    return _build_regnetx(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/regnet/regnetx_016/regnetx_016.pkl"
)
def regnetx_016(**kwargs):
    model_args = dict(depth=18, w0=80, wa=34.01, wm=2.25, group_w=24)
    recursive_update(model_args, kwargs)
    return _build_regnetx(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/regnet/regnetx_032/regnetx_032.pkl"
)
def regnetx_032(**kwargs):
    model_args = dict(depth=25, w0=88, wa=26.31, wm=2.25, group_w=48)
    recursive_update(model_args, kwargs)
    return _build_regnetx(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/regnet/regnetx_040/regnetx_040.pkl"
)
def regnetx_040(**kwargs):
    model_args = dict(depth=23, w0=96, wa=38.65, wm=2.43, group_w=40)
    recursive_update(model_args, kwargs)
    return _build_regnetx(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/regnet/regnetx_064/regnetx_064.pkl"
)
def regnetx_064(**kwargs):
    model_args = dict(depth=17, w0=184, wa=60.83, wm=2.07, group_w=56)
    recursive_update(model_args, kwargs)
    return _build_regnetx(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/regnet/regnetx_080/regnetx_080.pkl"
)
def regnetx_080(**kwargs):
    model_args = dict(depth=23, w0=80, wa=49.56, wm=2.88, group_w=120)
    recursive_update(model_args, kwargs)
    return _build_regnetx(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/regnet/regnetx_120/regnetx_120.pkl"
)
def regnetx_120(**kwargs):
    model_args = dict(depth=19, w0=168, wa=73.36, wm=2.37, group_w=112)
    recursive_update(model_args, kwargs)
    return _build_regnetx(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/regnet/regnetx_160/regnetx_160.pkl"
)
def regnetx_160(**kwargs):
    model_args = dict(depth=22, w0=216, wa=55.59, wm=2.1, group_w=128)
    recursive_update(model_args, kwargs)
    return _build_regnetx(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/regnet/regnetx_320/regnetx_320.pkl"
)
def regnetx_320(**kwargs):
    model_args = dict(depth=23, w0=320, wa=69.86, wm=2.0, group_w=168)
    recursive_update(model_args, kwargs)
    return _build_regnetx(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/regnet/regnety_002/regnety_002.pkl"
)
def regnety_002(**kwargs):
    model_args = dict(depth=13, w0=24, wa=36.44, wm=2.49, group_w=8)
    recursive_update(model_args, kwargs)
    return _build_regnety(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/regnet/regnety_004/regnety_004.pkl"
)
def regnety_004(**kwargs):
    model_args = dict(depth=16, w0=48, wa=27.89, wm=2.09, group_w=8)
    recursive_update(model_args, kwargs)
    return _build_regnety(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/regnet/regnety_006/regnety_006.pkl"
)
def regnety_006(**kwargs):
    model_args = dict(depth=15, w0=48, wa=32.54, wm=2.32, group_w=16)
    recursive_update(model_args, kwargs)
    return _build_regnety(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/regnet/regnety_008/regnety_008.pkl"
)
def regnety_008(**kwargs):
    model_args = dict(depth=14, w0=56, wa=38.84, wm=2.4, group_w=16)
    recursive_update(model_args, kwargs)
    return _build_regnety(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/regnet/regnety_016/regnety_016.pkl"
)
def regnety_016(**kwargs):
    model_args = dict(depth=27, w0=48, wa=20.71, wm=2.65, group_w=24)
    recursive_update(model_args, kwargs)
    return _build_regnety(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/regnet/regnety_032/regnety_032.pkl"
)
def regnety_032(**kwargs):
    model_args = dict(depth=21, w0=80, wa=42.63, wm=2.66, group_w=24)
    recursive_update(model_args, kwargs)
    return _build_regnety(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/regnet/regnety_040/regnety_040.pkl"
)
def regnety_040(**kwargs):
    model_args = dict(depth=22, w0=96, wa=31.41, wm=2.24, group_w=64)
    recursive_update(model_args, kwargs)
    return _build_regnety(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/regnet/regnety_064/regnety_064.pkl"
)
def regnety_064(**kwargs):
    model_args = dict(depth=25, w0=112, wa=33.22, wm=2.27, group_w=72)
    recursive_update(model_args, kwargs)
    return _build_regnety(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/regnet/regnety_080/regnety_080.pkl"
)
def regnety_080(**kwargs):
    model_args = dict(depth=17, w0=192, wa=76.82, wm=2.19, group_w=56)
    recursive_update(model_args, kwargs)
    return _build_regnety(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/regnet/regnety_120/regnety_120.pkl"
)
def regnety_120(**kwargs):
    model_args = dict(depth=19, w0=168, wa=73.36, wm=2.37, group_w=112)
    recursive_update(model_args, kwargs)
    return _build_regnety(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/regnet/regnety_160/regnety_160.pkl"
)
def regnety_160(**kwargs):
    model_args = dict(depth=18, w0=200, wa=106.23, wm=2.48, group_w=112)
    recursive_update(model_args, kwargs)
    return _build_regnety(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/regnet/regnety_320/regnety_320.pkl"
)
def regnety_320(**kwargs):
    model_args = dict(depth=20, w0=232, wa=115.89, wm=2.53, group_w=232)
    recursive_update(model_args, kwargs)
    return _build_regnety(**model_args)
