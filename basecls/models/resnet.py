#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
"""ResNet Series

ResNet: `"Deep Residual Learning for Image Recognition" <https://arxiv.org/abs/1512.03385>`_

ResNet-D: `"Bag of Tricks for Image Classification with Convolutional Neural Networks"
<https://arxiv.org/abs/1812.01187>`_

ResNeXt: `"Aggregated Residual Transformations for Deep Neural Networks"
<https://arxiv.org/abs/1611.05431>`_

Se-ResNet: `"Squeeze-and-Excitation Networks" <https://arxiv.org/abs/1709.01507>`_

Wide ResNet: `"Wide Residual Networks" <https://arxiv.org/abs/1605.07146>`_
"""
from functools import partial
from numbers import Real
from typing import Any, Callable, Mapping, Sequence, Union

import megengine.module as M

from basecls.layers import (
    SE,
    activation,
    build_head,
    conv2d,
    init_weights,
    make_divisible,
    norm2d,
    pool2d,
)
from basecls.utils import recursive_update, registers

__all__ = [
    "ResBasicBlock",
    "ResBottleneckBlock",
    "ResDeepStem",
    "ResStem",
    "SimpleStem",
    "AnyStage",
    "ResNet",
]


class ResBasicBlock(M.Module):
    """Residual basic block: x + f(x), f = [3x3 conv, BN, Act] x2."""

    def __init__(
        self,
        w_in: int,
        w_out: int,
        stride: int,
        bot_mul: float,
        se_r: float,
        avg_down: bool,
        norm_name: str,
        act_name: str,
        **kwargs,
    ):
        super().__init__()
        if w_in != w_out or stride > 1:
            if avg_down and stride > 1:
                self.pool = M.AvgPool2d(2, stride)
                self.proj = conv2d(w_in, w_out, 1)
            else:
                self.proj = conv2d(w_in, w_out, 1, stride=stride)
            self.bn = norm2d(norm_name, w_out)

        w_b = round(w_out * bot_mul)
        w_se = make_divisible(w_out * se_r) if se_r > 0.0 else 0
        self.a = conv2d(w_in, w_b, 3, stride=stride)
        self.a_bn = norm2d(norm_name, w_b)
        self.a_act = activation(act_name)
        self.b = conv2d(w_b, w_out, 3)
        self.b_bn = norm2d(norm_name, w_out)
        self.b_bn.final_bn = True
        if w_se > 0:
            self.se = SE(w_out, w_se, act_name)
        self.act = activation(act_name)

    def forward(self, x):
        x_p = x
        if getattr(self, "pool", None) is not None:
            x_p = self.pool(x_p)
        if getattr(self, "proj", None) is not None:
            x_p = self.proj(x_p)
            x_p = self.bn(x_p)

        x = self.a(x)
        x = self.a_bn(x)
        x = self.a_act(x)
        x = self.b(x)
        x = self.b_bn(x)
        if getattr(self, "se", None) is not None:
            x = self.se(x)
        x += x_p
        x = self.act(x)
        return x


class ResBottleneckBlock(M.Module):
    """Residual bottleneck block: x + f(x), f = 1x1, 3x3, 1x1 [+SE]."""

    def __init__(
        self,
        w_in: int,
        w_out: int,
        stride: int,
        bot_mul: float,
        group_w: int,
        se_r: float,
        avg_down: bool,
        norm_name: str,
        act_name: str,
        **kwargs,
    ):
        super().__init__()
        if w_in != w_out or stride > 1:
            if avg_down and stride > 1:
                self.pool = M.AvgPool2d(2, stride)
                self.proj = conv2d(w_in, w_out, 1)
            else:
                self.proj = conv2d(w_in, w_out, 1, stride=stride)
            self.bn = norm2d(norm_name, w_out)

        w_b = round(w_out * bot_mul)
        w_se = make_divisible(w_out * se_r) if se_r > 0.0 else 0
        groups = w_b // group_w
        self.a = conv2d(w_in, w_b, 1)
        self.a_bn = norm2d(norm_name, w_b)
        self.a_act = activation(act_name)
        self.b = conv2d(w_b, w_b, 3, stride=stride, groups=groups)
        self.b_bn = norm2d(norm_name, w_b)
        self.b_act = activation(act_name)
        self.c = conv2d(w_b, w_out, 1)
        self.c_bn = norm2d(norm_name, w_out)
        self.c_bn.final_bn = True
        if w_se > 0:
            self.se = SE(w_out, w_se, act_name)
        self.act = activation(act_name)

    def forward(self, x):
        x_p = x
        if getattr(self, "pool", None) is not None:
            x_p = self.pool(x_p)
        if getattr(self, "proj", None) is not None:
            x_p = self.proj(x_p)
            x_p = self.bn(x_p)

        x = self.a(x)
        x = self.a_bn(x)
        x = self.a_act(x)
        x = self.b(x)
        x = self.b_bn(x)
        x = self.b_act(x)
        x = self.c(x)
        x = self.c_bn(x)
        if getattr(self, "se", None) is not None:
            x = self.se(x)
        x += x_p
        x = self.act(x)
        return x


class ResDeepStem(M.Module):
    """ResNet-D stem: [3x3, BN, Act] x3, MaxPool."""

    def __init__(self, w_in: int, w_out: int, norm_name: str, act_name: str, **kwargs):
        super().__init__()
        w_b = w_out // 2
        self.a = conv2d(w_in, w_b, 3, stride=2)
        self.a_bn = norm2d(norm_name, w_b)
        self.a_act = activation(act_name)
        self.b = conv2d(w_b, w_b, 3, stride=1)
        self.b_bn = norm2d(norm_name, w_b)
        self.b_act = activation(act_name)
        self.c = conv2d(w_b, w_out, 3, stride=1)
        self.c_bn = norm2d(norm_name, w_out)
        self.c_act = activation(act_name)
        self.pool = pool2d(3, stride=2)

    def forward(self, x):
        x = self.a(x)
        x = self.a_bn(x)
        x = self.a_act(x)
        x = self.b(x)
        x = self.b_bn(x)
        x = self.b_act(x)
        x = self.c(x)
        x = self.c_bn(x)
        x = self.c_act(x)
        x = self.pool(x)
        return x


class ResStem(M.Module):
    """ResNet stem: 7x7, BN, Act, MaxPool."""

    def __init__(self, w_in: int, w_out: int, norm_name: str, act_name: str, **kwargs):
        super().__init__()
        self.conv = conv2d(w_in, w_out, 7, stride=2)
        self.bn = norm2d(norm_name, w_out)
        self.act = activation(act_name)
        self.pool = pool2d(3, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        return x


class SimpleStem(M.Module):
    """Simple stem: 3x3, BN, Act."""

    def __init__(self, w_in: int, w_out: int, norm_name: str, act_name: str, **kwargs):
        super().__init__()
        self.conv = conv2d(w_in, w_out, 3, stride=2)
        self.bn = norm2d(norm_name, w_out)
        self.act = activation(act_name)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class AnyStage(M.Module):
    """AnyNet stage (sequence of blocks w/ the same output shape)."""

    def __init__(
        self, w_in: int, w_out: int, stride: int, depth: int, block_func: Callable, **kwargs
    ):
        super().__init__()
        self.depth = depth
        for i in range(depth):
            block = block_func(w_in, w_out, stride, **kwargs)
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
class ResNet(M.Module):
    """ResNet model.

    Args:
        stem_name: stem name.
        stem_w: stem width.
        block_name: block name.
        depths: depth for each stage (number of blocks in the stage).
        widths: width for each stage (width of each block in the stage).
        strides: strides for each stage (applies to the first block of each stage).
        bot_muls: bottleneck multipliers for each stage (applies to bottleneck block).
            Default: ``1.0``
        group_ws: group widths for each stage (applies to bottleneck block). Default: ``None``
        se_r: Squeeze-and-Excitation (SE) ratio. Default: ``0.0``
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
        depths: Sequence[int],
        widths: Sequence[int],
        strides: Sequence[int],
        bot_muls: Union[float, Sequence[float]] = 1.0,
        group_ws: Sequence[int] = None,
        se_r: float = 0.0,
        avg_down: bool = False,
        zero_init_final_gamma: bool = False,
        norm_name: str = "BN",
        act_name: str = "relu",
        head: Mapping[str, Any] = None,
    ):
        super().__init__()
        self.depths = depths

        stem_func = self.get_stem_func(stem_name)
        self.stem = stem_func(3, stem_w, norm_name, act_name)

        block_func = self.get_block_func(block_name)
        if isinstance(bot_muls, Real):
            bot_muls = [bot_muls] * len(depths)
        if group_ws is None:
            group_ws = [None] * len(depths)
        model_args = [depths, widths, strides, bot_muls, group_ws]
        prev_w = stem_w
        for i, (d, w, s, b, g) in enumerate(zip(*model_args)):
            stage = AnyStage(
                prev_w,
                w,
                s,
                d,
                block_func,
                bot_mul=b,
                group_w=g,
                se_r=se_r,
                avg_down=avg_down,
                norm_name=norm_name,
                act_name=act_name,
            )
            setattr(self, f"s{i + 1}", stage)
            prev_w = w

        self.head = build_head(prev_w, head, norm_name, act_name)

        self.apply(partial(init_weights, zero_init_final_gamma=zero_init_final_gamma))

    def forward(self, x):
        x = self.stem(x)
        for i in range(len(self.depths)):
            stage = getattr(self, f"s{i + 1}")
            x = stage(x)
        if getattr(self, "head", None) is not None:
            x = self.head(x)
        return x

    @staticmethod
    def get_stem_func(name: Union[str, Callable]):
        """Retrieves the stem function by name."""
        if callable(name):
            return name
        if isinstance(name, str):
            stem_funcs = {
                "ResDeepStem": ResDeepStem,
                "ResStem": ResStem,
                "SimpleStem": SimpleStem,
            }
            if name in stem_funcs.keys():
                return stem_funcs[name]
        raise ValueError(f"Stem '{name}' not supported")

    @staticmethod
    def get_block_func(name: Union[str, Callable]):
        """Retrieves the block function by name."""
        if callable(name):
            return name
        if isinstance(name, str):
            block_funcs = {
                "ResBasicBlock": ResBasicBlock,
                "ResBottleneckBlock": ResBottleneckBlock,
            }
            if name in block_funcs.keys():
                return block_funcs[name]
        raise ValueError(f"Block '{name}' not supported")


def _build_resnet(**kwargs):
    model_args = dict(stem_name=ResStem, stem_w=64, head=dict(name="ClsHead"))
    recursive_update(model_args, kwargs)
    return ResNet(**model_args)


@registers.models.register()
def resnet18(**kwargs):
    model_args = dict(
        block_name=ResBasicBlock,
        depths=[2, 2, 2, 2],
        widths=[64, 128, 256, 512],
        strides=[1, 2, 2, 2],
    )
    recursive_update(model_args, kwargs)
    return _build_resnet(**model_args)


@registers.models.register()
def resnet34(**kwargs):
    model_args = dict(
        block_name=ResBasicBlock,
        depths=[3, 4, 6, 3],
        widths=[64, 128, 256, 512],
        strides=[1, 2, 2, 2],
    )
    recursive_update(model_args, kwargs)
    return _build_resnet(**model_args)


@registers.models.register()
def resnet50(**kwargs):
    model_args = dict(
        block_name=ResBottleneckBlock,
        depths=[3, 4, 6, 3],
        widths=[256, 512, 1024, 2048],
        strides=[1, 2, 2, 2],
        bot_muls=[0.25, 0.25, 0.25, 0.25],
        group_ws=[64, 128, 256, 512],
    )
    recursive_update(model_args, kwargs)
    return _build_resnet(**model_args)


@registers.models.register()
def resnet101(**kwargs):
    model_args = dict(
        block_name=ResBottleneckBlock,
        depths=[3, 4, 23, 3],
        widths=[256, 512, 1024, 2048],
        strides=[1, 2, 2, 2],
        bot_muls=[0.25, 0.25, 0.25, 0.25],
        group_ws=[64, 128, 256, 512],
    )
    recursive_update(model_args, kwargs)
    return _build_resnet(**model_args)


@registers.models.register()
def resnet152(**kwargs):
    model_args = dict(
        block_name=ResBottleneckBlock,
        depths=[3, 8, 36, 3],
        widths=[256, 512, 1024, 2048],
        strides=[1, 2, 2, 2],
        bot_muls=[0.25, 0.25, 0.25, 0.25],
        group_ws=[64, 128, 256, 512],
    )
    recursive_update(model_args, kwargs)
    return _build_resnet(**model_args)


@registers.models.register()
def resnet18d(**kwargs):
    model_args = dict(stem_name=ResDeepStem, avg_down=True)
    recursive_update(model_args, kwargs)
    return resnet18(**model_args)


@registers.models.register()
def resnet34d(**kwargs):
    model_args = dict(stem_name=ResDeepStem, avg_down=True)
    recursive_update(model_args, kwargs)
    return resnet34(**model_args)


@registers.models.register()
def resnet50d(**kwargs):
    model_args = dict(stem_name=ResDeepStem, avg_down=True)
    recursive_update(model_args, kwargs)
    return resnet50(**model_args)


@registers.models.register()
def resnet101d(**kwargs):
    model_args = dict(stem_name=ResDeepStem, avg_down=True)
    recursive_update(model_args, kwargs)
    return resnet101(**model_args)


@registers.models.register()
def resnet152d(**kwargs):
    model_args = dict(stem_name=ResDeepStem, avg_down=True)
    recursive_update(model_args, kwargs)
    return resnet152(**model_args)


@registers.models.register()
def resnext50_32x4d(**kwargs):
    model_args = dict(bot_muls=[0.5, 0.5, 0.5, 0.5], group_ws=[4, 8, 16, 32])
    recursive_update(model_args, kwargs)
    return resnet50(**model_args)


@registers.models.register()
def resnext101_32x4d(**kwargs):
    model_args = dict(bot_muls=[0.5, 0.5, 0.5, 0.5], group_ws=[4, 8, 16, 32])
    recursive_update(model_args, kwargs)
    return resnet101(**model_args)


@registers.models.register()
def resnext101_32x8d(**kwargs):
    model_args = dict(bot_muls=[1.0, 1.0, 1.0, 1.0], group_ws=[8, 16, 32, 64])
    recursive_update(model_args, kwargs)
    return resnet101(**model_args)


@registers.models.register()
def resnext101_64x4d(**kwargs):
    model_args = dict(bot_muls=[1.0, 1.0, 1.0, 1.0], group_ws=[4, 8, 16, 32])
    recursive_update(model_args, kwargs)
    return resnet101(**model_args)


@registers.models.register()
def resnext152_32x4d(**kwargs):
    model_args = dict(bot_muls=[0.5, 0.5, 0.5, 0.5], group_ws=[4, 8, 16, 32])
    recursive_update(model_args, kwargs)
    return resnet152(**model_args)


@registers.models.register()
def resnext152_32x8d(**kwargs):
    model_args = dict(bot_muls=[1.0, 1.0, 1.0, 1.0], group_ws=[8, 16, 32, 64])
    recursive_update(model_args, kwargs)
    return resnet152(**model_args)


@registers.models.register()
def resnext152_64x4d(**kwargs):
    model_args = dict(bot_muls=[1.0, 1.0, 1.0, 1.0], group_ws=[4, 8, 16, 32])
    recursive_update(model_args, kwargs)
    return resnet152(**model_args)


@registers.models.register()
def se_resnet18(**kwargs):
    model_args = dict(se_r=0.0625)
    recursive_update(model_args, kwargs)
    return resnet18(**model_args)


@registers.models.register()
def se_resnet34(**kwargs):
    model_args = dict(se_r=0.0625)
    recursive_update(model_args, kwargs)
    return resnet34(**model_args)


@registers.models.register()
def se_resnet50(**kwargs):
    model_args = dict(se_r=0.0625)
    recursive_update(model_args, kwargs)
    return resnet50(**model_args)


@registers.models.register()
def se_resnet101(**kwargs):
    model_args = dict(se_r=0.0625)
    recursive_update(model_args, kwargs)
    return resnet101(**model_args)


@registers.models.register()
def se_resnet152(**kwargs):
    model_args = dict(se_r=0.0625)
    recursive_update(model_args, kwargs)
    return resnet152(**model_args)


@registers.models.register()
def se_resnext50_32x4d(**kwargs):
    model_args = dict(se_r=0.0625)
    recursive_update(model_args, kwargs)
    return resnext50_32x4d(**model_args)


@registers.models.register()
def se_resnext101_32x4d(**kwargs):
    model_args = dict(se_r=0.0625)
    recursive_update(model_args, kwargs)
    return resnext101_32x4d(**model_args)


@registers.models.register()
def se_resnext101_32x8d(**kwargs):
    model_args = dict(se_r=0.0625)
    recursive_update(model_args, kwargs)
    return resnext101_32x8d(**model_args)


@registers.models.register()
def se_resnext101_64x4d(**kwargs):
    model_args = dict(se_r=0.0625)
    recursive_update(model_args, kwargs)
    return resnext101_64x4d(**model_args)


@registers.models.register()
def se_resnext152_32x4d(**kwargs):
    model_args = dict(se_r=0.0625)
    recursive_update(model_args, kwargs)
    return resnext152_32x4d(**model_args)


@registers.models.register()
def se_resnext152_32x8d(**kwargs):
    model_args = dict(se_r=0.0625)
    recursive_update(model_args, kwargs)
    return resnext152_32x8d(**model_args)


@registers.models.register()
def se_resnext152_64x4d(**kwargs):
    model_args = dict(se_r=0.0625)
    recursive_update(model_args, kwargs)
    return resnext152_64x4d(**model_args)


@registers.models.register()
def wide_resnet50_2(**kwargs):
    model_args = dict(bot_muls=[0.5, 0.5, 0.5, 0.5])
    recursive_update(model_args, kwargs)
    return resnet50(**model_args)


@registers.models.register()
def wide_resnet101_2(**kwargs):
    model_args = dict(bot_muls=[0.5, 0.5, 0.5, 0.5])
    recursive_update(model_args, kwargs)
    return resnet101(**model_args)
