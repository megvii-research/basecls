#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
"""VGG Series

VGG: `"Very Deep Convolutional Networks for Large-Scale Image Recognition"
<https://arxiv.org/abs/1409.1556>`_
"""
from typing import Any, Mapping, Sequence

import megengine as mge
import megengine.module as M

from basecls.layers import activation, build_head, conv2d, init_weights, norm2d
from basecls.utils import recursive_update, registers

__all__ = ["VGGStage", "VGG"]


class VGGStage(M.Module):
    """VGG stage (sequence of blocks w/ the same output shape)."""

    def __init__(self, w_in: int, w_out: int, depth: int, norm_name: str, act_name: str):
        super().__init__()
        self.depth = depth
        for i in range(depth):
            block = M.Sequential(
                conv2d(w_in, w_out, 3), norm2d(norm_name, w_out), activation(act_name)
            )
            setattr(self, f"b{i + 1}", block)
            w_in = w_out
        self.max_pool = M.MaxPool2d(kernel_size=2, stride=2)

    def __len__(self):
        return self.depth

    def forward(self, x: mge.Tensor) -> mge.Tensor:
        for i in range(self.depth):
            block = getattr(self, f"b{i + 1}")
            x = block(x)
        x = self.max_pool(x)
        return x


@registers.models.register()
class VGG(M.Module):
    """VGG model.

    Args:
        depths: depth for each stage (number of blocks in the stage).
        widths: width for each stage (width of each block in the stage).
        norm_name: normalization function. Default: ``None``
        act_name: activation function. Default: ``"relu"``
        head: head args. Default: ``None``
    """

    def __init__(
        self,
        depths: Sequence[int],
        widths: Sequence[int],
        norm_name: str = None,
        act_name: str = "relu",
        head: Mapping[str, Any] = None,
    ):
        super().__init__()
        self.depths = depths

        model_args = [depths, widths]
        prev_w = 3
        for i, (d, w) in enumerate(zip(*model_args)):
            stage = VGGStage(prev_w, w, d, norm_name, act_name)
            setattr(self, f"s{i + 1}", stage)
            prev_w = w

        self.head = build_head(prev_w, head, None, act_name)

        self.apply(init_weights)

    def forward(self, x: mge.Tensor) -> mge.Tensor:
        for i in range(len(self.depths)):
            stage = getattr(self, f"s{i + 1}")
            x = stage(x)
        if getattr(self, "head", None) is not None:
            x = self.head(x)
        return x


def _build_vgg(**kwargs):
    model_args = dict(head=dict(name="VGGHead", dropout_prob=0.5))
    recursive_update(model_args, kwargs)
    return VGG(**model_args)


@registers.models.register()
def vgg11(**kwargs):
    model_args = dict(depths=[1, 1, 2, 2, 2], widths=[64, 128, 256, 512, 512])
    recursive_update(model_args, kwargs)
    return _build_vgg(**model_args)


@registers.models.register()
def vgg11_bn(**kwargs):
    model_args = dict(norm_name="BN")
    recursive_update(model_args, kwargs)
    return vgg11(**model_args)


@registers.models.register()
def vgg13(**kwargs):
    model_args = dict(depths=[2, 2, 2, 2, 2], widths=[64, 128, 256, 512, 512])
    recursive_update(model_args, kwargs)
    return _build_vgg(**model_args)


@registers.models.register()
def vgg13_bn(**kwargs):
    model_args = dict(norm_name="BN")
    recursive_update(model_args, kwargs)
    return vgg13(**model_args)


@registers.models.register()
def vgg16(**kwargs):
    model_args = dict(depths=[2, 2, 3, 3, 3], widths=[64, 128, 256, 512, 512])
    recursive_update(model_args, kwargs)
    return _build_vgg(**model_args)


@registers.models.register()
def vgg16_bn(**kwargs):
    model_args = dict(norm_name="BN")
    recursive_update(model_args, kwargs)
    return vgg16(**model_args)


@registers.models.register()
def vgg19(**kwargs):
    model_args = dict(depths=[2, 2, 4, 4, 4], widths=[64, 128, 256, 512, 512])
    recursive_update(model_args, kwargs)
    return _build_vgg(**model_args)


@registers.models.register()
def vgg19_bn(**kwargs):
    model_args = dict(norm_name="BN")
    recursive_update(model_args, kwargs)
    return vgg19(**model_args)
