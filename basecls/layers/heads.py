#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import copy
from typing import Any, Mapping

import megengine as mge
import megengine.functional as F
import megengine.module as M

from .modules import SE, activation, conv2d, gap2d, linear, norm2d

__all__ = ["build_head", "ClsHead", "MBV3Head", "VGGHead"]


def build_head(
    w_in: int, head_args: Mapping[str, Any] = None, norm_name: str = "BN", act_name: str = "relu"
) -> M.Module:
    """The factory function to build head.

    Note:
        if ``head_args`` is ``None`` or ``head_args["name"]`` is ``None``, this function will do
        nothing and return ``None``.

    Args:
        w_in: input width.
        head_args: head args. Default: ``None``
        norm_name: default normalization function, will be overridden by the same key in
            ``head_args``. Default: ``"BN"``
        act_name: default activation function, will be overridden by the same key in ``head_args``.
            Default: ``"relu"``

    Returns:
        A head.
    """
    if head_args is None:
        return None
    head_args = copy.deepcopy(head_args)
    head_name = head_args.pop("name", None)
    if head_name is None:
        return None

    head_args["w_in"] = w_in
    head_args.setdefault("norm_name", norm_name)
    head_args.setdefault("act_name", act_name)

    if callable(head_name):
        return head_name(**head_args)
    if isinstance(head_name, str):
        head_funcs = {
            "ClsHead": ClsHead,
            "MBV3Head": MBV3Head,
            "VGGHead": VGGHead,
        }
        if head_name in head_funcs:
            return head_funcs[head_name](**head_args)
    raise ValueError(f"Head '{head_name}' not supported")


class ClsHead(M.Module):
    """Cls head: Conv, BN, Act, AvgPool, FC.

    Args:
        w_in: input width.
        w_out: output width, normally the number of classes. Default: ``1000``
        width: width for first conv in head, conv will be omitted if set to 0. Default: ``0``
        dropout_prob: dropout probability. Default: ``0.0``
        norm_name: normalization function. Default: ``"BN"``
        act_name: activation function. Default: ``"relu"``
        bias: whether fc has bias. Default: ``True``
    """

    def __init__(
        self,
        w_in: int,
        w_out: int = 1000,
        width: int = 0,
        dropout_prob: float = 0.0,
        norm_name: str = "BN",
        act_name: str = "relu",
        bias: bool = True,
    ):
        super().__init__()
        self.width = width
        if self.width > 0:
            self.conv = conv2d(w_in, self.width, 1)
            self.bn = norm2d(norm_name, self.width)
            self.act = activation(act_name)
            w_in = self.width
        self.avg_pool = gap2d()
        if dropout_prob > 0.0:
            self.dropout = M.Dropout(dropout_prob)
        self.fc = linear(w_in, w_out, bias=bias)

    def forward(self, x: mge.Tensor) -> mge.Tensor:
        if self.width > 0:
            x = self.conv(x)
            x = self.bn(x)
            x = self.act(x)
        x = self.avg_pool(x)
        x = F.flatten(x, 1)
        if getattr(self, "dropout", None) is not None:
            x = self.dropout(x)
        x = self.fc(x)
        return x


class MBV3Head(M.Module):
    """MobileNet V3 head: Conv, BN, Act, AvgPool, SE, FC, Act, FC.

    Args:
        w_in: input width.
        w_out: output width, normally the number of classes.
        width: width for first conv in head.
        w_h: width for first linear in head.
        dropout_prob: dropout probability. Default: ``0.0``
        se_r: Squeeze-and-Excitation (SE) ratio. Default: ``0.0``
        norm_name: normalization function. Default: ``"BN"``
        act_name: activation function. Default: ``"hswish"``
        bias: whether fc has bias. Default: ``True``
    """

    def __init__(
        self,
        w_in: int,
        w_out: int = 1000,
        width: int = 960,
        w_h: int = 1280,
        dropout_prob: float = 0.0,
        se_r: float = 0.0,
        norm_name: str = "BN",
        act_name: str = "hswish",
        bias: bool = True,
    ):
        super().__init__()
        self.conv = conv2d(w_in, width, 1)
        self.bn = norm2d(norm_name, width)
        self.act = activation(act_name)
        self.avg_pool = gap2d()
        if se_r > 0.0:
            self.se = SE(width, int(se_r * width), act_name)
        self.h_fc = linear(width, w_h, bias=bias)
        self.h_act = activation(act_name)
        if dropout_prob > 0.0:
            self.dropout = M.Dropout(dropout_prob)
        self.fc = linear(w_h, w_out, bias=bias)

    def forward(self, x: mge.Tensor) -> mge.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.avg_pool(x)
        if getattr(self, "se", None) is not None:
            x = self.se(x)
        x = F.flatten(x, 1)
        x = self.h_fc(x)
        x = self.h_act(x)
        if getattr(self, "dropout", None) is not None:
            x = self.dropout(x)
        x = self.fc(x)
        return x


class VGGHead(M.Module):
    """VGG head: AvgPool, [FC, Act, Dropout] x2, FC.

    Args:
        w_in: input width.
        w_out: output width, normally the number of classes. Default: ``1000``
        width: width for linear in head. Default: ``4096``
        dropout_prob: dropout probability. Default: ``0.5``
        act_name: activation function. Default: ``"relu"``
    """

    def __init__(
        self,
        w_in: int,
        w_out: int = 1000,
        width: int = 4096,
        dropout_prob: float = 0.5,
        act_name: str = "relu",
        **kwargs,
    ):
        super().__init__()
        self.avg_pool = gap2d(7)
        self.classifier = M.Sequential(
            linear(w_in * 7 * 7, width, bias=True),
            activation(act_name),
            M.Dropout(dropout_prob),
            linear(width, width, bias=True),
            activation(act_name),
            M.Dropout(dropout_prob),
            linear(width, w_out, bias=True),
        )

    def forward(self, x: mge.Tensor) -> mge.Tensor:
        x = self.avg_pool(x)
        x = F.flatten(x, 1)
        x = self.classifier(x)
        return x
