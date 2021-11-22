#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
from typing import Callable, Union

import megengine as mge
import megengine.functional as F
import megengine.module as M

from .activations import activation

__all__ = ["conv2d", "norm2d", "pool2d", "gap2d", "linear", "SE", "DropPath"]


def conv2d(
    w_in: int,
    w_out: int,
    k: int,
    *,
    stride: int = 1,
    dilation: int = 1,
    groups: int = 1,
    bias: bool = False,
) -> M.Conv2d:
    """Helper for building a conv2d layer.

    It will calculate padding automatically.

    Args:
        w_in: input width.
        w_out: output width.
        k: kernel size.
        stride: stride. Default: ``1``
        dilation: dilation. Default: ``1``
        groups: groups. Default: ``1``
        bias: enable bias or not. Default: ``False``

    Returns:
        A conv2d module.
    """
    assert k % 2 == 1, "Only odd size kernels supported to avoid padding issues."
    s, p, d, g, b = stride, (k - 1) * dilation // 2, dilation, groups, bias
    return M.Conv2d(w_in, w_out, k, stride=s, padding=p, dilation=d, groups=g, bias=b)


def norm2d(name: Union[str, Callable], w_in: int, **kwargs) -> M.Module:
    """Helper for building a norm2d layer.

    Args:
        norm_name: normalization name, supports ``None``, ``"BN"``, ``"GN"``, ``"IN"``, ``"LN"``
            and ``"SyncBN"``.
        w_in: input width.

    Returns:
        A norm2d module.
    """
    if name is None:
        return M.Identity()
    if callable(name):
        return name(w_in, **kwargs)
    if isinstance(name, str):
        norm_funcs = {
            "BN": M.BatchNorm2d,
            "GN": M.GroupNorm,
            "IN": M.InstanceNorm,
            "LN": M.LayerNorm,
            "SyncBN": M.SyncBatchNorm,
        }
        if name in norm_funcs.keys():
            return norm_funcs[name](w_in, **kwargs)
    raise ValueError(f"Norm name '{name}' not supported")


def pool2d(k: int, *, stride: int = 1, name: str = "max") -> M.Module:
    """Helper for building a pool2d layer.

    Args:
        k: kernel size.
        stride: stride. Default: ``1``
        name: pooling name, supports ``"avg"`` and ``"max"``.

    Returns:
        A pool2d module.
    """
    assert k % 2 == 1, "Only odd size kernels supported to avoid padding issues."
    pool_funcs = {
        "avg": M.AvgPool2d,
        "max": M.MaxPool2d,
    }
    if name not in pool_funcs.keys():
        raise ValueError(f"Pool name '{name}' not supported")
    return pool_funcs[name](k, stride=stride, padding=(k - 1) // 2)


def gap2d(shape=1) -> M.AdaptiveAvgPool2d:
    """Helper for building a gap2d layer.

    Args:
        shape: output shape. Default: ``1``

    Returns:
        A gap2d module.
    """
    return M.AdaptiveAvgPool2d(shape)


def linear(w_in: int, w_out: int, *, bias: bool = False) -> M.Linear:
    """Helper for building a linear layer.

    Args:
        w_in: input width.
        w_out: output width.
        bias: enable bias or not. Default: ``False``

    Returns:
        A linear module.
    """
    return M.Linear(w_in, w_out, bias=bias)


class SE(M.Module):
    """Squeeze-and-Excitation (SE) block: AvgPool, FC, Act, FC, Sigmoid.

    Args:
        w_in: input width.
        w_se: se width.
        act_name: activation name.
        approx_sigmoid: approximated sigmoid function.

    Attributes:
        avg_pool: gad2d layer.
        f_ex: sequantial which conbines conv2d -> act -> conv2d -> sigmoid.
    """

    def __init__(self, w_in: int, w_se: int, act_name: str, approx_sigmoid: bool = False):
        super().__init__()
        self.avg_pool = gap2d()
        self.f_ex = M.Sequential(
            conv2d(w_in, w_se, 1, bias=True),
            activation(act_name),
            conv2d(w_se, w_in, 1, bias=True),
            activation("hsigmoid") if approx_sigmoid else M.Sigmoid(),
        )

    def forward(self, x: mge.Tensor) -> mge.Tensor:
        return x * self.f_ex(self.avg_pool(x))


class DropPath(M.Dropout):
    """DropPath block.

    Args:
        drop_prob: the probability to drop (set to zero) each path.
    """

    def forward(self, x: mge.Tensor):
        if not self.training or self.drop_prob == 0.0:
            return x
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = F.ones(shape)
        mask = F.dropout(mask, self.drop_prob, training=self.training)
        return x * mask
