#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
from typing import Callable, Union

import megengine as mge
import megengine.functional as F
import megengine.module as M

__all__ = ["activation", "ELU", "HSigmoid", "HSwish", "ReLU6", "Tanh"]


def activation(name: Union[str, Callable], **kwargs) -> M.Module:
    """Helper for building an activation layer.

    Args:
        name: activation name, supports ``"elu"``, ``"gelu"``, ``"hsigmoid"``, ``"hswish"``,
            ``"leaky_relu"``, ``"relu"``, ``"relu6"``, ``"prelu"``, ``"silu"`` and ``"tanh"``.

    Returns:
        An activation module.
    """
    if name is None:
        return M.Identity()
    if callable(name):
        return name(**kwargs)
    if isinstance(name, str):
        act_funcs = {
            "elu": ELU,
            "gelu": M.GELU,
            "hsigmoid": HSigmoid,
            "hswish": HSwish,
            "leaky_relu": M.LeakyReLU,
            "relu": M.ReLU,
            "relu6": ReLU6,
            "prelu": M.PReLU,
            "silu": M.SiLU,
            "tanh": Tanh,
        }
        if name in act_funcs.keys():
            return act_funcs[name](**kwargs)
    raise ValueError(f"Activation name '{name}' not supported")


class ELU(M.Module):
    r"""ELU activation function.

    .. math::

        \text{ELU}(x) = \begin{cases}
            x, & \text{if } x > 0, \\
            \alpha \left( \exp(x) - 1 \right), & \text{if } x \le 0
        \end{cases}

    Args:
        alpha: the :math:`\alpha` value for the ELU formulation. Default: ``1.0``
    """

    def __init__(self, alpha: float = 1.0, name=None):
        super().__init__(name)
        self.alpha = alpha

    def forward(self, x: mge.Tensor) -> mge.Tensor:
        return F.relu(x) + F.minimum(0, self.alpha * (F.exp(x) - 1))


class HSigmoid(M.Module):
    r"""Hard sigmoid activation function.

    .. math::

        \text{HSigmoid}(x) = \begin{cases}
            0 & \text{if } x \le -3, \\
            1 & \text{if } x \ge 3, \\
            x / 6 + 1 / 2 & \text{otherwise}
        \end{cases}
    """

    def forward(self, x: mge.Tensor) -> mge.Tensor:
        return F.nn.hsigmoid(x)


class HSwish(M.Module):
    r"""Hard swish activation function.

    .. math::

        \text{HSwish}(x) = \begin{cases}
            0 & \text{if } x \le -3, \\
            x & \text{if } x \ge 3, \\
            x (x + 3) / 6 & \text{otherwise}
        \end{cases}
    """

    def forward(self, x: mge.Tensor) -> mge.Tensor:
        return F.nn.hswish(x)


class ReLU6(M.Module):
    r"""ReLU6 activation function.

    .. math::

        \text{ReLU6}(x) = \min \left( \max(0, x), 6 \right)
    """

    def forward(self, x: mge.Tensor) -> mge.Tensor:
        return F.nn.relu6(x)


class Tanh(M.Module):
    r"""Tanh activation function.

    .. math::

        \text{Tanh}(x) = \text{tanh}(x) = \frac{\exp(x) - \exp(-x)}{\exp(x) + \exp(-x)}
    """

    def forward(self, x: mge.Tensor) -> mge.Tensor:
        return F.tanh(x)
