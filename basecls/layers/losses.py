#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import megengine as mge
import megengine.functional as F
import megengine.module as M
from basecore.config import ConfigDict

__all__ = ["build_loss", "BinaryCrossEntropy", "CrossEntropy"]


def build_loss(cfg: ConfigDict) -> M.Module:
    """The factory function to build loss.

    Args:
        cfg: config for building loss function.

    Returns:
        A loss function.
    """
    loss_args = cfg.loss.to_dict()
    loss_name = loss_args.pop("name", None)
    if loss_name is None:
        raise ValueError("Loss name is missing")
    if callable(loss_name):
        return loss_name(**loss_args)
    if isinstance(loss_name, str):
        loss_funcs = {
            "BinaryCrossEntropy": BinaryCrossEntropy,
            "CrossEntropy": CrossEntropy,
        }
        if loss_name in loss_funcs:
            return loss_funcs[loss_name](**loss_args)
    raise ValueError(f"Loss '{loss_name}' not supported")


class BinaryCrossEntropy(M.Module):
    """The module for binary cross entropy.

    See :py:func:`~megengine.functional.loss.binary_cross_entropy` for more details.
    """

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x: mge.Tensor, y: mge.Tensor) -> mge.Tensor:
        return F.loss.binary_cross_entropy(x, y)


class CrossEntropy(M.Module):
    """The module for cross entropy.

    It supports both categorical labels and one-hot labels.
    See :py:func:`~megengine.functional.loss.cross_entropy` for more details.

    Args:
        axis: reduced axis. Default: ``1``
        label_smooth: label smooth factor. Default: ``0.0``
    """

    def __init__(self, axis: int = 1, label_smooth: float = 0.0):
        super().__init__()
        self.axis = axis
        self.label_smooth = label_smooth

    def forward(self, x: mge.Tensor, y: mge.Tensor) -> mge.Tensor:
        if x.ndim == y.ndim + 1:
            return F.loss.cross_entropy(x, y, axis=self.axis, label_smooth=self.label_smooth)
        else:
            assert x.ndim == y.ndim
            if self.label_smooth != 0:
                y = y * (1 - self.label_smooth) + self.label_smooth / y.shape[self.axis]
            return (-y * F.logsoftmax(x, axis=self.axis)).sum(self.axis).mean()
