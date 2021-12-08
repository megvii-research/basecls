#!/usr/bin/env python3
# Copyright (c) 2020 Ross Wightman
# This file has been modified by Megvii ("Megvii Modifications").
# All Megvii Modifications are Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
"""LARS optimizer

References: https://github.com/rwightman/pytorch-image-models/blob/master/timm/optim/lars.py
"""
import os
from typing import Iterable, Union

import megengine.functional as F
from megengine import Parameter, tensor
from megengine.functional.inplace import _inplace_add_
from megengine.optimizer import Optimizer


class LARS(Optimizer):
    r"""Implements LARS algorithm.

    LARS is proposed in `"Large Batch Optimization for Deep Learning: Training BERT in 76 minutes"
    <https://arxiv.org/abs/1904.00962>`_.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups.
        lr: learning rate.
        momentum: momentum factor. Default: ``0.0``
        nesterov: enables Nesterov momentum. Default: ``False``
        weight_decay: weight decay (L2 penalty). Default: ``0.0``
        always_adapt: apply adaptive lr to ``0.0`` weight decay parameter. Default: ``False``
    """

    def __init__(
        self,
        params: Union[Iterable[Parameter], dict],
        lr: float,
        momentum: float = 0.0,
        nesterov: bool = False,
        weight_decay: float = 0.0,
        always_adapt: bool = False,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires a momentum")

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.nesterov = nesterov
        self.always_adapt = always_adapt
        self._disable_type_convert = True

    def _create_state(self, param_group):
        if param_group["momentum"] != 0.0:
            for param in param_group["params"]:
                self._add_state(param, "momentum_buffer")

    def _updates(self, param_group):
        lr = param_group["lr"]
        weight_decay = param_group["weight_decay"]
        momentum = param_group["momentum"]

        # since `conver_inputs` is disabled for param updates,
        # scalar should be explicitly tansforred to tensor

        _lr = tensor(lr)
        _weight_decay = tensor(weight_decay)
        _momentum = tensor(momentum)

        c1, c05, c0 = map(tensor, (1.0, 0.5, 0.0))

        def norm(vec):
            return F.sum(vec * vec) ** c05

        inplace_mode = int(os.getenv("MEGENGINE_INPLACE_UPDATE", "0"))
        if inplace_mode:
            _neg_lr = tensor(-lr)

        for param in param_group["params"]:
            if param.grad is None:
                continue

            grad = param.grad
            if weight_decay != 0.0:
                grad = grad + param * _weight_decay

            p_norm = norm(param.flatten())

            if inplace_mode:
                if momentum != 0.0:
                    v = self._state[param]["momentum_buffer"]
                    _inplace_add_(v, grad, alpha=_momentum, beta=c1)
                    if self.nesterov:
                        grad = grad + v * _momentum
                    else:
                        grad = v
                d_norm = norm(grad.flatten())
                trust_ratio = (
                    p_norm / d_norm
                    if (self.always_adapt or weight_decay > 0) and p_norm > c0 and d_norm > c0
                    else c1
                )
                _inplace_add_(param, grad, alpha=c1, beta=_neg_lr * trust_ratio)
                continue

            if momentum != 0.0:
                v = self._state[param]["momentum_buffer"]
                v *= _momentum
                v += grad
                if self.nesterov:
                    grad = grad + v * _momentum
                else:
                    grad = v
            d_norm = norm(grad.flatten())
            trust_ratio = (
                p_norm / d_norm
                if (self.always_adapt or weight_decay > 0) and p_norm > c0 and d_norm > c0
                else c1
            )
            param -= _lr * trust_ratio * grad
