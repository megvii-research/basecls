#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import os
from typing import Iterable, Union

from megengine import Parameter, tensor
from megengine.functional.inplace import _inplace_add_
from megengine.optimizer import Optimizer


class SGD(Optimizer):
    r"""Implements stochastic gradient descent.

    Nesterov momentum is based on the formula from
    `"On the importance of initialization and momentum in deep learning"
    <http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf>`_.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups.
        lr: learning rate.
        momentum: momentum factor. Default: ``0.0``
        nesterov: enables Nesterov momentum. Default: ``False``
        weight_decay: weight decay (L2 penalty). Default: ``0.0``
    """

    def __init__(
        self,
        params: Union[Iterable[Parameter], dict],
        lr: float,
        momentum: float = 0.0,
        nesterov: bool = False,
        weight_decay: float = 0.0,
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

        inplace_mode = int(os.getenv("MEGENGINE_INPLACE_UPDATE", "0"))
        if inplace_mode:
            _neg_lr = tensor(-lr)
            c1 = tensor([1.0])

        for param in param_group["params"]:
            if param.grad is None:
                continue

            grad = param.grad
            if weight_decay != 0.0:
                grad = grad + param * _weight_decay

            if inplace_mode:
                if momentum != 0.0:
                    v = self._state[param]["momentum_buffer"]
                    _inplace_add_(v, grad, alpha=_momentum, beta=c1)
                    if self.nesterov:
                        grad = grad + v * _momentum
                    else:
                        grad = v
                _inplace_add_(param, grad, alpha=c1, beta=_neg_lr)
                continue

            if momentum != 0.0:
                v = self._state[param]["momentum_buffer"]
                v *= _momentum
                v += grad
                if self.nesterov:
                    grad = grad + v * _momentum
                else:
                    grad = v
            param -= _lr * grad
