#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import os
from typing import Iterable, Tuple, Union

import megengine.functional as F
from megengine import Parameter, tensor
from megengine.functional.inplace import _inplace_add_
from megengine.optimizer import Optimizer


class LAMB(Optimizer):
    r"""Implements LAMB algorithm.

    LAMB is proposed in `"Large Batch Optimization for Deep Learning: Training BERT in 76 minutes"
    <https://arxiv.org/abs/1904.00962>`_.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups.
        lr: learning rate.
        betas: coefficients used for computing running averages of gradient and its square.
            Default: ``(0.9, 0.999)``
        eps: term added to the denominator to improve numerical stability. Default: ``1e-8``
        bias_correction: enables bias correction by ``1 - beta ** step``. Default: ``True``
        weight_decay: weight decay (L2 penalty). Default: ``0.0``
        always_adapt: apply adaptive lr to ``0.0`` weight decay parameter. Default: ``False``
    """

    def __init__(
        self,
        params: Union[Iterable[Parameter], dict],
        lr: float,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        bias_correction: bool = True,
        weight_decay: float = 0.0,
        always_adapt: bool = False,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        defaults = dict(lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
        super().__init__(params, defaults)
        self.bias_correction = bias_correction
        self.always_adapt = always_adapt
        self._disable_type_convert = True

    def _create_state(self, param_group):
        for param in param_group["params"]:
            self._add_state(param, "exp_avg")
            self._add_state(param, "exp_avg_sq")
            self._add_state(param, "step", initializer=0.0)

    def _updates(self, param_group):
        lr = param_group["lr"]
        weight_decay = param_group["weight_decay"]
        eps = param_group["eps"]
        beta0, beta1 = param_group["betas"]

        # since `conver_inputs` is disabled for param updates,
        # scalar should be explicitly tansforred to tensor

        _lr, _neg_lr = map(tensor, (lr, -lr))
        _weight_decay = tensor(weight_decay)
        _eps = tensor(eps)
        _beta0, _beta1 = map(tensor, (beta0, beta1))

        c1, c05, c0 = map(tensor, (1.0, 0.5, 0.0))

        def norm(vec):
            return F.sum(vec * vec) ** c05

        inplace_mode = int(os.getenv("MEGENGINE_INPLACE_UPDATE", "0"))
        if inplace_mode:
            # reduce device sync
            c1_sub_beta0, c1_sub_beta1 = map(tensor, (1 - beta0, 1 - beta1))

        for param in param_group["params"]:

            if param.grad is None:
                continue

            grad = param.grad

            states = self._state[param]

            step, exp_avg, exp_avg_sq = (
                states["step"],
                states["exp_avg"],
                states["exp_avg_sq"],
            )

            p_norm = norm(param.flatten())

            if inplace_mode:
                _inplace_add_(step, c1, alpha=c1, beta=c1)
                _inplace_add_(exp_avg, grad, alpha=_beta0, beta=c1_sub_beta0)
                _inplace_add_(exp_avg_sq, grad * grad, alpha=_beta1, beta=c1_sub_beta1)

                bias_correction1 = c1 - _beta0 ** step if self.bias_correction else c1
                bias_correction2 = c1 - _beta1 ** step if self.bias_correction else c1
                delta = (exp_avg / bias_correction1) / (
                    (exp_avg_sq / bias_correction2) ** c05 + _eps
                )
                if weight_decay != 0.0:
                    _inplace_add_(delta, param, alpha=c1, beta=_weight_decay)

                d_norm = norm(delta.flatten())
                trust_ratio = (
                    p_norm / d_norm
                    if (self.always_adapt or weight_decay > 0) and p_norm > c0 and d_norm > c0
                    else c1
                )
                _inplace_add_(param, delta, alpha=c1, beta=_neg_lr * trust_ratio)
                continue

            # step = step + c1
            step += c1

            # exp_avg = _beta0 * exp_avg + grad * (c1 - _beta0)
            exp_avg *= _beta0
            exp_avg += grad * (c1 - _beta0)

            # exp_avg_sq = _beta1 * exp_avg_sq + (c1 - _beta1) * (grad * grad)
            exp_avg_sq *= _beta1
            exp_avg_sq += (c1 - _beta1) * (grad * grad)

            bias_correction1 = c1 - _beta0 ** step if self.bias_correction else c1
            bias_correction2 = c1 - _beta1 ** step if self.bias_correction else c1
            delta = (exp_avg / bias_correction1) / ((exp_avg_sq / bias_correction2) ** c05 + _eps)
            if weight_decay != 0.0:
                delta += param * _weight_decay

            d_norm = norm(delta.flatten())
            trust_ratio = (
                p_norm / d_norm
                if (self.always_adapt or weight_decay > 0) and p_norm > c0 and d_norm > c0
                else c1
            )
            param -= _lr * trust_ratio * delta
