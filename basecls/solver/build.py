#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
from collections import namedtuple
from typing import Iterable, Union

import megengine.distributed as dist
import megengine.module as M
import megengine.optimizer as optim
from basecore.config import ConfigDict
from megengine import Parameter
from megengine.amp import GradScaler
from megengine.autodiff import GradManager

from basecls.utils import registers

from .optimizer import LAMB, LARS, SGD
from .weight_decay import get_param_groups

__all__ = ["Solver", "BaseSolver", "DefaultSolver"]

Solver = namedtuple("Solver", ["optimizer", "grad_manager", "grad_scaler"])


class BaseSolver:
    """Base class for solver factory.

    A solver factory should return a :py:class:`~Solver` object, which combines
    an :py:class:`~megengine.optimizer.Optimizer` and
    a :py:class:`~megengine.autodiff.GradManager`.
    """

    @classmethod
    def build(cls, cfg: ConfigDict, model: M.Module) -> Solver:
        """Abstract build function

        Args:
            cfg: config for training.
            model: model for training.

        Returns:
            A solver.

        """
        raise NotImplementedError


@registers.solvers.register()
class DefaultSolver(BaseSolver):
    """The default solver factory.

    According to ``cfg.reduce_mode``, learning rate and weight decay will be scaled automatically
    following the linear scaling rule, see
    `"Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour"
    <https://arxiv.org/abs/1706.02677>`_ and `科学地调整lr和wd
    <https://wiki.megvii-inc.com/pages/viewpage.action?pageId=237798086>`_ for more details.

    It supports ``"sgd"``, ``"adam"`` and ``"adamw"``.

    Note:
        This linear scaling rule can only work well with SGD. We are still looking for
        the applicable scaling rule for Adam and AdamW. Thus we recommend keeping default
        training settings (like learning rate and world size) when using Adam and AdamW.
    """

    @classmethod
    def build(cls, cfg: ConfigDict, model: M.Module) -> Solver:
        """Build function with the linear scaling strategy.

        Args:
            cfg: config for training.
            model: model for training.

        Returns:
            A solver.
        """
        amp_cfg = cfg.amp
        cfg = cfg.solver

        world_size = dist.get_world_size()

        # build optimizer
        lr = cfg.basic_lr * world_size  # linear scaling rule

        optim_params = get_param_groups(model, cfg.weight_decay)

        optimizer = cls.build_optimizer(cfg, optim_params, lr, 0)

        # build grad_manager
        gm = GradManager()
        callbacks = [dist.make_allreduce_cb("mean", dist.WORLD)] if world_size > 1 else None
        gm.attach(model.parameters(), callbacks=callbacks)

        # build grad_scaler
        scaler = (
            GradScaler(init_scale=65536.0, growth_interval=2000)
            if amp_cfg.dynamic_scale
            else GradScaler(init_scale=128.0, growth_interval=0)
        )

        return Solver(optimizer, gm, scaler)

    @classmethod
    def build_optimizer(
        cls, cfg: ConfigDict, params: Union[Iterable[Parameter], dict], lr: float, wd: float
    ) -> optim.Optimizer:
        """Build optimizer according to training config.

        Args:
            cfg: config for training.
            params: iterable of parameters to optimize or dicts defining parameter groups.
            lr: learning rate.
            weight_decay: weight decay (L2, penalty).

        Returns:
            An optimizer.
        """
        if cfg.optimizer == "adam":
            return optim.Adam(params, lr=lr, weight_decay=wd, betas=cfg.betas)
        elif cfg.optimizer == "adamw":
            return optim.AdamW(params, lr=lr, weight_decay=wd, betas=cfg.betas)
        elif cfg.optimizer == "lamb":
            return LAMB(
                params, lr=lr, weight_decay=wd, betas=cfg.betas, always_adapt=cfg.always_adapt
            )
        elif cfg.optimizer == "lars":
            return LARS(
                params,
                lr=lr,
                weight_decay=wd,
                momentum=cfg.momentum,
                nesterov=cfg.nesterov,
                always_adapt=cfg.always_adapt,
            )
        elif cfg.optimizer == "sgd":
            return SGD(params, lr=lr, weight_decay=wd, momentum=cfg.momentum, nesterov=cfg.nesterov)
        else:
            raise NotImplementedError(f"Optimizer '{cfg.optimizer}' not supported")
