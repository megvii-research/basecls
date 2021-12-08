#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import copy
import time
from typing import Iterable

import megengine as mge
import megengine.amp as amp
import megengine.distributed as dist
import megengine.functional as F
import megengine.module as M
import megengine.optimizer as optim
from basecore.config import ConfigDict
from basecore.engine import BaseHook, BaseTrainer
from basecore.utils import MeterBuffer
from megengine import jit

from basecls.data import DataLoaderType
from basecls.layers import Preprocess, build_loss
from basecls.solver import Solver
from basecls.utils import registers

__all__ = ["ClsTrainer"]


@registers.trainers.register()
class ClsTrainer(BaseTrainer):
    """Classification trainer.

    Args:
        cfg: config for training.
        model: model for training.
        dataloader: dataloader for training.
        solver: solver for training.
        hooks: hooks for training.

    Attributes:
        cfg: config for training.
        model: model for training.
        ema: model exponential moving average.
        dataloader: dataloader for training.
        solver: solver for training.
        progress: object for recording training process.
        loss: loss function for training.
        meter : object for recording metrics.
    """

    def __init__(
        self,
        cfg: ConfigDict,
        model: M.Module,
        dataloader: DataLoaderType,
        solver: Solver,
        hooks: Iterable[BaseHook] = None,
    ):
        super().__init__(model, dataloader, solver, hooks)
        self.cfg = cfg
        self.ema = copy.deepcopy(model) if cfg.model_ema.enabled else None
        self.preprocess = Preprocess(cfg.preprocess.img_mean, cfg.preprocess.img_std)
        self.loss = build_loss(cfg)
        self.meter = MeterBuffer(cfg.log_every_n_iter)
        if cfg.trace:
            # FIXME: tracing makes the training slower than before, why?
            self.model_step = jit.trace(self.model_step, symbolic=True)

    def train(self):
        start_training_info = (1, 1)
        max_iter = len(self.dataloader)
        max_training_info = (self.cfg.solver.max_epoch, max_iter)
        super().train(start_training_info, max_training_info)

    def before_train(self):
        super().before_train()

    def before_epoch(self):
        super().before_epoch()
        self.dataloader_iter = iter(self.dataloader)

    def after_epoch(self):
        del self.dataloader_iter
        super().after_epoch()

    def train_one_iter(self):
        """Basic logic of training one iteration."""
        data_tik = time.perf_counter()
        data = next(self.dataloader_iter)
        samples, targets = self.preprocess(data)
        mge._full_sync()  # use full_sync func to sync launch queue for dynamic execution
        data_tok = time.perf_counter()

        train_tik = time.perf_counter()
        losses, accs = self.model_step(samples, targets)
        mge._full_sync()  # use full_sync func to sync launch queue for dynamic execution
        train_tok = time.perf_counter()

        # TODO: stats and accs
        loss_meters = {"loss": losses.item()}
        stat_meters = {"stat_acc@1": accs[0].item() * 100, "stat_acc@5": accs[1].item() * 100}
        time_meters = {"train_time": train_tok - train_tik, "data_time": data_tok - data_tik}
        self.meter.update(**loss_meters, **stat_meters, **time_meters)

    def model_step(self, samples, targets):
        optimizer = self.solver.optimizer
        grad_manager = self.solver.grad_manager
        grad_scaler = self.solver.grad_scaler

        with grad_manager:
            with amp.autocast(enabled=self.cfg.amp.enabled):
                outputs = self.model(samples)
                losses = self.loss(outputs, targets)

            if isinstance(losses, mge.Tensor):
                total_loss = losses
            elif isinstance(losses, dict):
                if "total_loss" in losses:
                    total_loss = losses["total_loss"]
                else:
                    # only key contains "loss" will be calculated.
                    total_loss = sum([v for k, v in losses.items() if "loss" in k])
                    losses["total_loss"] = total_loss
            else:
                # list or tuple
                total_loss = sum(losses)

            total_loss = total_loss / self.cfg.solver.accumulation_steps

            # this is made compatible with one hot labels
            if targets.ndim == 2:
                targets = F.argmax(targets, axis=1)
            accs = F.metric.topk_accuracy(outputs, targets, (1, 5))

            if self.cfg.amp.enabled:
                grad_scaler.backward(grad_manager, total_loss)
            else:
                grad_manager.backward(total_loss)

        if self.progress.iter % self.cfg.solver.accumulation_steps == 0:
            self.modify_grad()

            optimizer.step().clear_grad()

            self.model_ema_step()

        return losses, accs

    def modify_grad(self):
        grad_cfg = self.cfg.solver.grad_clip
        # TODO: support advanced params for grad clip in the future
        params = self.model.parameters()
        if grad_cfg.name is None:
            return
        elif grad_cfg.name == "norm":
            optim.clip_grad_norm(params, grad_cfg.max_norm)
        elif grad_cfg.name == "value":
            optim.clip_grad_value(params, grad_cfg.lower, grad_cfg.upper)
        else:
            raise ValueError(f"Grad clip type '{grad_cfg.name}' not supported")

    def model_ema_step(self):
        """Implement momentum based Exponential Moving Average (EMA) for model states
        https://github.com/rwightman/pytorch-image-models/blob/master/timm/utils/model_ema.py

        Also inspired by Pycls https://github.com/facebookresearch/pycls/pull/138/, which is more
        flexible and efficient

        Heuristically, one can use a momentum of 0.9999 as used by Tensorflow and 0.9998 as used
        by timm, which updates model ema every iter. To be more efficient, one can set
        ``update_period`` to e.g. 8 or 32 to speed up your training, and decrease your momentum
        at scale: set ``momentum=0.9978`` from 0.9999 (32 times) when you ``update_period=32``.

        Also, to make model EMA really work (improve generalization), one should carefully tune
        the momentum based on various factors, e.g. the learning rate scheduler,
        the total batch size, the training epochs, e.t.c.

        To initialize a momentum in Pycls style, one set ``model_ema.alpha = 1e-5`` instead.
        Momentum will be calculated through ``_calculate_pycls_momentum``.
        """
        if self.ema is None:
            return

        ema_cfg = self.cfg.model_ema
        cur_iter, cur_epoch = self.progress.iter, self.progress.epoch
        if cur_iter % ema_cfg.update_period == 0:
            if cur_epoch > (ema_cfg.start_epoch or self.cfg.solver.warmup_epochs):
                momentum = (
                    ema_cfg.momentum
                    if ema_cfg.alpha is None
                    else _calculate_pycls_momentum(
                        alpha=ema_cfg.alpha,
                        total_batch_size=self.cfg.batch_size * dist.get_world_size(),
                        max_epoch=self.cfg.solver.max_epoch,
                        update_period=ema_cfg.update_period,
                    )
                )
            else:
                # copy model to ema
                momentum = 0.0

            if not hasattr(self, "_ema_states"):
                self._ema_states = (
                    list(self.ema.parameters()) + list(self.ema.buffers()),
                    list(self.model.parameters()) + list(self.model.buffers()),
                )

            for e, p in zip(*self._ema_states):
                # _inplace_add_(e, p, alpha=mge.tensor(momentum), beta=mge.tensor(1 - momentum))
                e._reset(e * momentum + p * (1 - momentum))


def _calculate_pycls_momentum(
    alpha: float, total_batch_size: int, max_epoch: int, update_period: int
):
    """pycls style momentum calculation which uses a relative model_ema to decouple momentum with
    other training hyper-parameters e.g.

        * training epochs
        * interval to update ema
        * batch sizes

    Usually the alpha is a tiny positive floating number, e.g. 1e-4 or 1e-5,
    with ``max_epoch=100``, ``total_batch_size=1024`` and ``update_period=32``, the ema
    momentum should be 0.996723175, which has roughly same behavior to the default setting.
    i.e. ``momentum=0.9999`` together with ``update_period=1``
    """
    return max(0, 1 - alpha * (total_batch_size / max_epoch * update_period))
