#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import bisect
import datetime
import math
import os
import pickle
import time
from typing import Optional

import megengine as mge
import megengine.distributed as dist
import megengine.module as M
from basecore.config import ConfigDict
from basecore.engine import BaseHook, BaseTrainer
from basecore.utils import (
    Checkpoint,
    MeterBuffer,
    cached_property,
    ensure_dir,
    get_last_call_deltatime,
)
from loguru import logger
from tensorboardX import SummaryWriter

from basecls.layers import compute_precise_bn_stats
from basecls.models import sync_model
from basecls.utils import default_logging, registers

from .tester import ClsTester

__all__ = [
    "CheckpointHook",
    "EvalHook",
    "LoggerHook",
    "LRSchedulerHook",
    "PreciseBNHook",
    "ResumeHook",
    "TensorboardHook",
]


def _create_checkpoint(trainer: BaseTrainer, save_dir: str) -> Checkpoint:
    """Create a checkpoint for save and resume"""
    model = trainer.model
    ema = trainer.ema
    ckpt_kws = {"ema": ema} if ema is not None else {}
    optim = trainer.solver.optimizer
    scaler = trainer.solver.grad_scaler
    progress = trainer.progress
    ckpt = Checkpoint(
        save_dir,
        model,
        tag_file=None,
        optimizer=optim,
        scaler=scaler,
        progress=progress,
        **ckpt_kws,
    )
    return ckpt


class CheckpointHook(BaseHook):
    """Hook for managing checkpoints during training.

    Effect during ``after_epoch`` and ``after_train`` procedure.

    Args:
        save_dir: checkpoint directory.
        save_every_n_epoch: interval for saving checkpoint. Default: ``1``
    """

    def __init__(self, save_dir: str = None, save_every_n_epoch: int = 1):
        super().__init__()
        ensure_dir(save_dir)
        self.save_dir = save_dir
        self.save_every_n_epoch = save_every_n_epoch

    def after_epoch(self):
        progress = self.trainer.progress
        ckpt = _create_checkpoint(self.trainer, self.save_dir)
        ckpt.save("latest.pkl")

        if progress.epoch % self.save_every_n_epoch == 0:
            progress_str = progress.progress_str_list()
            save_name = "_".join(progress_str[:-1]) + ".pkl"
            ckpt.save(save_name)
            logger.info(f"Save checkpoint {save_name} to {self.save_dir}")

    def after_train(self):
        # NOTE: usually final ema is not the best so we dont save it
        mge.save(
            {"state_dict": self.trainer.model.state_dict()},
            os.path.join(self.save_dir, "dumped_model.pkl"),
            pickle_protocol=pickle.DEFAULT_PROTOCOL,
        )


class EvalHook(BaseHook):
    """Hook for evaluating during training.

    Effect during ``after_epoch`` and ``after_train`` procedure.

    Args:
        save_dir: checkpoint directory.
        eval_every_n_epoch: interval for evaluating. Default: ``1``
    """

    def __init__(self, save_dir: str = None, eval_every_n_epoch: int = 1):
        super().__init__()
        ensure_dir(save_dir)
        self.save_dir = save_dir
        self.eval_every_n_epoch = eval_every_n_epoch
        self.best_acc1 = 0
        self.best_ema_acc1 = 0

    def after_epoch(self):
        trainer = self.trainer
        cfg = trainer.cfg
        model = trainer.model
        ema = trainer.ema
        progress = trainer.progress
        if progress.epoch % self.eval_every_n_epoch == 0 and progress.epoch != progress.max_epoch:
            self.test(cfg, model, ema)

    def after_train(self):
        trainer = self.trainer
        cfg = trainer.cfg
        model = trainer.model
        ema = trainer.ema
        # TODO: actually useless maybe when precise_bn is on
        sync_model(model)
        if ema is not None:
            sync_model(ema)
        self.test(cfg, model, ema)

    def test(self, cfg: ConfigDict, model: M.Module, ema: Optional[M.Module] = None):
        dataloader = registers.dataloaders.get(cfg.data.name).build(cfg, False)
        # FIXME: need atomic user_pop, maybe in MegEngine 1.5?
        # tester = BaseTester(model, dataloader, AccEvaluator())
        tester = ClsTester(cfg, model, dataloader)
        acc1, _ = tester.test()

        if acc1 > self.best_acc1:
            self.best_acc1 = acc1
            if dist.get_rank() == 0:
                mge.save(
                    {"state_dict": model.state_dict(), "acc1": self.best_acc1},
                    os.path.join(self.save_dir, "best_model.pkl"),
                    pickle_protocol=pickle.DEFAULT_PROTOCOL,
                )

        logger.info(
            f"Epoch: {self.trainer.progress.epoch}, Test Acc@1: {acc1:.3f}, "
            f"Best Test Acc@1: {self.best_acc1:.3f}"
        )

        if ema is None:
            return

        tester_ema = ClsTester(cfg, ema, dataloader)
        ema_acc1, _ = tester_ema.test()

        if ema_acc1 > self.best_ema_acc1:
            self.best_ema_acc1 = ema_acc1
            if dist.get_rank() == 0:
                mge.save(
                    {"state_dict": ema.state_dict(), "acc1": self.best_ema_acc1},
                    os.path.join(self.save_dir, "best_ema_model.pkl"),
                    pickle_protocol=pickle.DEFAULT_PROTOCOL,
                )

        logger.info(
            f"Epoch: {self.trainer.progress.epoch}, EMA Acc@1: {ema_acc1:.3f}, "
            f"Best EMA Acc@1: {self.best_ema_acc1:.3f}"
        )


class LoggerHook(BaseHook):
    """Hook for logging during training.

    Effect during ``before_train``, ``after_train``, ``before_iter`` and ``after_iter`` procedure.

    Args:
        log_every_n_iter: interval for logging. Default: ``20``
    """

    def __init__(self, log_every_n_iter: int = 20):
        super().__init__()
        self.log_every_n_iter = log_every_n_iter
        self.meter = MeterBuffer(self.log_every_n_iter)

    def before_train(self):
        trainer = self.trainer
        progress = trainer.progress

        default_logging(trainer.cfg, trainer.model)

        logger.info(f"Starting training from epoch {progress.epoch}, iteration {progress.iter}")

        self.start_training_time = time.perf_counter()

    def after_train(self):
        total_training_time = time.perf_counter() - self.start_training_time
        total_time_str = str(datetime.timedelta(seconds=total_training_time))
        logger.info(
            "Total training time: {} ({:.4f} s / iter)".format(
                total_time_str, self.meter["iters_time"].global_avg
            )
        )

    def before_iter(self):
        self.iter_start_time = time.perf_counter()

    def after_iter(self):
        single_iter_time = time.perf_counter() - self.iter_start_time

        delta_time = get_last_call_deltatime()
        if delta_time is None:
            delta_time = single_iter_time

        self.meter.update(
            {
                "iters_time": single_iter_time,  # to get global average iter time
                "eta_iter_time": delta_time,  # to get ETA time
                "extra_time": delta_time - single_iter_time,  # to get extra time
            }
        )

        trainer = self.trainer
        progress = trainer.progress
        epoch_id, iter_id = progress.epoch, progress.iter
        max_epoch, max_iter = progress.max_epoch, progress.max_iter

        if iter_id % self.log_every_n_iter == 0 or (iter_id == 1 and epoch_id == 1):
            log_str_list = []

            # step info string
            log_str_list.append(str(progress))

            # loss string
            log_str_list.append(self.get_loss_str(trainer.meter))

            # stat string
            log_str_list.append(self.get_stat_str(trainer.meter))

            # other training info like learning rate.
            log_str_list.append(self.get_train_info_str())

            # memory useage.
            log_str_list.append(self.get_memory_str(trainer.meter))

            # time string
            left_iters = max_iter - iter_id + (max_epoch - epoch_id) * max_iter
            time_str = self.get_time_str(left_iters)
            log_str_list.append(time_str)

            # filter empty strings
            log_str_list = [s for s in log_str_list if len(s) > 0]
            log_str = ", ".join(log_str_list)
            logger.info(log_str)

            # reset meters in trainer
            trainer.meter.reset()

    def get_loss_str(self, meter):
        """Get loss information during trainging process."""
        loss_dict = meter.get_filtered_meter(filter_key="loss")
        loss_str = ", ".join(
            [f"{name}:{value.latest:.3f}({value.avg:.3f})" for name, value in loss_dict.items()]
        )
        return loss_str

    def get_stat_str(self, meter):
        """Get stat information during trainging process."""
        stat_dict = meter.get_filtered_meter(filter_key="stat")
        stat_str = ", ".join(
            [f"{name}:{value.latest:.3f}({value.avg:.3f})" for name, value in stat_dict.items()]
        )
        return stat_str

    def get_memory_str(self, meter):
        """Get memory information during trainging process."""

        def mem_in_Mb(mem_value):
            return math.ceil(mem_value / 1024 / 1024)

        mem_dict = meter.get_filtered_meter(filter_key="memory")
        mem_str = ", ".join(
            [
                f"{name}:{mem_in_Mb(value.latest)}({mem_in_Mb(value.avg)})Mb"
                for name, value in mem_dict.items()
            ]
        )
        return mem_str

    def get_train_info_str(self):
        """Get training process related information such as learning rate."""
        # extra info to display, such as learning rate
        trainer = self.trainer
        lr = trainer.solver.optimizer.param_groups[0]["lr"]
        lr_str = f"lr:{lr:.3e}"
        loss_scale = trainer.solver.grad_scaler.scale_factor
        loss_scale_str = f", amp_loss_scale:{loss_scale:.1f}" if trainer.cfg.amp.enabled else ""
        return lr_str + loss_scale_str

    def get_time_str(self, left_iters: int) -> str:
        """Get time related information sucn as data_time, train_time, ETA and so on."""
        # time string
        trainer = self.trainer
        time_dict = trainer.meter.get_filtered_meter(filter_key="time")
        train_time_str = ", ".join(
            [f"{name}:{value.avg:.3f}s" for name, value in time_dict.items()]
        )
        train_time_str += ", extra_time:{:.3f}s, ".format(self.meter["extra_time"].avg)

        eta_seconds = self.meter["eta_iter_time"].global_avg * left_iters
        eta_string = "ETA:{}".format(datetime.timedelta(seconds=int(eta_seconds)))
        time_str = train_time_str + eta_string
        return time_str


class LRSchedulerHook(BaseHook):
    """Hook for learning rate scheduling during training.

    Effect during ``before_epoch`` procedure.
    """

    def before_epoch(self):
        trainer = self.trainer
        epoch_id = trainer.progress.epoch
        cfg = trainer.cfg.solver

        lr_factor = self.get_lr_factor(cfg, epoch_id)

        if epoch_id <= cfg.warmup_epochs:
            alpha = (epoch_id - 1) / cfg.warmup_epochs
            lr_factor *= cfg.warmup_factor * (1 - alpha) + alpha

        scaled_lr = self.total_lr * lr_factor
        for param_group in trainer.solver.optimizer.param_groups:
            param_group["lr"] = scaled_lr

    def get_lr_factor(self, cfg: ConfigDict, epoch_id: int) -> float:
        """Calculate learning rate factor.

        It supports ``"step"``, ``"linear"``, ``"cosine"``, ``"exp"``, and ``"rel_exp"`` schedule.

        Args:
            cfg: config for training.
            epoch_id: current epoch.

        Returns:
            Learning rate factor.
        """
        if cfg.lr_schedule == "step":
            return cfg.lr_decay_factor ** bisect.bisect_left(cfg.lr_decay_steps, epoch_id)
        elif cfg.lr_schedule == "linear":
            alpha = 1 - (epoch_id - 1) / cfg.max_epoch
            return (1 - cfg.lr_min_factor) * alpha + cfg.lr_min_factor
        elif cfg.lr_schedule == "cosine":
            alpha = 0.5 * (1 + math.cos(math.pi * (epoch_id - 1) / cfg.max_epoch))
            return (1 - cfg.lr_min_factor) * alpha + cfg.lr_min_factor
        elif cfg.lr_schedule == "exp":
            return cfg.lr_decay_factor ** (epoch_id - 1)
        elif cfg.lr_schedule == "rel_exp":
            if cfg.lr_min_factor <= 0:
                raise ValueError(
                    "Exponential lr schedule requires lr_min_factor to be greater than 0"
                )
            return cfg.lr_min_factor ** ((epoch_id - 1) / cfg.max_epoch)
        else:
            raise NotImplementedError(f"Learning rate schedule '{cfg.lr_schedule}' not supported")

    @cached_property
    def total_lr(self) -> float:
        """Total learning rate."""
        cfg = self.trainer.cfg.solver
        total_lr = cfg.basic_lr * dist.get_world_size()  # linear scaling rule
        return total_lr


class PreciseBNHook(BaseHook):
    """Hook for precising BN during training.

    Effect during ``after_epoch`` procedure.

    Args:
        precise_every_n_epoch: interval for precising BN. Default: ``1``
    """

    def __init__(self, precise_every_n_epoch: int = 1):
        super().__init__()
        self.precise_every_n_epoch = precise_every_n_epoch

    def before_train(self):
        if self.precise_every_n_epoch == -1:
            self.precise_every_n_epoch = self.trainer.progress.max_epoch

    def after_epoch(self):
        trainer = self.trainer
        if (
            trainer.progress.epoch % self.precise_every_n_epoch == 0
            and trainer.cfg.bn.num_samples_precise > 0
        ):
            logger.info(f"Apply Precising BN at epoch{trainer.progress.epoch}")
            compute_precise_bn_stats(trainer.cfg, trainer.model, trainer.dataloader)
            if trainer.ema is not None:
                logger.info(f"Apply Precising BN for EMA at epoch{trainer.progress.epoch}")
                compute_precise_bn_stats(trainer.cfg, trainer.ema, trainer.dataloader)


class ResumeHook(BaseHook):
    """Hook for resuming training process.

    Effect during ``before_train`` procedure.

    Args:
        save_dir: checkpoint directory.
        resume: enable resume or not. Default: ``False``
    """

    def __init__(self, save_dir: int = None, resume: bool = False):
        super().__init__()
        ensure_dir(save_dir)
        self.save_dir = save_dir
        self.resume = resume

    def before_train(self):
        trainer = self.trainer
        if self.resume:
            progress = trainer.progress
            ckpt = _create_checkpoint(self.trainer, self.save_dir)
            filename = ckpt.get_checkpoint_file("latest.pkl")
            logger.info(f"Load checkpoint from {filename}")
            ckpt.resume(filename)
            # since ckpt is dumped after every epoch,
            # resume training requires epoch + 1 and set iter to 1
            progress.epoch += 1
            progress.iter = 1


class TensorboardHook(BaseHook):
    """Hook for tensorboard during training.

    Effect during ``before_train``, ``after_train`` and ``after_iter`` procedure.

    Args:
        log_dir: tensorboard directory.
        log_every_n_iter: interval for logging. Default: ``20``
        scalar_type: statistic to record, supports ``"latest"``, ``"avg"``, ``"global_avg"`` and
            ``"median"``. Default: ``"latest"``
    """

    def __init__(self, log_dir: str, log_every_n_iter: int = 20, scalar_type: str = "latest"):
        super().__init__()
        if scalar_type not in ("latest", "avg", "global_avg", "median"):
            raise ValueError(f"Tensorboard scalar type '{scalar_type}' not supported")
        ensure_dir(log_dir)
        self.log_dir = log_dir
        self.log_every_n_iter = log_every_n_iter
        self.scalar_type = scalar_type

    def before_train(self):
        self.writer = SummaryWriter(self.log_dir)

    def after_train(self):
        self.writer.close()

    def after_iter(self):
        trainer = self.trainer
        epoch_id, iter_id = trainer.progress.epoch, trainer.progress.iter
        if iter_id % self.log_every_n_iter == 0 or (iter_id == 1 and epoch_id == 1):
            self.write(context=trainer)

    def write(self, context):
        cur_iter = self.calc_iter(context.progress)
        for key, meter in context.meter.items():
            value = getattr(meter, self.scalar_type, meter.latest)
            for prefix in ("loss", "stat", "time", "memory"):
                if prefix in key:
                    key = f"{prefix}/{key}"
                    break
            self.writer.add_scalar(key, value, cur_iter)
        # write lr into tensorboard
        lr = context.solver.optimizer.param_groups[0]["lr"]
        self.writer.add_scalar("lr", lr, cur_iter)
        # write loss_scale into tensorboard
        if context.cfg.amp.enabled:
            loss_scale = context.solver.grad_scaler.scale_factor
            self.writer.add_scalar("amp_loss_scale", loss_scale, cur_iter)

    @classmethod
    def calc_iter(cls, progress):
        return (progress.epoch - 1) * progress.max_iter + progress.iter - 1
