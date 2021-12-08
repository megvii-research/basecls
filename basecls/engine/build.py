#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import os
from typing import List

import megengine.distributed as dist
from basecore.config import ConfigDict
from basecore.engine import BaseHook
from basecore.utils import str_timestamp

from basecls.utils import registers

from .hooks import (
    CheckpointHook,
    EvalHook,
    LoggerHook,
    LRSchedulerHook,
    PreciseBNHook,
    ResumeHook,
    TensorboardHook,
)

__all__ = ["DefaultHooks"]


@registers.hooks.register()
class DefaultHooks:
    """The default hooks factory.

    It combines :py:class:`~basecls.engine.LRSchedulerHook` ->
    :py:class:`~basecls.engine.PreciseBNHook` -> :py:class:`~basecls.engine.ResumeHook` ->
    :py:class:`~basecls.engine.TensorboardHook` -> :py:class:`~basecls.engine.LoggerHook` ->
    :py:class:`~basecls.engine.CheckpointHook` -> :py:class:`~basecls.engine.EvalHook`.
    """

    @classmethod
    def build(cls, cfg: ConfigDict) -> List[BaseHook]:
        """Build function with a simple strategy.

        Args:
            cfg: config for setting hooks.

        Returns:
            A hook list.
        """
        output_dir = cfg.output_dir
        hook_list = [
            LRSchedulerHook(),
            PreciseBNHook(cfg.bn.precise_every_n_epoch),
            ResumeHook(output_dir, cfg.resume),
        ]

        if dist.get_rank() == 0:
            # Since LoggerHook will reset value, TensorboardHook should be added before LoggerHook
            hook_list.append(
                TensorboardHook(
                    os.path.join(output_dir, "tensorboard", str_timestamp()), cfg.tb_every_n_iter
                )
            )
            hook_list.append(LoggerHook(cfg.log_every_n_iter))
            hook_list.append(CheckpointHook(output_dir, cfg.save_every_n_epoch))

        # Hooks better work after CheckpointHook
        hook_list.append(EvalHook(output_dir, cfg.eval_every_n_epoch))

        return hook_list
