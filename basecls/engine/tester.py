#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import datetime
import time

import megengine as mge
import megengine.distributed as dist
import megengine.functional as F
import megengine.module as M
from basecore.config import ConfigDict
from basecore.engine import BaseTester
from basecore.network import adjust_stats
from basecore.utils import log_every_n_seconds
from loguru import logger
from megengine import jit

from basecls.data import DataLoaderType
from basecls.layers import Preprocess

__all__ = ["ClsTester"]


class ClsTester(BaseTester):
    def __init__(self, cfg: ConfigDict, model: M.Module, dataloader: DataLoaderType):
        super().__init__(model, dataloader)
        self.cfg = cfg
        self.preprocess = Preprocess(cfg.preprocess.img_mean, cfg.preprocess.img_std)

    def test(self, warm_iters=5, log_seconds=5):
        cnt = 0
        acc1 = 0
        acc5 = 0

        total_iters = len(self.dataloader)
        warm_iters = min(warm_iters, total_iters)

        total_time = 0
        with adjust_stats(self.model, training=False) as model:
            model_step = jit.trace(model, symbolic=True) if self.cfg.trace else model
            for iters, data in enumerate(self.dataloader, 1):
                if iters == warm_iters + 1:
                    total_time = 0

                samples, targets = self.preprocess(data)
                start_time = time.perf_counter()
                outputs = model_step(samples)
                mge._full_sync()  # use full_sync func to sync launch queue for dynamic execution
                total_time += time.perf_counter() - start_time

                accs = F.metric.topk_accuracy(outputs, targets, (1, 5))
                cnt += targets.shape[0]
                acc1 += accs[0].item() * 100 * targets.shape[0]
                acc5 += accs[1].item() * 100 * targets.shape[0]

                if log_seconds > 0:
                    count_iters = iters - warm_iters if iters > warm_iters else iters
                    time_per_iter = total_time / count_iters
                    infer_eta = (total_iters - iters) * time_per_iter
                    log_every_n_seconds(
                        "Inference process {}/{}, average speed:{:.4f}s/iters. ETA:{}".format(
                            iters,
                            total_iters,
                            time_per_iter,
                            datetime.timedelta(seconds=int(infer_eta)),
                        ),
                        n=log_seconds,
                    )
            logger.info(
                "Finish inference process, total time:{}, average speed:{:.4f}s/iters.".format(
                    datetime.timedelta(seconds=int(total_time)),
                    total_time / (len(self.dataloader) - warm_iters),
                )
            )

        cnt = dist.functional.all_reduce_sum(mge.Tensor(cnt)).item()
        acc1 = dist.functional.all_reduce_sum(mge.Tensor(acc1)).item() / cnt
        acc5 = dist.functional.all_reduce_sum(mge.Tensor(acc5)).item() / cnt
        if dist.get_rank() == 0:
            logger.info(f"Test Acc@1: {acc1:.3f}, Acc@5: {acc5:.3f}")
        return acc1, acc5
