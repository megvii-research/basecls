#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import argparse
import datetime
import multiprocessing as mp
import time

import megengine as mge
import megengine.amp as amp
import megengine.autodiff as autodiff
import megengine.distributed as dist
import megengine.functional as F
import megengine.jit as jit
import megengine.optimizer as optim
from basecore.utils import log_every_n_seconds
from loguru import logger

from basecls.data.fake_data import FakeDataLoader
from basecls.layers import Preprocess
from basecls.utils import registers, set_nccl_env, set_num_threads


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default="resnet50", type=str)
    parser.add_argument("--mode", default="eval", type=str)
    parser.add_argument("-d", "--device", default="gpu", type=str)
    parser.add_argument("--amp", default=0, type=int)
    parser.add_argument("--fastrun", action="store_true")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--dtr", action="store_true")
    parser.add_argument("-b", "--batch-size", default=32, type=int)
    parser.add_argument("--channel", default=3, type=int)
    parser.add_argument("--height", default=224, type=int)
    parser.add_argument("--width", default=224, type=int)
    parser.add_argument("-n", "--world-size", default=8, type=int)
    parser.add_argument("--warm-iters", default=50, type=int)
    parser.add_argument("-t", "--total-iters", default=200, type=int)
    parser.add_argument("--log-seconds", default=2, type=int)
    args = parser.parse_args()

    mp.set_start_method("spawn")

    set_nccl_env()
    set_num_threads()

    if args.world_size == 1:
        worker(args)
    else:
        dist.launcher(worker, n_gpus=args.world_size)(args)


@logger.catch
def worker(args: argparse.Namespace):
    if dist.get_rank() != 0:
        logger.remove()
    logger.info(f"args: {args}")

    if args.fastrun:
        logger.info("Using fastrun mode...")
        mge.functional.debug_param.set_execution_strategy("PROFILE")

    if args.dtr:
        logger.info("Enabling DTR...")
        mge.dtr.enable()

    mge.set_default_device(f"{args.device}{dist.get_rank()}")

    model = registers.models.get(args.model)(head=dict(w_out=1000))

    dataloader = FakeDataLoader(
        args.batch_size,
        (args.height, args.width),
        args.channel,
        length=args.warm_iters + args.total_iters,
        num_classes=1000,
    )

    if args.mode == "train":
        BenchCls = TrainBench
    elif args.mode == "eval":
        BenchCls = EvalBench
    else:
        raise NotImplementedError(f"Benchmark mode '{args.mode}' not supported")

    bench = BenchCls(model, dataloader, args.trace, args.amp)
    bench.benchmark(args.warm_iters, args.log_seconds)


class ClsBench:
    def __init__(self, model, dataloader, trace: bool = False):
        self.model = model
        self.dataloader = dataloader
        self.preprocess = Preprocess(mean=127, std=128)

        if trace:
            self.model_step = jit.trace(self.model_step, symbolic=True)

    def benchmark(self, warm_iters=50, log_seconds=2):
        total_iters = len(self.dataloader) - warm_iters

        total_time = 0
        for i, data in enumerate(self.dataloader, 1):
            if i == warm_iters + 1:
                total_time = 0

            samples, targets = self.preprocess(data)

            mge._full_sync()
            t = time.perf_counter()

            self.model_step(samples, targets)

            mge._full_sync()
            total_time += time.perf_counter() - t

            if log_seconds > 0:
                cnt = i - warm_iters if i > warm_iters else i
                tot = total_iters if i > warm_iters else warm_iters
                cycle = total_time / cnt
                eta = (tot - cnt) * cycle
                log_every_n_seconds(
                    "{} process {}/{}, average speed:{:0.3f}ms/iters. ETA:{}".format(
                        "Benchmark" if i > warm_iters else "Warmup",
                        cnt,
                        tot,
                        cycle * 1000,
                        datetime.timedelta(seconds=int(eta)),
                    ),
                    n=log_seconds,
                )

        avg_speed_ms = total_time / total_iters * 1000
        logger.info(
            "Benchmark total time:{}, average speed:{:0.3f}ms/iters.".format(
                datetime.timedelta(seconds=int(total_time)), avg_speed_ms
            )
        )
        return avg_speed_ms

    def model_step(self, samples, targets):
        raise NotImplementedError


class TrainBench(ClsBench):
    def __init__(self, model, dataloader, trace: bool = False, amp_version: int = 0):
        model.train()
        super().__init__(model, dataloader, trace)

        self.opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)

        self.gm = autodiff.GradManager()
        callbacks = (
            [dist.make_allreduce_cb("mean", dist.WORLD)] if dist.get_world_size() > 1 else None
        )
        self.gm.attach(model.parameters(), callbacks=callbacks)

        self.amp_version = amp_version
        self.scaler = (
            amp.GradScaler(init_scale=65536.0, growth_interval=2000)
            if amp_version == 2
            else amp.GradScaler(init_scale=128.0, growth_interval=0)
        )

    def model_step(self, samples, targets):
        with self.gm:
            with amp.autocast(enabled=self.amp_version > 0):
                pred = self.model(samples)
                loss = F.loss.cross_entropy(pred, targets)
            if self.amp_version > 0:
                self.scaler.backward(self.gm, loss, update_scale=False)
                self.scaler.update()
            else:
                self.gm.backward(loss)
        self.opt.step().clear_grad()


class EvalBench(ClsBench):
    def __init__(self, model, dataloader, trace: bool = False, amp_version: int = 0):
        model.eval()
        super().__init__(model, dataloader, trace)

        self.amp_version = amp_version

    def model_step(self, samples, targets):
        with amp.autocast(enabled=self.amp_version > 0):
            self.model(samples)


if __name__ == "__main__":
    main()
