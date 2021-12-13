#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import argparse
import datetime
import time
from typing import Sequence, Tuple, Union

import numpy as np
import timm
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.jit as jit
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from basecore.utils import log_every_n_seconds
from loguru import logger

from basecls.data.fake_data import FakeDataLoader
from basecls.utils import set_nccl_env, set_num_threads

DEVICE = dict(cpu="cpu", gpu="cuda")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default="resnet50", type=str)
    parser.add_argument("--mode", default="eval", type=str)
    parser.add_argument("-d", "--device", default="gpu", type=str)
    parser.add_argument("--amp", default=0, type=int)
    parser.add_argument("--fastrun", action="store_true")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("-b", "--batch-size", default=32, type=int)
    parser.add_argument("--channel", default=3, type=int)
    parser.add_argument("--height", default=224, type=int)
    parser.add_argument("--width", default=224, type=int)
    parser.add_argument("-n", "--world-size", default=8, type=int)
    parser.add_argument("--warm-iters", default=50, type=int)
    parser.add_argument("-t", "--total-iters", default=200, type=int)
    parser.add_argument("--log-seconds", default=2, type=int)
    args = parser.parse_args()

    set_nccl_env()
    set_num_threads()

    if args.world_size == 1:
        worker(0, args)
    else:
        mp.spawn(worker, nprocs=args.world_size, args=(args,))


@logger.catch
def worker(local_rank, args):
    if args.world_size > 1:
        dist.init_process_group(
            backend="nccl" if args.device == "gpu" else "gloo",
            init_method="tcp://127.0.0.1:6789",
            world_size=args.world_size,
            rank=local_rank,
        )

    if local_rank != 0:
        logger.remove()
    logger.info(f"args: {args}")

    if args.fastrun:
        logger.info("Using fastrun mode...")
        torch.backends.cudnn.benchmark = True

    device = torch.device(DEVICE[args.device], local_rank)
    if args.device == "gpu":
        torch.cuda.set_device(local_rank)

    model = timm.create_model(args.model, pretrained=False).to(device)
    if args.world_size > 1:
        model = nn.parallel.DistributedDataParallel(model)

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

    bench = BenchCls(model, dataloader, device, args.trace, args.amp)
    bench.benchmark(args.warm_iters, args.log_seconds)


class ClsBench:
    def __init__(self, model, dataloader, device: torch.device, trace: bool = False):
        self.model = model
        self.dataloader = dataloader
        self.preprocess = Preprocess(mean=127, std=128, device=device)
        self.device = device

        if trace:
            if isinstance(self.model, nn.parallel.DistributedDataParallel):
                self.model.module = jit.script(self.model.module)
            else:
                self.model = jit.script(self.model)

    def benchmark(self, warm_iters=50, log_seconds=2):
        total_iters = len(self.dataloader) - warm_iters

        total_time = 0
        for i, data in enumerate(self.dataloader, 1):
            if i == warm_iters + 1:
                total_time = 0

            samples, targets = self.preprocess(data)

            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()

            self.model_step(samples, targets)

            if self.device.type == "cuda":
                torch.cuda.synchronize()
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
    def __init__(
        self, model, dataloader, device: torch.device, trace: bool = False, amp_version: int = 0
    ):
        model.train()
        super().__init__(model, dataloader, device, trace)

        self.opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)

        self.amp_version = amp_version
        self.scaler = (
            amp.GradScaler(init_scale=65536.0, growth_interval=2000)
            if amp_version == 2
            else amp.GradScaler(init_scale=128.0, growth_interval=0)
        )

    def model_step(self, samples, targets):
        with amp.autocast(enabled=self.amp_version > 0):
            pred = self.model(samples)
            loss = F.cross_entropy(pred, targets)
        self.opt.zero_grad()
        if self.amp_version > 0:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            loss.backward()
            self.opt.step()


class EvalBench(ClsBench):
    def __init__(
        self, model, dataloader, device: torch.device, trace: bool = False, amp_version: int = 0
    ):
        model.eval()
        super().__init__(model, dataloader, device, trace)

        self.amp_version = amp_version

    def model_step(self, samples, targets):
        with amp.autocast(enabled=self.amp_version > 0):
            self.model(samples)


class Preprocess(nn.Module):
    def __init__(
        self,
        mean: Union[float, Sequence[float]],
        std: Union[float, Sequence[float]],
        device=torch.device,
    ):
        super().__init__()
        self.mean = torch.tensor(mean, dtype=torch.float32, device=device).reshape(1, -1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32, device=device).reshape(1, -1, 1, 1)
        self.device = device

    def forward(self, inputs: Sequence[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        samples, targets = [torch.tensor(x, device=self.device) for x in inputs]
        samples = (samples - self.mean) / self.std
        targets = targets.long()
        return samples, targets


if __name__ == "__main__":
    main()
