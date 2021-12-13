#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import argparse
import importlib
import json
import multiprocessing as mp
import os
import pathlib
import sys

import megengine as mge
import megengine.distributed as dist
from basecore.config import ConfigDict
from loguru import logger

from basecls.engine import ClsTester
from basecls.models import build_model, load_model
from basecls.utils import default_logging, registers, set_nccl_env, set_num_threads, setup_logger


def make_parser() -> argparse.ArgumentParser:
    """Build args parser for testing script.

    Returns:
        The args parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", type=str, help="testing directory")
    return parser


@logger.catch
def worker(args: argparse.Namespace):
    """Worker function for testing script.

    Args:
        args: args for testing script.
    """
    logger.info(f"Init process group for gpu{dist.get_rank()} done")
    args.dir = os.path.abspath(args.dir)
    setup_logger(args.dir, "test_all_log.txt", to_loguru=True)
    logger.info(f"args: {args}")

    result = dict()
    for f in pathlib.Path(args.dir).glob("**/*.py"):
        sys.path.append(os.path.dirname(f))

        module_name = os.path.splitext(os.path.basename(f))[0]
        current_network = importlib.import_module(module_name)
        cfg = current_network.Cfg()

        weight_path = f"{os.path.splitext(f)[0]}.pkl"
        if os.path.isfile(weight_path):
            cfg.weights = weight_path
        else:
            sys.path.pop(-1)
            continue

        cfg.set_mode("freeze")

        if cfg.fastrun:
            logger.info("Using fastrun mode...")
            mge.functional.debug_param.set_execution_strategy("PROFILE")

        tester = build(cfg)
        acc1, acc5 = tester.test()

        result[module_name] = dict(acc1=acc1, acc5=acc5)

        sys.path.pop(-1)

    logger.info(json.dumps(result, indent=4))
    with open("result.json", "w") as f:
        json.dump(result, f)


def build(cfg: ConfigDict):
    """Build function for testing script.

    Args:
        cfg: config for testing.

    Returns:
        A tester.
    """
    model = build_model(cfg)
    load_model(model, cfg.weights)
    model.eval()

    default_logging(cfg, model)

    dataloader = registers.dataloaders.get(cfg.data.name).build(cfg, False)
    # FIXME: need atomic user_pop, maybe in MegEngine 1.5?
    # tester = BaseTester(model, dataloader, AccEvaluator())
    return ClsTester(cfg, model, dataloader)


def main():
    """Main function for testing script."""
    parser = make_parser()
    args = parser.parse_args()

    mp.set_start_method("spawn")

    set_nccl_env()
    set_num_threads()

    if not os.path.exists(args.dir):
        raise ValueError("Directory does not exist")

    device_count = mge.device.get_device_count("gpu")

    if device_count == 0:
        logger.warning("No GPU was found, testing on CPU")
        worker(args)
    elif device_count > 1:
        mp_worker = dist.launcher(worker)
        mp_worker(args)
    else:
        worker(args)


if __name__ == "__main__":
    main()
