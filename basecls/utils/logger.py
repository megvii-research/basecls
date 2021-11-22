#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import os

import megengine.distributed as dist
import megengine.module as M
from basecore.config import ConfigDict
from basecore.network import adjust_stats
from basecore.utils import get_env_info_table, setup_mge_logger, str_timestamp
from loguru import logger
from megengine.utils.module_stats import module_stats

import basecls

__all__ = ["default_logging", "setup_logger"]


def default_logging(cfg: ConfigDict, model: M.Module):
    logger.info(f"\nSystem env:\n{get_env_info_table(basecls=basecls.__version__)}")

    # logging config
    logger.info(f"\nTraining full config:\n{cfg}")
    # logging changed value in config
    base_cfg = cfg.__class__.__base__()
    logger.info(f"Diff value in config:\n{cfg.diff(base_cfg)}")

    # logging model
    logger.info(f"\nModel structure:\n{model}")
    logger.info("Model status:")
    with adjust_stats(model, training=False) as eval_model:
        input_size = (
            1,
            1 if cfg.preprocess.img_color_space == "GRAY" else 3,
            cfg.preprocess.img_size,
            cfg.preprocess.img_size,
        )
        module_stats(eval_model, input_shapes=input_size)


def setup_logger(log_path: str, log_file: str, to_loguru: bool = False):
    filename, suffix = os.path.splitext(log_file)
    if dist.get_rank() == 0:
        time_stamp = str_timestamp()
        logger.add(os.path.join(log_path, f"{filename}_{time_stamp}{suffix}"))
    else:
        # logger to stdout/stderr only available on main process
        logger.remove()
    setup_mge_logger(path=log_path, log_level="INFO", to_loguru=to_loguru)
