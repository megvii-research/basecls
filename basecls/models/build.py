#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import megengine as mge
import megengine.distributed as dist
import megengine.module as M
from basecore.config import ConfigDict
from loguru import logger
from megfile import smart_open

from basecls.utils import registers

__all__ = ["build_model", "load_model", "sync_model"]


def build_model(cfg: ConfigDict) -> M.Module:
    """The factory function to build model.

    Note:
        if ``cfg.model`` does not have the attr ``head``, this function will build model with
        the default head. Otherwise if ``cfg.model.head`` is ``None``, this function will build
        model without any head.

    Note:
        if ``cfg.model.head`` does not have the attr ``w_out`` and ``cfg.num_classes`` exists,
        ``w_out`` will be overridden by ``cfg.num_classes``.

    Args:
        cfg: config for building model.

    Returns:
        A model.
    """
    model_args = cfg.model.to_dict()
    model_name = model_args.pop("name", None)
    if model_name is None:
        raise ValueError("Model name is missing")

    # override w_out by the global number of classes if exists
    if getattr(cfg, "num_classes", None) is not None:
        model_args.setdefault("head", dict())
        if model_args["head"] is not None:
            model_args["head"].setdefault("w_out", cfg.num_classes)

    logger.info(f"Building model named {model_name}")
    model = registers.models.get(model_name)(**model_args)
    return model


def load_model(model: M.Module, weight_path: str, strict: bool = True):
    """Load model weights.

    Args:
        model: model for loading weights.
        weight_path: weight path, both local path and OSS path are supported.
        strict: load weights in strict mode or not. Default: ``True``
    """
    logger.info(f"Loading model weights from {weight_path}")
    with smart_open(weight_path, "rb") as f:
        state_dict = mge.load(f)
    # keyname model could be found in checkpoint
    if "model" in state_dict:
        state_dict = state_dict["model"]
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict, strict=strict)


def sync_model(model: M.Module):
    """Sync parameters and buffers.

    Args:
        model: model for syncing.
    """
    if dist.get_world_size() > 1:
        dist.bcast_list_(model.parameters())
        dist.bcast_list_(model.buffers())
