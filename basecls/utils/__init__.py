#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
from collections import abc
from typing import Any, Mapping

import megfile
from basecore.utils import Registry

from .env import set_nccl_env, set_num_threads
from .logger import default_logging, setup_logger


class registers:
    """All registried module could be found here."""

    augments = Registry("augments")
    dataloaders = Registry("dataloaders")
    hooks = Registry("hooks")
    models = Registry("models")
    solvers = Registry("solvers")
    trainers = Registry("trainers")


def recursive_update(d: Mapping[str, Any], u: Mapping[str, Any]):
    for k, v in u.items():
        if isinstance(d.get(k), abc.Mapping) and isinstance(v, abc.Mapping):
            d[k] = recursive_update(d[k], v)
        else:
            d[k] = v
    return d


_EXCLUDE = {}
__all__ = [k for k in globals().keys() if k not in _EXCLUDE and not k.startswith("_")]
