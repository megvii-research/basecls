#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
from basecls.layers import NORM_TYPES

from .base_cfg import BaseConfig

__all__ = ["VGGConfig"]

_cfg = dict(
    model=dict(
        name="VGG",
    ),
    bn=dict(
        precise_every_n_epoch=1,
    ),
    solver=dict(
        basic_lr=0.025,
        weight_decay=(
            (0, "bias"),
            (0, NORM_TYPES),
            5e-5,
        ),
        nesterov=True,
        lr_schedule="cosine",
    ),
)


class VGGConfig(BaseConfig):
    def __init__(self, values_or_file=None, **kwargs):
        super().__init__(_cfg)
        self.merge(values_or_file, **kwargs)
