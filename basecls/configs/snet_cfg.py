#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
from basecls.layers import NORM_TYPES

from .base_cfg import BaseConfig

__all__ = ["SNetConfig"]

_cfg = dict(
    batch_size=128,
    bn=dict(
        precise_every_n_epoch=1,
    ),
    loss=dict(
        label_smooth=0.1,
    ),
    augments=dict(
        color_aug=dict(
            lighting=0.0,
        ),
    ),
    solver=dict(
        basic_lr=0.0625,
        weight_decay=(
            (0, "bias"),
            (0, NORM_TYPES),
            4e-5,
        ),
        max_epoch=240,
        warmup_epochs=0,
        lr_schedule="linear",
    ),
)


class SNetConfig(BaseConfig):
    def __init__(self, values_or_file=None, **kwargs):
        super().__init__(_cfg)
        self.merge(values_or_file, **kwargs)
