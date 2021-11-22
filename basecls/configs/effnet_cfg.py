#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
from basecls.layers import NORM_TYPES

from .base_cfg import BaseConfig

__all__ = ["EffNetConfig", "EffNetLiteConfig"]

_cfg = dict(
    model=dict(
        name="EffNet",
    ),
    bn=dict(
        precise_every_n_epoch=1,
    ),
    loss=dict(
        label_smooth=0.1,
    ),
    augments=dict(
        name="RandAugment",
        resize=dict(
            interpolation="bicubic",
        ),
        rand_erase=dict(
            prob=0.25,
            mode="pixel",
        ),
    ),
    solver=dict(
        basic_lr=0.05,
        weight_decay=(
            (0, "bias"),
            (0, NORM_TYPES),
            1e-5,
        ),
        nesterov=True,
        max_epoch=350,
        lr_schedule="cosine",
    ),
    model_ema=dict(
        enabled=True,
        momentum=0.99998,
    ),
)


class EffNetConfig(BaseConfig):
    def __init__(self, values_or_file=None, **kwargs):
        super().__init__(_cfg)
        self.merge(values_or_file, **kwargs)


_cfg_lite = dict(
    augments=dict(
        name="ColorAugment",
        resize=dict(
            interpolation="bilinear",
        ),
        rand_erase=dict(
            prob=0.0,
        ),
    ),
)


class EffNetLiteConfig(EffNetConfig):
    def __init__(self, values_or_file=None, **kwargs):
        super().__init__(_cfg_lite)
        self.merge(values_or_file, **kwargs)
