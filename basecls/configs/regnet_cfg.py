#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
from basecls.layers import NORM_TYPES

from .base_cfg import BaseConfig

__all__ = ["RegNetConfig"]

_cfg = dict(
    test=dict(
        crop_pct=224 / 232,
    ),
    model=dict(
        name="RegNet",
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
        rand_aug=dict(
            magnitude=7,
        ),
        rand_erase=dict(
            prob=0.1,
            mode="pixel",
        ),
        mixup=dict(
            mixup_alpha=0.2,
            cutmix_alpha=1.0,
        ),
    ),
    solver=dict(
        basic_lr=0.025,
        weight_decay=(
            (0, "bias"),
            (0, NORM_TYPES),
            2e-5,
        ),
        nesterov=True,
        max_epoch=300,
        lr_schedule="cosine",
    ),
    model_ema=dict(
        enabled=True,
        momentum=0.99998,
    ),
)


class RegNetConfig(BaseConfig):
    def __init__(self, values_or_file=None, **kwargs):
        super().__init__(_cfg)
        self.merge(values_or_file, **kwargs)
