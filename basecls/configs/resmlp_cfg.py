#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
from basecls.layers import NORM_TYPES
from basecls.models.resmlp import Affine

from .base_cfg import BaseConfig

__all__ = ["ResMLPConfig"]

_cfg = dict(
    batch_size=64,
    model=dict(
        name="ResMLP",
    ),
    test=dict(
        crop_pct=0.9,
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
        mixup=dict(
            mixup_alpha=0.8,
            cutmix_alpha=1.0,
        ),
    ),
    solver=dict(
        optimizer="lamb",
        basic_lr=6.25e-4,
        lr_min_factor=1e-2,
        weight_decay=(
            (0, "bias"),
            (0, "gamma"),
            (0, NORM_TYPES),
            (0, Affine),
            0.2,
        ),
        max_epoch=400,
        warmup_factor=1e-3,
        lr_schedule="cosine",
    ),
    model_ema=dict(
        enabled=True,
        momentum=0.99996,
    ),
)


class ResMLPConfig(BaseConfig):
    def __init__(self, values_or_file=None, **kwargs):
        super().__init__(_cfg)
        self.merge(values_or_file, **kwargs)
