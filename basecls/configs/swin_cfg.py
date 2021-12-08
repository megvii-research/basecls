#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
from basecls.layers import NORM_TYPES

from .base_cfg import BaseConfig

__all__ = ["SwinConfig"]

_cfg = dict(
    batch_size=64,
    model=dict(
        name="Swin",
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
        optimizer="adamw",
        basic_lr=1.25e-4,
        lr_min_factor=1e-2,
        weight_decay=(
            (0, "abs_pos_embed"),
            (0, "rel_pos_bias_table"),
            (0, "bias"),
            (0, NORM_TYPES),
            0.05,
        ),
        max_epoch=300,
        warmup_epochs=20,
        warmup_factor=1e-3,
        lr_schedule="cosine",
        grad_clip=dict(
            name="norm",
            max_norm=5.0,
        ),
        accumulation_steps=2,
    ),
)


class SwinConfig(BaseConfig):
    def __init__(self, values_or_file=None, **kwargs):
        super().__init__(_cfg)
        self.merge(values_or_file, **kwargs)
