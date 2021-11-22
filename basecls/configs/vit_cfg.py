#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
from basecls.layers import NORM_TYPES

from .base_cfg import BaseConfig

__all__ = ["ViTConfig"]

_cfg = dict(
    batch_size=64,
    model=dict(
        name="ViT",
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
        basic_lr=6.25e-5,
        lr_min_factor=1e-2,
        weight_decay=(
            (0, "pos_embed"),
            (0, "cls_token"),
            (0, "bias"),
            (0, NORM_TYPES),
            0.05,
        ),
        max_epoch=300,
        warmup_epochs=20,
        warmup_factor=1e-3,
        lr_schedule="cosine",
    ),
    model_ema=dict(
        enabled=True,
        momentum=0.99996,
    ),
)


class ViTConfig(BaseConfig):
    def __init__(self, values_or_file=None, **kwargs):
        super().__init__(_cfg)
        self.merge(values_or_file, **kwargs)
