#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
from .base_cfg import BaseConfig

__all__ = ["RepVGGConfig"]

_cfg = dict(
    batch_size=64,
    model=dict(
        name="RepVGG",
    ),
    bn=dict(
        precise_every_n_epoch=1,
    ),
    augments=dict(
        name="SimpleAugment",
    ),
    solver=dict(
        max_epoch=120,
        lr_schedule="cosine",
        basic_lr=0.025,
        weight_decay=(
            (0, "bn"),  # NOTE: not NORM_TYPES since we require decay on rbr_identity
            (0, "bias"),
            1e-4,
        ),
    ),
)


class RepVGGConfig(BaseConfig):
    def __init__(self, values_or_file=None, **kwargs):
        super().__init__(_cfg)
        self.merge(values_or_file, **kwargs)
