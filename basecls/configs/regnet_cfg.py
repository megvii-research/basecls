#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
from .base_cfg import BaseConfig

__all__ = ["RegNetConfig"]

_cfg = dict(
    model=dict(
        name="RegNet",
    ),
    bn=dict(
        precise_every_n_epoch=1,
    ),
    solver=dict(
        basic_lr=0.025,
        weight_decay=5e-5,
        nesterov=True,
        lr_schedule="cosine",
    ),
)


class RegNetConfig(BaseConfig):
    def __init__(self, values_or_file=None, **kwargs):
        super().__init__(_cfg)
        self.merge(values_or_file, **kwargs)
