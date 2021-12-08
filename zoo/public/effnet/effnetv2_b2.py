#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
from basecls.configs import EffNetConfig

_cfg = dict(
    batch_size=64,
    preprocess=dict(
        img_size=208,
    ),
    test=dict(
        img_size=260,
        crop_pct=260 / 292,
    ),
    model=dict(
        name="effnetv2_b2",
    ),
    solver=dict(
        basic_lr=0.1,
    ),
    model_ema=dict(
        enabled=True,
        momentum=0.99996,
    ),
)


class Cfg(EffNetConfig):
    def __init__(self, values_or_file=None, **kwargs):
        super().__init__(_cfg)
        self.merge(values_or_file, **kwargs)
