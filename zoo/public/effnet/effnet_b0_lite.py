#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
from basecls.configs import EffNetLiteConfig

_cfg = dict(
    batch_size=64,
    model=dict(
        name="effnet_b0_lite",
    ),
    solver=dict(
        basic_lr=0.1,
    ),
    model_ema=dict(
        enabled=True,
        momentum=0.99996,
    ),
)


class Cfg(EffNetLiteConfig):
    def __init__(self, values_or_file=None, **kwargs):
        super().__init__(_cfg)
        self.merge(values_or_file, **kwargs)
