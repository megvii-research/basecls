#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
from basecls.configs import RegNetConfig

_cfg = dict(
    batch_size=128,
    model=dict(
        name="regnety_002",
    ),
    solver=dict(
        basic_lr=0.1,
    ),
    model_ema=dict(
        enabled=True,
        momentum=0.99992,
    ),
)


class Cfg(RegNetConfig):
    def __init__(self, values_or_file=None, **kwargs):
        super().__init__(_cfg)
        self.merge(values_or_file, **kwargs)
