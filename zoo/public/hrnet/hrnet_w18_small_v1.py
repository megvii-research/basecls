#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
from basecls.configs import HRNetConfig

_cfg = dict(
    batch_size=64,
    model=dict(
        name="hrnet_w18_small_v1",
    ),
    solver=dict(
        basic_lr=0.05,
    ),
)


class Cfg(HRNetConfig):
    def __init__(self, values_or_file=None, **kwargs):
        super().__init__(_cfg)
        self.merge(values_or_file, **kwargs)
