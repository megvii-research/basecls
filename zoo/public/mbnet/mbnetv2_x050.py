#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
from basecls.configs import MBConfig

_cfg = dict(
    model=dict(
        name="mbnetv2_x050",
    ),
)


class Cfg(MBConfig):
    def __init__(self, values_or_file=None, **kwargs):
        super().__init__(_cfg)
        self.merge(values_or_file, **kwargs)
