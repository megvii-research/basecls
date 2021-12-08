#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
from basecls.configs import SwinConfig

_cfg = dict(
    batch_size=128,
    model=dict(
        name="swin_small_patch4_window7_224",
    ),
    solver=dict(
        accumulation_steps=1,
    ),
)


class Cfg(SwinConfig):
    def __init__(self, values_or_file=None, **kwargs):
        super().__init__(_cfg)
        self.merge(values_or_file, **kwargs)
