#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
from basecls.configs import SwinConfig

_cfg = dict(
    model=dict(
        name="swin_large_patch4_window7_224",
    ),
)


class Cfg(SwinConfig):
    def __init__(self, values_or_file=None, **kwargs):
        super().__init__(_cfg)
        self.merge(values_or_file, **kwargs)
