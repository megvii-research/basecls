#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
from basecls.configs import EffNetConfig

_cfg = dict(
    preprocess=dict(
        img_size=300,
    ),
    test=dict(
        img_size=384,
        crop_pct=1.0,
    ),
    model=dict(
        name="effnetv2_s",
    ),
)


class Cfg(EffNetConfig):
    def __init__(self, values_or_file=None, **kwargs):
        super().__init__(_cfg)
        self.merge(values_or_file, **kwargs)
