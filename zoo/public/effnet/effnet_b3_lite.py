#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
from basecls.configs import EffNetLiteConfig

_cfg = dict(
    preprocess=dict(
        img_size=280,
    ),
    test=dict(
        img_size=280,
        crop_pct=280 / 312,
    ),
    model=dict(
        name="effnet_b3_lite",
    ),
    augments=dict(
        rand_aug=dict(
            magnitude=11,
        ),
    ),
)


class Cfg(EffNetLiteConfig):
    def __init__(self, values_or_file=None, **kwargs):
        super().__init__(_cfg)
        self.merge(values_or_file, **kwargs)
