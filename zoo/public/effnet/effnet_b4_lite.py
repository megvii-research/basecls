#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
from basecls.configs import EffNetLiteConfig

_cfg = dict(
    preprocess=dict(
        img_size=300,
    ),
    test=dict(
        img_size=300,
        crop_pct=300 / 332,
    ),
    model=dict(
        name="effnet_b4_lite",
    ),
    augments=dict(
        rand_aug=dict(
            magnitude=13,
        ),
    ),
)


class Cfg(EffNetLiteConfig):
    def __init__(self, values_or_file=None, **kwargs):
        super().__init__(_cfg)
        self.merge(values_or_file, **kwargs)
