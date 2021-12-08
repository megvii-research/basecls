#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
from basecls.configs import EffNetConfig

_cfg = dict(
    preprocess=dict(
        img_size=380,
    ),
    test=dict(
        img_size=380,
        crop_pct=380 / 412,
    ),
    model=dict(
        name="effnet_b4",
    ),
    augments=dict(
        rand_aug=dict(
            magnitude=13,
        ),
    ),
)


class Cfg(EffNetConfig):
    def __init__(self, values_or_file=None, **kwargs):
        super().__init__(_cfg)
        self.merge(values_or_file, **kwargs)
