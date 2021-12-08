#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
from basecls.configs import EffNetConfig

_cfg = dict(
    preprocess=dict(
        img_size=384,
    ),
    test=dict(
        img_size=480,
        crop_pct=1.0,
    ),
    model=dict(
        name="effnetv2_l",
    ),
    augments=dict(
        rand_aug=dict(
            magnitude=21,
        ),
        mixup=dict(
            mixup_alpha=0.5,
            cutmix_alpha=0.5,
        ),
    ),
)


class Cfg(EffNetConfig):
    def __init__(self, values_or_file=None, **kwargs):
        super().__init__(_cfg)
        self.merge(values_or_file, **kwargs)
