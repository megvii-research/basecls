#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
from basecls.configs import ViTConfig

_cfg = dict(
    model=dict(
        name="vit_small_patch32_384",
    ),
    preprocess=dict(
        img_size=384,
    ),
    test=dict(
        img_size=384,
        crop_pct=1.0,
    ),
)


class Cfg(ViTConfig):
    def __init__(self, values_or_file=None, **kwargs):
        super().__init__(_cfg)
        self.merge(values_or_file, **kwargs)
