#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
from basecls.configs import ViTConfig

_cfg = dict(
    batch_size=128,
    model=dict(
        name="vit_tiny_patch16_224",
    ),
    solver=dict(
        basic_lr=1.25e-4,
    ),
    model_ema=dict(
        momentum=0.99992,
    ),
)


class Cfg(ViTConfig):
    def __init__(self, values_or_file=None, **kwargs):
        super().__init__(_cfg)
        self.merge(values_or_file, **kwargs)
