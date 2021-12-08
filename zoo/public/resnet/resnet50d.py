#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
from basecls.configs import ResNetConfig

_cfg = dict(
    batch_size=64,
    model=dict(
        name="resnet50d",
    ),
    solver=dict(
        basic_lr=0.05,
    ),
    model_ema=dict(
        enabled=True,
        momentum=0.99996,
    ),
)


class Cfg(ResNetConfig):
    def __init__(self, values_or_file=None, **kwargs):
        super().__init__(_cfg)
        self.merge(values_or_file, **kwargs)
