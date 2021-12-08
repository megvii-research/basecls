#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
from basecls.configs import ResNetConfig

_cfg = dict(
    batch_size=128,
    model=dict(
        name="se_resnet18",
    ),
    solver=dict(
        basic_lr=0.1,
    ),
)


class Cfg(ResNetConfig):
    def __init__(self, values_or_file=None, **kwargs):
        super().__init__(_cfg)
        self.merge(values_or_file, **kwargs)
