#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
from basecls.configs import VGGConfig

_cfg = dict(
    model=dict(
        name="vgg16_bn",
    ),
)


class Cfg(VGGConfig):
    def __init__(self, values_or_file=None, **kwargs):
        super().__init__(_cfg)
        self.merge(values_or_file, **kwargs)
