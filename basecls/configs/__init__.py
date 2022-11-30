#!/usr/bin/env python3
# Copyright (c) Megvii Inc. All rights reserved.

from .base_cfg import BaseConfig
from .effnet_cfg import EffNetConfig, EffNetLiteConfig
from .hrnet_cfg import HRNetConfig
from .mbnet_cfg import MBConfig
from .regnet_cfg import RegNetConfig
from .repvgg_cfg import RepVGGConfig
from .resmlp_cfg import ResMLPConfig
from .resnet_cfg import ResNetConfig
from .snet_cfg import SNetConfig
from .swin_cfg import SwinConfig
from .vgg_cfg import VGGConfig
from .vit_cfg import ViTConfig

_EXCLUDE = {}
__all__ = [k for k in globals().keys() if k not in _EXCLUDE and not k.startswith("_")]
