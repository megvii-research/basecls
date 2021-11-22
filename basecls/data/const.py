#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import cv2
from PIL import Image

__all__ = ["CV2_INTERP", "PIL_INTERP"]

CV2_INTERP = {
    "bicubic": cv2.INTER_CUBIC,
    "bilinear": cv2.INTER_LINEAR,
    "lanczos": cv2.INTER_LANCZOS4,
    "nearest": cv2.INTER_NEAREST,
}

PIL_INTERP = {
    "bicubic": Image.BICUBIC,
    "bilinear": Image.BILINEAR,
    "lanczos": Image.LANCZOS,
    "nearest": Image.NEAREST,
}
