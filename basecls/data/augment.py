#!/usr/bin/env python3
# Copyright (c) 2021 Facebook, Inc. and its affiliates.
# This file has been modified by Megvii ("Megvii Modifications").
# All Megvii Modifications are Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
"""AutoAugment and RandAugment

AutoAugment: `"AutoAugment: Learning Augmentation Policies from Data"
<https://arxiv.org/abs/1805.09501>`_

RandAugment: `"RandAugment: Practical automated data augmentation with a reduced search space"
<https://arxiv.org/abs/1909.13719>`_

"""
import random
from numbers import Real
from typing import List, Tuple, Union

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

__all__ = ["MAX_LEVEL", "POSTERIZE_MIN", "TorchAutoAugment", "TorchRandAugment", "WARP_PARAMS"]


# This signifies the max integer that the controller RNN could predict for the augmentation scheme.
MAX_LEVEL = 10

# Minimum value for posterize (0 in EfficientNet implementation).
POSTERIZE_MIN = 1

# Parameters for affine warping and rotation.
WARP_PARAMS = {"fillcolor": (128, 128, 128), "resample": Image.BILINEAR}


def affine_warp(img: Image, mat) -> Image:
    """Applies affine transform to image."""
    return img.transform(img.size, Image.AFFINE, mat, **WARP_PARAMS)


OP_FUNCTIONS = {
    # Each op takes an image x and a level v and returns an augmented image.
    "auto_contrast": lambda x, _: ImageOps.autocontrast(x),
    "equalize": lambda x, _: ImageOps.equalize(x),
    "invert": lambda x, _: ImageOps.invert(x),
    "rotate": lambda x, v: x.rotate(v, **WARP_PARAMS),
    "posterize": lambda x, v: ImageOps.posterize(x, max(POSTERIZE_MIN, int(v))),
    "posterize_inc": lambda x, v: ImageOps.posterize(x, max(POSTERIZE_MIN, 4 - int(v))),
    "solarize": lambda x, v: ImageOps.solarize(x, int(v)),
    # x.point(lambda i: i if i < int(v) else 255 - i),
    "solarize_inc": lambda x, v: ImageOps.solarize(x, 256 - int(v)),
    # x.point(lambda i: i if i < 256 - int(v) else 255 - i),
    "solarize_add": lambda x, v: x.point(lambda i: min(255, i + int(v)) if i < 128 else i),
    "color": lambda x, v: ImageEnhance.Color(x).enhance(v),
    "contrast": lambda x, v: ImageEnhance.Contrast(x).enhance(v),
    "brightness": lambda x, v: ImageEnhance.Brightness(x).enhance(v),
    "sharpness": lambda x, v: ImageEnhance.Sharpness(x).enhance(v),
    "color_inc": lambda x, v: ImageEnhance.Color(x).enhance(1 + v),
    "contrast_inc": lambda x, v: ImageEnhance.Contrast(x).enhance(1 + v),
    "brightness_inc": lambda x, v: ImageEnhance.Brightness(x).enhance(1 + v),
    "sharpness_inc": lambda x, v: ImageEnhance.Sharpness(x).enhance(1 + v),
    "shear_x": lambda x, v: affine_warp(x, (1, v, 0, 0, 1, 0)),
    "shear_y": lambda x, v: affine_warp(x, (1, 0, 0, v, 1, 0)),
    "trans_x": lambda x, v: affine_warp(x, (1, 0, v * x.size[0], 0, 1, 0)),
    "trans_y": lambda x, v: affine_warp(x, (1, 0, 0, 0, 1, v * x.size[1])),
}


OP_RANGES = {
    # Ranges for each op in the form of a (min, max, negate).
    "auto_contrast": (0, 1, False),
    "equalize": (0, 1, False),
    "invert": (0, 1, False),
    "rotate": (0.0, 30.0, True),
    "posterize": (0, 4, False),
    "posterize_inc": (0, 4, False),
    "solarize": (0, 256, False),
    "solarize_inc": (0, 256, False),
    "solarize_add": (0, 110, False),
    "color": (0.1, 1.9, False),
    "color_inc": (0, 0.9, True),
    "contrast": (0.1, 1.9, False),
    "contrast_inc": (0, 0.9, True),
    "brightness": (0.1, 1.9, False),
    "brightness_inc": (0, 0.9, True),
    "sharpness": (0.1, 1.9, False),
    "sharpness_inc": (0, 0.9, True),
    "shear_x": (0.0, 0.3, True),
    "shear_y": (0.0, 0.3, True),
    "trans_x": (0.0, 0.45, True),
    "trans_y": (0.0, 0.45, True),
}


AUTOAUG_POLICY = [
    # AutoAugment "policy_v0" in form of (op, prob, magnitude).
    [("equalize", 0.8, 1), ("shear_y", 0.8, 4)],
    [("color", 0.4, 9), ("equalize", 0.6, 3)],
    [("color", 0.4, 1), ("rotate", 0.6, 8)],
    [("solarize", 0.8, 3), ("equalize", 0.4, 7)],
    [("solarize", 0.4, 2), ("solarize", 0.6, 2)],
    [("color", 0.2, 0), ("equalize", 0.8, 8)],
    [("equalize", 0.4, 8), ("solarize_add", 0.8, 3)],
    [("shear_x", 0.2, 9), ("rotate", 0.6, 8)],
    [("color", 0.6, 1), ("equalize", 1.0, 2)],
    [("invert", 0.4, 9), ("rotate", 0.6, 0)],
    [("equalize", 1.0, 9), ("shear_y", 0.6, 3)],
    [("color", 0.4, 7), ("equalize", 0.6, 0)],
    [("posterize", 0.4, 6), ("auto_contrast", 0.4, 7)],
    [("solarize", 0.6, 8), ("color", 0.6, 9)],
    [("solarize", 0.2, 4), ("rotate", 0.8, 9)],
    [("rotate", 1.0, 7), ("trans_y", 0.8, 9)],
    [("shear_x", 0.0, 0), ("solarize", 0.8, 4)],
    [("shear_y", 0.8, 0), ("color", 0.6, 4)],
    [("color", 1.0, 0), ("rotate", 0.6, 2)],
    [("equalize", 0.8, 4), ("equalize", 0.0, 8)],
    [("equalize", 1.0, 4), ("auto_contrast", 0.6, 2)],
    [("shear_y", 0.4, 7), ("solarize_add", 0.6, 7)],
    [("posterize", 0.8, 2), ("solarize", 0.6, 10)],
    [("solarize", 0.6, 8), ("equalize", 0.6, 1)],
    [("color", 0.8, 6), ("rotate", 0.4, 5)],
]


RANDAUG_OPS = [
    # RandAugment list of operations using "increasing" transforms.
    "auto_contrast",
    "equalize",
    "invert",
    "rotate",
    "posterize_inc",
    "solarize_inc",
    "solarize_add",
    "color_inc",
    "contrast_inc",
    "brightness_inc",
    "sharpness_inc",
    "shear_x",
    "shear_y",
    "trans_x",
    "trans_y",
]


def apply_op(
    img: Image,
    op: str,
    prob: Union[float, Tuple[float, float]],
    magnitude: Real,
    magnitude_std: float = 0.0,
) -> Image:
    """Apply the selected op to image with given probability and magnitude."""
    if op not in OP_RANGES and op not in OP_FUNCTIONS:
        raise ValueError(f"Operation '{op}' not supported")
    if isinstance(prob, tuple):
        assert len(prob) == 2
        prob = random.uniform(**prob)

    if random.random() > prob:
        return img

    if magnitude_std == float("inf"):
        magnitude = random.uniform(0, magnitude)
    elif magnitude_std > 0.0:
        magnitude = max(0, random.gauss(magnitude, magnitude_std))

    min_v, max_v, negate = OP_RANGES[op]
    # The magnitude is converted to an absolute value v for an op (some ops use -v or v)
    v = magnitude / MAX_LEVEL * (max_v - min_v) + min_v
    v = -v if negate and random.random() > 0.5 else v
    return OP_FUNCTIONS[op](img, v)


def auto_augment(img: Image, policy: List[Tuple] = None) -> Image:
    """Apply auto augmentation to an image."""
    policy = policy if policy else AUTOAUG_POLICY
    for op, prob, magnitude in random.choice(policy):
        img = apply_op(img, op, prob, magnitude)
    return img


def rand_augment(
    img: Image,
    magnitude: Real,
    magnitude_std: float = 0.0,
    prob: Union[float, Tuple[float, float]] = 0.5,
    n_ops: int = 2,
    ops: List[str] = None,
) -> Image:
    """Apply random augmentation to an image."""
    ops = ops if ops else RANDAUG_OPS
    for op in np.random.choice(ops, n_ops):
        img = apply_op(img, op, prob, magnitude, magnitude_std)
    return img


class TorchAutoAugment:
    def __call__(self, img: Image) -> Image:
        return auto_augment(img)


class TorchRandAugment:
    def __init__(
        self,
        magnitude: Real,
        magnitude_std: float = 0.0,
        prob: Union[float, Tuple[float, float]] = 0.5,
        n_ops: int = 2,
    ):
        self.magnitude = magnitude
        self.magnitude_std = magnitude_std
        self.prob = prob
        self.n_ops = n_ops

    def __call__(self, img: Image) -> Image:
        return rand_augment(img, self.magnitude, self.magnitude_std, self.prob, self.n_ops)
