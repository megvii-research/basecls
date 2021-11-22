#!/usr/bin/env python3
# Copyright (c) 2020 Ross Wightman
# This file has been modified by Megvii ("Megvii Modifications").
# All Megvii Modifications are Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
"""Random Erasing (Cutout)

Random Erasing: `"Random Erasing Data Augmentation" <https://arxiv.org/abs/1708.04896>`_

"""
import math
import random
from numbers import Real
from typing import Sequence, Tuple, Union

import megengine.data.transform as T
import numpy as np

__all__ = ["RandomErasing"]


class RandomErasing(T.VisionTransform):
    """Randomly selects a rectangle region in an image and erases its pixels.

    This variant of RandomErasing is intended to be applied to either a batch
    or single image tensor after it has been normalized by dataset mean and std.

    Args:
        prob: probability that Random Erasing operation will be performed. Default: ``0.5``
        scale_range: percentage of erased area wrt input image area. Default: ``(0.02, 1.0 / 3)``
        ratio: aspect ratio of erased area. if a scalar, range will be (ratio, 1.0 / ratio).
            Default: ``0.3``
        mode: pixel color mode, one of "const", "rand", or "pixel". Default: ``"const"``
            ``"const"`` - erase block is constant color of 0 for all channels
            ``"rand"``  - erase block is same per-channel random (normal) color
            ``"pixel"`` - erase block is per-pixel random (normal) color
        count: maximum number or range of erasing blocks per image, area per box is scaled by count.
            if a scalar, per-image count is randomly chosen between 1 and this value. if a range,
            per-image count is randomly chosen between this range. Default: ``1``
        num_splits: augmentation splits. if > 1, the first split will not be erased. Default: ``0``
        pad_mean: the mean of padding pixels. Default: ``0.0``
        pad_std: the std of padding pixels. Default: ``1.0``
    """

    def __init__(
        self,
        prob: float = 0.5,
        scale_range: Tuple[float, float] = (0.02, 1 / 3),
        ratio: Union[float, Tuple[float, float]] = 0.3,
        mode: str = "const",
        count: Union[int, Tuple[int, int]] = 1,
        num_splits: int = 0,
        pad_mean: Union[float, Tuple[float, float, float]] = 0.0,
        pad_std: Union[float, Tuple[float, float, float]] = 1.0,
        *,
        order: Sequence = None,
    ):
        super().__init__(order)
        self.prob = prob
        self.scale_range = scale_range
        if isinstance(ratio, Real):
            ratio = (ratio, 1 / ratio)
        self.log_ratio_range = tuple(math.log(x) for x in ratio)
        if mode not in ("const", "rand", "pixel"):
            raise ValueError(f"RandomErasing mode '{mode}' not supported")
        self.mode = mode
        if isinstance(count, int):
            count = (1, count)
        self.count = count
        self.num_splits = num_splits
        self.pad_mean = np.array(pad_mean, dtype=np.float32)
        self.pad_std = np.array(pad_std, dtype=np.float32)

    def apply_batch(self, inputs: Sequence[Tuple]):
        batch_start = len(inputs) // self.num_splits if self.num_splits > 1 else 0
        return tuple([x if i < batch_start else self.apply(x) for i, x in enumerate(inputs)])

    def _apply_image(self, image: np.ndarray) -> np.ndarray:
        if random.random() > self.prob:
            return image

        dtype = image.dtype
        image = image.astype(np.float32)

        height, width, c = image.shape
        area = height * width
        count = random.randint(*self.count)

        for _ in range(count):
            for attempt in range(10):
                target_area = random.uniform(*self.scale_range) * area / count
                aspect_ratio = math.exp(random.uniform(*self.log_ratio_range))

                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))

                if w < width and h < height:
                    y = random.randint(0, height - h)
                    x = random.randint(0, width - w)
                    image[y : y + h, x : x + w] = self._get_pixels((h, w, c))
                    break

        return image.clip(0, 255).astype(dtype)

    def _get_pixels(self, patch_size: Tuple[int, int, int]):
        if self.mode == "const":
            pad = np.zeros((1, 1, patch_size[-1]), dtype=np.float32)
        elif self.mode == "rand":
            pad = np.random.normal(size=(1, 1, patch_size[-1])).astype(np.float32)
        elif self.mode == "pixel":
            pad = np.random.normal(size=patch_size).astype(np.float32)
        else:
            raise ValueError(f"Mode '{self.mode}' not supported")
        return self.pad_mean + self.pad_std * pad
