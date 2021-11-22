#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
from numbers import Real
from typing import Tuple, Union

import numpy as np


class FakeDataLoader:
    """FakeDataLoader

    Args:
        batch_size: batch size
        img_size: height and width. Default: 224
        channels: color channels. Default: 3
        length: loader length. Default: 100
        num_classes: number of classes. Default: 1000
    """

    def __init__(
        self,
        batch_size: int,
        img_size: Union[int, Tuple[int, int]] = 224,
        channels: int = 3,
        length: int = 100,
        num_classes: int = 1000,
    ):
        self.batch_size = batch_size
        self.channels = channels
        if isinstance(img_size, Real):
            img_size = (img_size, img_size)
        self.img_size = img_size
        self.length = length
        self.num_classes = num_classes

    def __len__(self):
        return self.length

    def __iter__(self):

        images = np.random.randint(
            256, dtype=np.uint8, size=(self.batch_size, self.channels, *self.img_size)
        )
        labels = np.random.randint(self.num_classes, dtype=np.int32, size=(self.batch_size,))

        for _ in range(len(self)):
            yield images, labels
