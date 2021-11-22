#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import megengine.data.transform as T
import numpy as np
import pytest

from basecls.data.rand_erase import RandomErasing


@pytest.mark.parametrize("prob", [0.25])
@pytest.mark.parametrize("ratio", [0.4, (0.4, 1.5)])
@pytest.mark.parametrize("mode", ["const", "rand", "pixel"])
@pytest.mark.parametrize("count", [1, (2, 3)])
@pytest.mark.parametrize("pad_mean", [0.0, (0.0, 0.0, 0.0)])
@pytest.mark.parametrize("pad_std", [1.0, (1.0, 1.0, 1.0)])
def test_rand_erase(prob, ratio, mode, count, pad_mean, pad_std):
    B = 2

    transform = RandomErasing(
        prob, ratio=ratio, mode=mode, count=count, pad_mean=pad_mean, pad_std=pad_std
    )
    assert isinstance(transform, T.VisionTransform)

    x = tuple(np.random.randint(256, size=(8, 8, 3)).astype("uint8") for _ in range(B))
    y = tuple(np.random.randint(10) for _ in range(B))
    inputs = tuple(zip(x, y))
    outputs = transform.apply_batch(inputs)
    assert len(outputs) == len(inputs)
    assert len(outputs[0]) == len(inputs[0])
    assert outputs[0][0].shape == inputs[0][0].shape and outputs[0][0].dtype == inputs[0][0].dtype
