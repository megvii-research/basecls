#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import megengine.data as data
import megengine.data.transform as T
import numpy as np
import pytest

from basecls.data.mixup import MixupCutmixCollator, MixupCutmixTransform


@pytest.mark.parametrize("mixup_alpha", [0.8])
@pytest.mark.parametrize("cutmix_alpha", [0.0, 1.0])
@pytest.mark.parametrize("cutmix_minmax", [None, (0.2, 0.8)])
@pytest.mark.parametrize("mode", ["batch", "pair", "elem"])
def test_mixup_transform(mixup_alpha, cutmix_alpha, cutmix_minmax, mode):
    B = 2
    K = 10

    transform = MixupCutmixTransform(
        mixup_alpha, cutmix_alpha, cutmix_minmax, mode=mode, num_classes=K
    )
    assert isinstance(transform, T.VisionTransform)

    x = tuple(np.random.randint(256, size=(8, 8, 3)).astype("uint8") for _ in range(B))
    y = tuple(np.random.randint(K) for _ in range(B))
    inputs = tuple(zip(x, y))
    outputs = transform.apply_batch(inputs)
    assert len(outputs) == len(inputs)
    assert len(outputs[0]) == len(inputs[0])
    assert outputs[0][0].shape == inputs[0][0].shape and outputs[0][0].dtype == inputs[0][0].dtype
    assert outputs[0][1].shape == (K,) and outputs[0][1].dtype == np.float32


def test_mixup_collator():
    B = 2
    K = 10

    collator = MixupCutmixCollator(num_classes=K)
    assert isinstance(collator, data.Collator)

    x = tuple(np.random.randint(256, size=(3, 8, 8)).astype("uint8") for _ in range(B))
    y = tuple(np.random.randint(K) for _ in range(B))
    x, y = collator.apply(tuple(zip(x, y)))
    assert x.shape == (B, 3, 8, 8) and x.dtype == np.uint8
    assert y.shape == (B, K) and y.dtype == np.float32
