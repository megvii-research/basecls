#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import megengine.data as data
import megengine.data.transform as T
import numpy as np
import pytest

from basecls.configs import BaseConfig
from basecls.data import (
    AutoAugment,
    ColorAugment,
    RandAugment,
    SimpleAugment,
    build_mixup,
    build_transform,
)


@pytest.mark.parametrize("Augment", [AutoAugment, ColorAugment, RandAugment, SimpleAugment])
def test_augment(Augment):
    B = 2
    K = 10

    cfg = BaseConfig()
    transform = Augment.build(cfg)
    assert isinstance(transform, T.Transform)

    x = tuple(np.random.randint(256, size=(8, 8, 3)).astype("uint8") for _ in range(B))
    y = tuple(np.random.randint(K) for _ in range(B))
    inputs = tuple(zip(x, y))
    outputs = transform.apply_batch(inputs)
    assert len(outputs) == len(inputs)
    assert len(outputs[0]) == len(inputs[0])
    assert outputs[0][0].shape == inputs[0][0].shape and outputs[0][0].dtype == inputs[0][0].dtype


@pytest.mark.parametrize("train", [True, False])
def test_build_mixup(train):
    cfg = BaseConfig()
    collator = build_mixup(cfg, train)
    assert collator is None or isinstance(collator, data.Collator)


@pytest.mark.parametrize("train", [True, False])
def test_build_transform(train):
    B = 2
    TRAIN_SIZE = 7
    TEST_SIZE = 14

    cfg = BaseConfig(preprocess=dict(img_size=TRAIN_SIZE), test=dict(img_size=TEST_SIZE))
    augments = ColorAugment.build(cfg) if train else None
    transform = build_transform(cfg, train, augments)
    assert isinstance(transform, T.VisionTransform)

    x = tuple(np.random.randint(256, size=(12, 12, 3)).astype("uint8") for _ in range(B))
    y = tuple(np.random.randint(10) for _ in range(B))
    inputs = tuple(zip(x, y))
    outputs = transform.apply_batch(inputs)
    assert len(outputs) == len(inputs)
    assert len(outputs[0]) == len(inputs[0])
    S = TRAIN_SIZE if train else TEST_SIZE
    assert outputs[0][0].shape == (3, S, S)
    assert outputs[0][0].dtype == inputs[0][0].dtype
