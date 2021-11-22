#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import megengine as mge
import megengine.module as M
import numpy as np
import pytest
import torch
import torch.nn as nn

from basecls.configs import BaseConfig
from basecls.layers import BinaryCrossEntropy, CrossEntropy, build_loss


@pytest.mark.parametrize("name", [CrossEntropy, "BinaryCrossEntropy", "CrossEntropy"])
def test_build_loss(name):
    cfg = BaseConfig(loss=dict(name=name))

    m = build_loss(cfg)
    assert isinstance(m, M.Module)


def test_bce():
    x = np.random.rand(2, 8, 4).astype("float32")
    y = np.random.rand(2, 8, 4).astype("float32")

    ml = BinaryCrossEntropy()(mge.Tensor(x), mge.Tensor(y)).numpy()
    tl = nn.BCEWithLogitsLoss()(torch.tensor(x), torch.tensor(y)).numpy()
    np.testing.assert_allclose(ml, tl, rtol=1e-4, atol=1e-6)


def test_ce():
    K = 4

    x = np.random.rand(2, 8, K).astype("float32")
    y = np.random.randint(K, size=(2, 8)).astype("int32")
    oy = np.eye(K, dtype="int32")[y]

    ml = CrossEntropy(axis=2)(mge.Tensor(x), mge.Tensor(y)).numpy()
    tl = nn.CrossEntropyLoss()(
        torch.tensor(x).reshape(-1, K), torch.tensor(y).flatten().long()
    ).numpy()
    np.testing.assert_allclose(ml, tl, rtol=1e-4, atol=1e-6)

    # one hot
    ol = CrossEntropy(axis=2)(mge.Tensor(x), mge.Tensor(oy)).numpy()
    np.testing.assert_allclose(ml, ol, rtol=1e-4, atol=1e-6)

    # label smoothing
    ml = CrossEntropy(axis=2, label_smooth=0.1)(mge.Tensor(x), mge.Tensor(y)).numpy()
    ol = CrossEntropy(axis=2, label_smooth=0.1)(mge.Tensor(x), mge.Tensor(oy)).numpy()
    np.testing.assert_allclose(ml, ol, rtol=1e-4, atol=1e-6)
