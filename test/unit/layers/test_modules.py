#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import megengine as mge
import megengine.module as M
import numpy as np
import pytest

from basecls.layers import SE, DropPath, conv2d, gap2d, linear, norm2d, pool2d


@pytest.mark.parametrize("w_in", [4])
@pytest.mark.parametrize("w_out", [8])
@pytest.mark.parametrize("k", [3, 5])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("dilation", [1, 2])
@pytest.mark.parametrize("groups", [1, 2])
@pytest.mark.parametrize("bias", [True, False])
def test_conv2d(w_in, w_out, k, stride, dilation, groups, bias):
    m = conv2d(w_in, w_out, k, stride=stride, dilation=dilation, groups=groups, bias=bias)
    assert isinstance(m, M.Conv2d)

    m(mge.random.normal(size=(2, 4, 8, 8)))


@pytest.mark.parametrize("drop_prob", [0.0, 0.5])
def test_drop_path(drop_prob):
    m = DropPath(drop_prob)
    assert isinstance(m, M.Module)

    m.training = True
    m(mge.random.normal(size=(2, 4, 8, 8)))

    m.training = False
    x = np.random.rand(2, 4, 8, 8).astype("float32")
    y = m(mge.Tensor(x)).numpy()
    np.testing.assert_allclose(y, x, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("shape", [1, (7, 7)])
def test_gap2d(shape):
    m = gap2d(shape)
    assert isinstance(m, M.AdaptiveAvgPool2d)

    m(mge.random.normal(size=(2, 4, 8, 8)))


@pytest.mark.parametrize("w_in", [4])
@pytest.mark.parametrize("w_out", [8])
@pytest.mark.parametrize("bias", [True, False])
def test_linear(w_in, w_out, bias):
    m = linear(w_in, w_out, bias=bias)
    assert isinstance(m, M.Linear)

    m(mge.random.normal(size=(2, 8, 4)))


# TODO: "GN", "IN" and "LN" need different hyper-parameters
@pytest.mark.parametrize("name", [None, "BN", "SyncBN"])
@pytest.mark.parametrize("w_in", [4])
def test_norm2d(name, w_in):
    m = norm2d(name, w_in)
    assert isinstance(m, M.Module)

    m(mge.random.normal(size=(2, 4, 8, 8)))


@pytest.mark.parametrize("k", [3, 5])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("name", ["avg", "max"])
def test_pool2d(k, stride, name):
    m = pool2d(k, stride=stride, name=name)
    assert isinstance(m, M.Module)

    m(mge.random.normal(size=(2, 4, 8, 8)))


@pytest.mark.parametrize("w_in", [8])
@pytest.mark.parametrize("w_se", [4])
@pytest.mark.parametrize("act_name", ["relu", "silu"])
def test_se(w_in, w_se, act_name):
    m = SE(w_in, w_se, act_name)
    assert isinstance(m, M.Module)

    m(mge.random.normal(size=(2, 8, 8, 8)))
