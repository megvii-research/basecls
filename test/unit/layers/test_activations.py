#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import megengine as mge
import megengine.module as M
import numpy as np
import pytest
import torch
import torch.nn as nn

from basecls.layers import ELU, HSigmoid, HSwish, ReLU6, Tanh, activation


@pytest.mark.parametrize(
    "name",
    [
        None,
        M.ReLU,
        "elu",
        "gelu",
        "hsigmoid",
        "hswish",
        "leaky_relu",
        "relu",
        "relu6",
        "prelu",
        "silu",
        "tanh",
    ],
)
def test_activation(name):
    m = activation(name)
    assert isinstance(m, M.Module)

    m(mge.random.normal(size=(2, 4, 8, 8)))


@pytest.mark.parametrize(
    "m,tm",
    [
        (ELU, nn.ELU),
        (HSigmoid, nn.Hardsigmoid),
        (HSwish, nn.Hardswish),
        (ReLU6, nn.ReLU6),
        (Tanh, nn.Tanh),
    ],
)
def test_act_modules(m, tm):
    x = np.random.rand(2, 4, 8, 8).astype("float32")

    my = m()(mge.Tensor(x)).numpy()
    ty = tm()(torch.tensor(x)).numpy()
    np.testing.assert_allclose(my, ty, rtol=1e-4, atol=1e-6)
