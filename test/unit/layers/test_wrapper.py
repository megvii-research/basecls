#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import megengine.module as M
import numpy as np
import pytest

from basecls.layers import Preprocess, make_divisible


def test_preprocess():
    m = Preprocess(mean=[0.0] * 3, std=[1.0] * 3)
    assert isinstance(m, M.Module)

    x = np.random.rand(2, 3, 8, 8).astype("float32")
    y = np.random.rand(2).astype("float32")
    mx, my = m((x, y))
    np.testing.assert_allclose(mx.numpy(), x, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(my.numpy(), y, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize(
    "value,divisor,min_value,round_limit,result",
    [
        (18, 8, None, 0.0, 16),
        (16, 8, None, 0.0, 16),
        (14, 8, None, 0.0, 16),
        (10, 8, None, 0.0, 8),
        (4, 8, None, 0.0, 8),
        (4, 8, 12, 0.0, 12),
        (14, 8, 12, 0.0, 16),
        (12, 8, None, 0.0, 16),
        (20, 8, None, 0.0, 24),
        (18, 8, None, 0.9, 24),
    ],
)
def test_make_divisible(value, divisor, min_value, round_limit, result):
    assert make_divisible(value, divisor, min_value, round_limit) == result
