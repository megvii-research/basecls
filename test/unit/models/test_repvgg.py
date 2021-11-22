#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import megengine as mge
import megengine.module as M
import numpy as np
import pytest

from basecls.models.repvgg import RepVGGBlock


@pytest.mark.parametrize("w_in", [32, 64])
@pytest.mark.parametrize("w_out", [64])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("groups", [1, 2, 4])
@pytest.mark.parametrize("se_r", [0.0, 0.25])
@pytest.mark.parametrize("act_name", ["relu"])
def test_block(w_in, w_out, stride, groups, se_r, act_name):
    m = RepVGGBlock(w_in, w_out, stride, groups, se_r, act_name, deploy=False)
    assert isinstance(m, M.Module)
    m.eval()

    x = mge.random.uniform(size=(2, w_in, 8, 8))
    y0 = m(x)

    m = RepVGGBlock.convert_to_deploy(m)
    y1 = m(x)

    np.testing.assert_allclose(y1.numpy(), y0.numpy(), rtol=1e-4, atol=1e-6)
