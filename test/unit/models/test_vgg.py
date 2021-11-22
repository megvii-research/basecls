#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import megengine as mge
import megengine.module as M
import pytest

from basecls.models.vgg import VGGStage


@pytest.mark.parametrize("w_in", [4])
@pytest.mark.parametrize("w_out", [8])
@pytest.mark.parametrize("depth", [2])
@pytest.mark.parametrize("norm_name", [None, "BN"])
@pytest.mark.parametrize("act_name", ["relu"])
def test_vgg_stage(w_in, w_out, depth, norm_name, act_name):
    m = VGGStage(w_in, w_out, depth, norm_name, act_name)
    assert isinstance(m, M.Module)
    assert len(m) == depth

    m(mge.random.normal(size=(2, 4, 8, 8)))
