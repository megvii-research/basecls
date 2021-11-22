#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import megengine as mge
import megengine.module as M
import pytest

from basecls.models.resmlp import Affine, ResMLPBlock


@pytest.mark.parametrize("dim", [8])
def test_affine(dim):
    m = Affine(dim)
    assert isinstance(m, M.Module)

    m(mge.random.normal(size=(2, 3, 8)))


@pytest.mark.parametrize("dim", [8])
@pytest.mark.parametrize("drop", [0, 0.05])
@pytest.mark.parametrize("drop_path", [0, 0.1])
@pytest.mark.parametrize("num_patches", [16])
@pytest.mark.parametrize("init_scale", [0, 0.001])
@pytest.mark.parametrize("ffn_ratio", [0.5, 2.0])
@pytest.mark.parametrize("act_name", ["relu", "gelu"])
def test_resmlpblock(dim, drop, drop_path, num_patches, init_scale, ffn_ratio, act_name):
    m = ResMLPBlock(
        dim=dim,
        drop=drop,
        drop_path=drop_path,
        num_patches=num_patches,
        init_scale=init_scale,
        ffn_ratio=ffn_ratio,
        act_name=act_name,
    )
    assert isinstance(m, M.Module)

    m(mge.random.normal(size=(2, 16, 8)))
