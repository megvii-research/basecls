#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import megengine as mge
import megengine.module as M
import pytest

from basecls.models.mbnet import MBConv, MBStage


@pytest.mark.parametrize("w_in", [32])
@pytest.mark.parametrize("w_out", [32, 64])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("kernel", [3, 5])
@pytest.mark.parametrize("exp_r", [1, 2.5, 3])
@pytest.mark.parametrize("se_r", [0.0, 0.25])
@pytest.mark.parametrize("se_from_exp", [True, False])
@pytest.mark.parametrize("se_act_name", ["relu"])
@pytest.mark.parametrize("se_approx", [True, False])
@pytest.mark.parametrize("se_rd_fn", [None, int, round])
@pytest.mark.parametrize("has_proj_act", [True, False])
@pytest.mark.parametrize("has_skip", [True, False])
@pytest.mark.parametrize("drop_path_prob", [0.0, 0.1])
@pytest.mark.parametrize("norm_name", ["BN"])
@pytest.mark.parametrize("act_name", ["relu"])
def test_block(
    w_in,
    w_out,
    stride,
    kernel,
    exp_r,
    se_r,
    se_from_exp,
    se_act_name,
    se_approx,
    se_rd_fn,
    has_proj_act,
    has_skip,
    drop_path_prob,
    norm_name,
    act_name,
):
    m = MBConv(
        w_in,
        w_out,
        stride,
        kernel,
        exp_r,
        se_r,
        se_from_exp,
        se_act_name,
        se_approx,
        se_rd_fn,
        has_proj_act,
        has_skip,
        drop_path_prob,
        norm_name,
        act_name,
    )
    assert isinstance(m, M.Module)

    m(mge.random.normal(size=(2, 32, 8, 8)))


@pytest.mark.parametrize("w_in", [4])
@pytest.mark.parametrize("w_out", [4, 8])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("depth", [2])
@pytest.mark.parametrize("exp_r", [1.0, [1.0, 3.0]])
@pytest.mark.parametrize("drop_path_prob", [[0.05, 0.1]])
def test_mb_stage(w_in, w_out, stride, depth, exp_r, drop_path_prob):
    m = MBStage(
        w_in,
        w_out,
        stride,
        depth,
        exp_r,
        drop_path_prob,
        kernel=3,
        se_r=0.0,
        se_from_exp=True,
        se_act_name="relu",
        se_approx=False,
        se_rd_fn=None,
        has_proj_act=False,
        has_skip=True,
        norm_name="BN",
        act_name="relu",
    )
    assert isinstance(m, M.Module)
    assert len(m) == depth

    m(mge.random.normal(size=(2, 4, 8, 8)))
