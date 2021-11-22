#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import megengine as mge
import megengine.module as M
import pytest

from basecls.models.hrnet import HRFusion, HRMerge, HRModule, HRStage, HRTrans, UpsampleNearest


@pytest.mark.parametrize("scale_factor", [1, 2])
def test_upsample_nearest(scale_factor):
    m = UpsampleNearest(scale_factor)
    assert isinstance(m, M.Module)

    m(mge.random.normal(size=(2, 3, 4, 4)))


@pytest.mark.parametrize(
    "channels, inp_shps",
    [([9], [(2, 9, 3, 3)]), ([3, 5, 7], [(2, 3, 16, 16), (2, 5, 8, 8), (2, 7, 4, 4)])],
)
@pytest.mark.parametrize("multi_scale_output", [True, False])
@pytest.mark.parametrize("norm_name", ["BN"])
@pytest.mark.parametrize("act_name", ["relu"])
def test_hrfusion(channels, multi_scale_output, norm_name, act_name, inp_shps):
    m = HRFusion(channels, multi_scale_output, norm_name, act_name)
    assert isinstance(m, M.Module)

    x_list = [mge.random.normal(size=sz) for sz in inp_shps]
    m(x_list)


@pytest.mark.parametrize("block_name", ["basic", "bottleneck"])
@pytest.mark.parametrize(
    "num_blocks, in_channels, channels, inp_shps",
    [
        ([1], [9], [8], [(2, 9, 3, 3)]),
        ([2, 3, 3], [3, 5, 7], [2, 4, 6], [(2, 3, 16, 16), (2, 5, 8, 8), (2, 7, 4, 4)]),
    ],
)
@pytest.mark.parametrize("multi_scale_output", [True, False])
@pytest.mark.parametrize("norm_name", ["BN"])
@pytest.mark.parametrize("act_name", ["relu"])
def test_hrmodule(
    block_name, num_blocks, in_channels, channels, multi_scale_output, norm_name, act_name, inp_shps
):
    m = HRModule(
        block_name, num_blocks, in_channels, channels, multi_scale_output, norm_name, act_name
    )
    assert isinstance(m, M.Module)

    x_list = [mge.random.normal(size=sz) for sz in inp_shps]
    m(x_list)


@pytest.mark.parametrize(
    "in_chs, inp_shps",
    [([9], [(2, 9, 3, 3)]), ([3, 5, 7], [(2, 3, 16, 16), (2, 5, 8, 8), (2, 7, 4, 4)])],
)
@pytest.mark.parametrize("out_chs", [[2, 4, 6], [7, 5, 3, 1], [1, 2, 3, 4, 5]])
@pytest.mark.parametrize("norm_name", ["BN"])
@pytest.mark.parametrize("act_name", ["relu"])
def test_hrtrans(in_chs, out_chs, norm_name, act_name, inp_shps):
    m = HRTrans(in_chs, out_chs, norm_name, act_name)
    assert isinstance(m, M.Module)

    x_list = [mge.random.normal(size=sz) for sz in inp_shps]
    m(x_list)


@pytest.mark.parametrize("num_modules", [1, 4])
@pytest.mark.parametrize("block_name", ["basic", "bottleneck"])
@pytest.mark.parametrize(
    "num_blocks, pre_channels, cur_channels, w_fst, inp_shps",
    [
        ([2], [], [8], 9, [(2, 9, 8, 8)]),
        ([2, 3, 3], [3, 5, 7], [2, 4, 6], None, [(2, 3, 16, 16), (2, 5, 8, 8), (2, 7, 4, 4)]),
        ([2, 3, 3, 3], [3, 5, 7], [2, 4, 6, 8], None, [(2, 3, 16, 16), (2, 5, 8, 8), (2, 7, 4, 4)]),
        (
            [2, 3, 3, 3, 5],
            [3, 5, 7],
            [2, 4, 6, 8, 10],
            None,
            [(2, 3, 16, 16), (2, 5, 8, 8), (2, 7, 4, 4)],
        ),
    ],
)
@pytest.mark.parametrize("multi_scale_output", [True, False])
@pytest.mark.parametrize("norm_name", ["BN"])
@pytest.mark.parametrize("act_name", ["relu"])
def test_hrstage(
    num_modules,
    num_blocks,
    block_name,
    pre_channels,
    cur_channels,
    multi_scale_output,
    w_fst,
    norm_name,
    act_name,
    inp_shps,
):
    m = HRStage(
        num_modules,
        num_blocks,
        block_name,
        pre_channels,
        cur_channels,
        multi_scale_output,
        w_fst,
        norm_name,
        act_name,
    )
    assert isinstance(m, M.Module)

    x_list = [mge.random.normal(size=sz) for sz in inp_shps]
    m(x_list)


@pytest.mark.parametrize("block_name", ["basic", "bottleneck"])
@pytest.mark.parametrize(
    "pre_channels, channels, inp_shps",
    [
        ([2], [3], [(2, 2, 4, 4)]),
        ([3, 5, 7], [2, 4, 6], [(2, 3, 8, 8), (2, 5, 4, 4), (2, 7, 2, 2)]),
    ],
)
@pytest.mark.parametrize("norm_name", ["BN"])
@pytest.mark.parametrize("act_name", ["relu"])
def test_hrmerge(block_name, pre_channels, channels, norm_name, act_name, inp_shps):
    m = HRMerge(block_name, pre_channels, channels, norm_name, act_name)
    assert isinstance(m, M.Module)

    x_list = [mge.random.normal(size=sz) for sz in inp_shps]
    m(x_list)
