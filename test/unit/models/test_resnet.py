#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import megengine as mge
import megengine.module as M
import pytest

from basecls.models.regnet import RegBottleneckBlock
from basecls.models.resnet import (
    AnyStage,
    ResBasicBlock,
    ResBottleneckBlock,
    ResDeepStem,
    ResStem,
    SimpleStem,
)


@pytest.mark.parametrize("Block", [RegBottleneckBlock, ResBasicBlock, ResBottleneckBlock])
@pytest.mark.parametrize("w_in", [32])
@pytest.mark.parametrize("w_out", [32, 64])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("bot_mul", [1.0, 0.25])
@pytest.mark.parametrize("group_w", [8])
@pytest.mark.parametrize("se_r", [0.0, 0.25])
@pytest.mark.parametrize("avg_down", [True, False])
@pytest.mark.parametrize("norm_name", ["BN"])
@pytest.mark.parametrize("act_name", ["relu"])
def test_block(Block, w_in, w_out, stride, bot_mul, group_w, se_r, avg_down, norm_name, act_name):
    m = Block(
        w_in,
        w_out,
        stride,
        bot_mul=bot_mul,
        group_w=group_w,
        se_r=se_r,
        avg_down=avg_down,
        norm_name=norm_name,
        act_name=act_name,
    )
    assert isinstance(m, M.Module)

    m(mge.random.normal(size=(2, 32, 8, 8)))


@pytest.mark.parametrize("Stem", [ResDeepStem, ResStem, SimpleStem])
@pytest.mark.parametrize("w_in", [3])
@pytest.mark.parametrize("w_out", [8, 16])
@pytest.mark.parametrize("norm_name", ["BN"])
@pytest.mark.parametrize("act_name", ["relu"])
def test_stem(Stem, w_in, w_out, norm_name, act_name):
    m = Stem(w_in, w_out, norm_name=norm_name, act_name=act_name)
    assert isinstance(m, M.Module)

    m(mge.random.normal(size=(2, 3, 8, 8)))


@pytest.mark.parametrize("w_in", [4])
@pytest.mark.parametrize("w_out", [4, 8])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("depth", [2])
@pytest.mark.parametrize("block_func", [RegBottleneckBlock, ResBasicBlock, ResBottleneckBlock])
def test_any_stage(w_in, w_out, stride, depth, block_func):
    m = AnyStage(
        w_in,
        w_out,
        stride,
        depth,
        block_func,
        bot_mul=1.0,
        group_w=4,
        se_r=0.0,
        avg_down=False,
        norm_name="BN",
        act_name="relu",
    )
    assert isinstance(m, M.Module)
    assert len(m) == depth

    m(mge.random.normal(size=(2, 4, 8, 8)))
