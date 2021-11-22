#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import megengine as mge
import megengine.module as M
import numpy as np
import pytest

from basecls.layers import ClsHead, MBV3Head, VGGHead, build_head


@pytest.mark.parametrize("w_in", [4])
@pytest.mark.parametrize(
    "head_args",
    [
        None,
        dict(name=None),
        dict(name="ClsHead", w_out=8),
        dict(name="ClsHead", w_out=8, width=8, norm_name="BN", act_name="relu"),
        dict(name="ClsHead", w_out=8, width=8, norm_name="BN", act_name="relu", bias=False),
        dict(name="VGGHead", w_out=8, width=8),
        dict(name="VGGHead", w_out=8, width=8, dropout_prob=0.5, act_name="relu"),
        dict(name="MBV3Head", w_out=8, width=8, w_h=16, norm_name="BN", act_name="relu"),
        dict(name="MBV3Head", w_out=8, width=8, w_h=16, norm_name="BN", act_name="relu", se_r=0.25),
        dict(
            name="MBV3Head",
            w_out=8,
            width=8,
            w_h=16,
            norm_name="BN",
            act_name="relu",
            se_r=0.25,
            bias=False,
        ),
    ],
)
@pytest.mark.parametrize("norm_name", ["BN"])
@pytest.mark.parametrize("act_name", ["relu"])
def test_build_head(w_in, head_args, norm_name, act_name):
    m = build_head(w_in, head_args, norm_name, act_name)
    if head_args is None or head_args.get("name") is None:
        assert m is None
    else:
        assert isinstance(m, M.Module)


def test_cls_head():
    C = 4
    K = 8

    x = np.random.rand(2, C, 8, 8).astype("float32")
    y = ClsHead(C, K)(mge.Tensor(x)).numpy()
    assert len(y.shape) == 2 and y.shape[1] == K


def test_mbv3_head():
    C = 4
    K = 8

    x = np.random.rand(2, C, 8, 8).astype("float32")
    y = MBV3Head(C, K, 8, 16)(mge.Tensor(x)).numpy()
    assert len(y.shape) == 2 and y.shape[1] == K


def test_vgg_head():
    C = 4
    K = 8

    x = np.random.rand(2, C, 8, 8).astype("float32")
    y = VGGHead(C, K, 8)(mge.Tensor(x)).numpy()
    assert len(y.shape) == 2 and y.shape[1] == K
