#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import megengine as mge
import numpy as np
import pytest
from megengine.autodiff import GradManager

from basecls.solver.optimizer import LAMB


@pytest.mark.parametrize("weight_decay", [0.0, 0.001])
@pytest.mark.parametrize("betas", [(0.9, 0.999)])
def test_lamb(weight_decay, betas):
    w = np.random.rand(4, 8).astype("float32")
    x = np.random.rand(8, 2).astype("float32")
    lr = 0.1
    n = 5

    w = mge.Parameter(w)
    x = mge.Tensor(x)
    gm = GradManager()
    gm.attach([w])
    lamb = LAMB([w], lr=lr, weight_decay=weight_decay, betas=betas)
    for _ in range(n):
        with gm:
            y = (w @ x).sum()
            gm.backward(y)
        lamb.step().clear_grad()
