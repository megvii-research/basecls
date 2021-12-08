#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import megengine as mge
import numpy as np
import pytest
from megengine.autodiff import GradManager

from basecls.solver.optimizer import LARS


@pytest.mark.parametrize("weight_decay", [0.0, 0.001])
@pytest.mark.parametrize("momentum,nesterov", [(0.0, False), (0.9, False), (0.9, True)])
def test_lars(weight_decay, momentum, nesterov):
    w = np.random.rand(4, 8).astype("float32")
    x = np.random.rand(8, 2).astype("float32")
    lr = 0.1
    n = 5

    w = mge.Parameter(w)
    x = mge.Tensor(x)
    gm = GradManager()
    gm.attach([w])
    lars = LARS([w], lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov)
    for _ in range(n):
        with gm:
            y = (w @ x).sum()
            gm.backward(y)
        lars.step().clear_grad()
