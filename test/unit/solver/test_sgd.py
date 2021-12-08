#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import megengine as mge
import numpy as np
import pytest
import torch
import torch.optim as optim
from megengine.autodiff import GradManager

from basecls.solver.optimizer import SGD


@pytest.mark.parametrize("weight_decay", [0.0, 0.001])
@pytest.mark.parametrize("momentum,nesterov", [(0.0, False), (0.9, False), (0.9, True)])
def test_sgd(weight_decay, momentum, nesterov):
    w = np.random.rand(4, 8).astype("float32")
    x = np.random.rand(8, 2).astype("float32")
    lr = 0.1
    n = 5

    mw = mge.Parameter(w)
    mx = mge.Tensor(x)
    gm = GradManager()
    gm.attach([mw])
    msgd = SGD([mw], lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov)
    for _ in range(n):
        with gm:
            my = (mw @ mx).sum()
            gm.backward(my)
        msgd.step().clear_grad()

    tw = torch.tensor(w, requires_grad=True)
    tx = torch.tensor(x)
    tsgd = optim.SGD([tw], lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov)
    for _ in range(n):
        ty = (tw @ tx).sum()
        ty.backward()
        tsgd.step()
        tsgd.zero_grad(set_to_none=True)

    np.testing.assert_allclose(mw.numpy(), tw.detach().numpy(), rtol=1e-4, atol=1e-6)
