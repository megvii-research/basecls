#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import pytest

from basecls.layers import NORM_TYPES
from basecls.models.resnet import resnet18
from basecls.solver.optimizer import SGD
from basecls.solver.weight_decay import get_param_groups


@pytest.mark.parametrize("weight_decay", [0, 1e-4, [(1e-5, "bias"), (0, NORM_TYPES), 1e-4]])
def test_weight_decay(weight_decay):
    model = resnet18()
    params = get_param_groups(model, weight_decay)
    SGD(params, 0.1, momentum=0.9)
