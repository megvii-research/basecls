#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import pytest

from basecls.configs import BaseConfig
from basecls.models.resnet import resnet18
from basecls.solver import DefaultSolver, Solver


@pytest.mark.parametrize("optimizer", ["adam", "adamw", "lamb", "lars", "sgd"])
def test_default_solver(optimizer):
    cfg = BaseConfig(solver=dict(optimizer=optimizer))
    model = resnet18()
    solver = DefaultSolver.build(cfg, model)
    assert isinstance(solver, Solver)
