#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import megengine.module as M

from basecls.configs import BaseConfig
from basecls.models import build_model, load_model, sync_model
from basecls.models.resnet import resnet18

# from basecls.utils import registers


def test_build_model():
    cfg = BaseConfig(model=dict(name="resnet18"))
    m = build_model(cfg)
    assert isinstance(m, M.Module)
    assert m.head.fc.out_features == 1000

    cfg = BaseConfig(model=dict(name="resnet18"), num_classes=10)
    m = build_model(cfg)
    assert m.head.fc.out_features == 10

    cfg = BaseConfig(model=dict(name="resnet18", head=None))
    m = build_model(cfg)
    assert m.head is None

    cfg = BaseConfig(model=dict(name="resnet18", head=dict(w_out=10)), num_classes=100)
    m = build_model(cfg)
    assert m.head.fc.out_features == 10


def test_load_model():
    m = resnet18()
    load_model(
        m, "https://data.megengine.org.cn/research/basecls/models/resnet/resnet18/resnet18.pkl"
    )


def test_sync_model():
    m = resnet18()
    sync_model(m)


# def test_all_models():
#     for m_cls in registers.models.values():
#         if not isinstance(m_cls, type):
#             m = m_cls()
#             assert isinstance(m, M.Module)
