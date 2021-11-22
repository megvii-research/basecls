#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import tempfile

import pytest

from basecls.configs import BaseConfig
from basecls.data import ColorAugment, FakeData, FolderLoader


@pytest.mark.parametrize("Data", [FakeData, FolderLoader])
def test_data(Data):
    with tempfile.TemporaryDirectory() as train_path:
        cfg = BaseConfig(data=dict(num_workers=2))
        if Data == FolderLoader:
            cfg.data.train_path = train_path

        augments = ColorAugment.build(cfg)

        if Data in [FolderLoader]:
            Data.build(cfg, True, augments)
        else:
            Data.build(cfg, augments)
