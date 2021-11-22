#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import tempfile

import megengine.data as data
import pytest

from basecls.configs import BaseConfig
from basecls.data import ColorAugment, build_dataloader


@pytest.mark.parametrize("train", [True, False])
def test_build_folderloader(train):
    with tempfile.TemporaryDirectory() as dataset_path:
        subset_name = "train" if train else "val"
        data_dict = dict(num_workers=2)
        data_dict[f"{subset_name}_path"] = dataset_path
        cfg = BaseConfig(data=data_dict)

        augments = ColorAugment.build(cfg) if train else None
        dataloader = build_dataloader(cfg, train, augments, "folder")
        assert isinstance(dataloader, data.DataLoader)
