#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import tempfile

import pytest

from basecls.configs import BaseConfig
from basecls.data import build_dataset


@pytest.mark.parametrize(
    "dataset_path,train,mode",
    [
        (None, True, "folder"),
        (None, False, "folder"),
    ],
)
def test_build_dataset_by_path(dataset_path, train, mode):
    with tempfile.TemporaryDirectory() as folder_path:
        subset_name = "train" if train else "val"

        data_dict = dict(num_workers=2)
        data_dict[f"{subset_name}_path"] = folder_path if dataset_path is None else dataset_path
        cfg = BaseConfig(data=data_dict)
        build_dataset(cfg, train, mode)
