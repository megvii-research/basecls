#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
from typing import Union

import megengine.data as data
import megengine.data.transform as T
from basecore.config import ConfigDict

from basecls.utils import registers

from .dataloader import build_dataloader
from .fake_data import FakeDataLoader

__all__ = ["DataLoaderType", "FakeData", "FolderLoader"]

DataLoaderType = Union[data.DataLoader, FakeDataLoader]


@registers.dataloaders.register()
class FakeData:
    """Fake data useful for benchmark."""

    @classmethod
    def build(
        cls, cfg: ConfigDict, train: bool = True, augments: T.Transform = None
    ) -> data.DataLoader:
        return FakeDataLoader(
            batch_size=cfg.batch_size,
            img_size=cfg.preprocess.img_size,
            channels=1 if cfg.preprocess.img_color_space == "GRAY" else 3,
            length=200,
            num_classes=cfg.num_classes,
        )


@registers.dataloaders.register()
class FolderLoader:
    """Local dataloader factory.

    The source is the local folder.
    """

    @classmethod
    def build(
        cls, cfg: ConfigDict, train: bool = True, augments: T.Transform = None
    ) -> data.DataLoader:
        return build_dataloader(cfg, train, augments, mode="folder")
