#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
from basecore.config import ConfigDict
from megengine.data.dataset import ImageFolder, VisionDataset

__all__ = ["build_dataset"]


def build_dataset(cfg: ConfigDict, train: bool = True, mode: str = "folder") -> VisionDataset:
    """Build function for dataset.

    Args:
        cfg: config for building dataset.
        train: train set or test set. Default: ``True``

    Returns:
        A dataset which loads data from Nori on OSS.
    """
    subset_name = "train" if train else "val"

    if mode == "folder":
        if getattr(cfg.data, f"{subset_name}_path", None) is None:
            raise KeyError(f"Dataset mode 'folder' need to specify cfg.data.{subset_name}_path")
        return ImageFolder(getattr(cfg.data, f"{subset_name}_path"))
    else:
        raise NotImplementedError
