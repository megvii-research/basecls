#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import megengine.data as data
import megengine.data.transform as T
from basecore.config import ConfigDict

from .dataset import build_dataset
from .transform import build_mixup, build_transform

__all__ = ["build_dataloader"]


def build_dataloader(
    cfg: ConfigDict,
    train: bool,
    augments: T.Transform = None,
    mode: str = "folder",
    infinite: bool = False,
    rank: int = None,
) -> data.DataLoader:
    """Build function for MegEngine dataloader.

    Args:
        cfg: config for building dataloader.
        train: train set or test set.
        augments: augments for building dataloder. Default: ``None``
        infinite: make dataloader infinite or not. default: ``False``
        rank: machine rank, only useful for infinite dataloader. Default: ``None``

    Returns:
        A dataloader.
    """
    dataset = build_dataset(cfg, train, mode)
    if train:
        if infinite:  # for DPFlow producer
            assert rank is not None
            sampler = data.Infinite(
                data.RandomSampler(
                    dataset,
                    cfg.batch_size,
                    drop_last=True,
                    world_size=1,
                    rank=0,
                    seed=cfg.seed + rank,
                )
            )
        else:
            sampler = data.RandomSampler(dataset, cfg.batch_size, drop_last=True, seed=cfg.seed)
    else:
        sampler = data.SequentialSampler(dataset, 25)  # can divide 50000 / 8 = 6250
    transform = build_transform(cfg, train, augments)

    mixup = build_mixup(cfg, train)

    dataloader = data.DataLoader(
        dataset,
        sampler=sampler,
        transform=transform,
        num_workers=cfg.data.num_workers,
        collator=mixup,
    )
    return dataloader
