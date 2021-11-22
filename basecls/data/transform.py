#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import copy
from typing import Optional, Sequence

import cv2
import megengine.data as data
import megengine.data.transform as T
import numpy as np
from basecore.config import ConfigDict
from loguru import logger

from basecls.utils import registers

from .augment import WARP_PARAMS, TorchAutoAugment, TorchRandAugment
from .const import CV2_INTERP, PIL_INTERP
from .mixup import MixupCutmixCollator
from .rand_erase import RandomErasing

__all__ = [
    "build_transform",
    "AutoAugment",
    "SimpleAugment",
    "ColorAugment",
    "RandAugment",
    "build_mixup",
]


def build_transform(
    cfg: ConfigDict, train: bool = True, augments: T.Transform = None
) -> T.Transform:
    """Build function for MegEngine transform.

    Args:
        cfg: config for building transform.
        train: train set or test set. Default: ``True``
        augments: augments for building transform.

    Returns:
        A transform.
    """
    if train:
        assert augments is not None
        bgr_mean = copy.deepcopy(cfg.preprocess.img_mean)
        bgr_std = copy.deepcopy(cfg.preprocess.img_std)
        if cfg.preprocess.img_color_space == "RGB":
            bgr_mean = bgr_mean[::-1]
            bgr_std = bgr_std[::-1]
        WARP_PARAMS["fillcolor"] = tuple(round(v) for v in bgr_mean[::-1])  # need RGB
        WARP_PARAMS["resample"] = PIL_INTERP[cfg.augments.resize.interpolation]

        transforms = [
            T.RandomResizedCrop(
                cfg.preprocess.img_size,
                cfg.augments.resize.scale_range,
                cfg.augments.resize.ratio_range,
                CV2_INTERP[cfg.augments.resize.interpolation],
            ),
            T.RandomHorizontalFlip(),
            augments,
            RandomErasing(
                **cfg.augments.rand_erase.to_dict(),
                pad_mean=bgr_mean,  # need BGR
                pad_std=bgr_std,  # need BGR
            ),
            ToColorSpace(cfg.preprocess.img_color_space),
            T.ToMode(),
        ]
    else:
        assert augments is None
        transforms = [
            T.Resize(
                int(cfg.test.img_size / cfg.test.crop_pct / 2 + 0.5) * 2,  # make it even
                CV2_INTERP[cfg.augments.resize.interpolation],
            ),
            T.CenterCrop(cfg.test.img_size),
            ToColorSpace(cfg.preprocess.img_color_space),
            T.ToMode(),
        ]
    return T.Compose(transforms=transforms, order=["image", "image_category"])


class ToColorSpace(T.VisionTransform):
    """Transform to transfer color space.

    Args:
        color_space: color space, supports ``"BGR"``, ``"RGB"`` and ``"GRAY"``.
    """

    def __init__(self, color_space: str, *, order: Sequence = None):
        super().__init__(order)
        if color_space not in ("BGR", "RGB", "GRAY"):
            raise ValueError(f"Color space '{color_space}' not supported")
        self.color_space = color_space

    def _apply_image(self, image: np.ndarray) -> np.ndarray:
        if self.color_space == "BGR":
            return image
        elif self.color_space == "RGB":
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif self.color_space == "GRAY":
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[..., np.newaxis]
        else:
            raise ValueError(f"Color space '{self.color_space}' not supported")


@registers.augments.register()
class SimpleAugment:
    """Simple augmentation."""

    @classmethod
    def build(cls, cfg: ConfigDict) -> T.Transform:
        return T.PseudoTransform()


@registers.augments.register()
class ColorAugment:
    """Color augmentation."""

    @classmethod
    def build(cls, cfg: ConfigDict) -> T.Transform:
        aug_args = cfg.augments.color_aug.to_dict()
        lighting_scale = aug_args.pop("lighting")
        return T.Compose([T.ColorJitter(**aug_args), T.Lighting(lighting_scale)])


@registers.augments.register()
class AutoAugment:
    """AutoAugment."""

    @classmethod
    def build(cls, cfg: ConfigDict) -> T.Transform:
        return T.TorchTransformCompose([TorchAutoAugment()])


@registers.augments.register()
class RandAugment:
    """Random augmentation."""

    @classmethod
    def build(cls, cfg: ConfigDict) -> T.Transform:
        return T.TorchTransformCompose([TorchRandAugment(**cfg.augments.rand_aug.to_dict())])


def build_mixup(cfg: ConfigDict, train: bool = True) -> Optional[data.Collator]:
    """Build (optionally) Mixup/CutMix augment.

    Args:
        cfg: config for building Mixup/CutMix collator.
        train: train set or test set. Default: ``True``

    Returns:
        :py:class:`~basecls.data.mixup.MixupCutmixCollator` or ``None``
    """
    mixup_cfg = cfg.augments.mixup
    if train and (
        mixup_cfg.mixup_alpha > 0.0
        or mixup_cfg.cutmix_alpha > 0.0
        or mixup_cfg.cutmix_minmax is not None
    ):
        mixup_collator = MixupCutmixCollator(**mixup_cfg.to_dict(), num_classes=cfg.num_classes)
        logger.info(f"Using mixup with configuration:\n{mixup_cfg}")
    else:
        mixup_collator = None
    return mixup_collator
