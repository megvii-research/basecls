#!/usr/bin/env python3
# Copyright (c) 2020 Ross Wightman
# This file has been modified by Megvii ("Megvii Modifications").
# All Megvii Modifications are Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
"""Mixup and CutMix

Mixup: `"Mixup: Beyond Empirical Risk Minimization" <https://arxiv.org/abs/1710.09412>`_

CutMix: `"CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features"
<https://arxiv.org/abs/1905.04899>`_

References:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/mixup.py
"""
from typing import List, Sequence, Tuple, Union

import megengine.data as data
import megengine.data.transform as T
import numpy as np

__all__ = ["MixupCutmixTransform", "MixupCutmixCollator"]


def _one_hot(
    x: np.ndarray, num_classes: int, on_value: float = 1.0, off_value: float = 0.0
) -> np.ndarray:
    one_hot = np.full((x.shape[0], num_classes), off_value)
    np.put_along_axis(one_hot, x[:, np.newaxis], values=on_value, axis=1)
    return one_hot


def _mixup_target(
    target: np.ndarray,
    num_classes: int,
    lam: Union[float, np.ndarray] = 1.0,
    perm: Sequence[int] = None,
) -> np.ndarray:
    if perm is None:
        perm = slice(None, None, -1)
    y1 = _one_hot(target, num_classes)
    y2 = _one_hot(target[perm], num_classes)
    return y1 * lam + y2 * (1.0 - lam)


def _rand_bbox(img_shape: Tuple, lam: float, margin: float = 0.0, count: int = None) -> Tuple:
    """Standard CutMix bounding-box.

    Generates a random square bbox based on lambda value. This impl includes
    support for enforcing a border margin as percent of bbox dimensions.

    Args:
        img_shape: image shape as tuple.
        lam: cutmix lambda value.
        margin: percentage of bbox dimension to enforce as margin
            (reduce amount of box outside image).
        count: number of bbox to generate.

    Returns:
        (y1, y2, x1, x2) represents (up, down, left, righ) of bbox.
    """
    ratio = np.sqrt(1.0 - lam)
    img_h, img_w = img_shape
    cut_h, cut_w = int(img_h * ratio), int(img_w * ratio)
    margin_y, margin_x = int(margin * cut_h), int(margin * cut_w)
    cy = np.random.randint(0 + margin_y, img_h - margin_y, size=count)
    cx = np.random.randint(0 + margin_x, img_w - margin_x, size=count)
    y1 = np.clip(cy - cut_h // 2, 0, img_h)
    y2 = np.clip(cy + cut_h // 2, 0, img_h)
    x1 = np.clip(cx - cut_w // 2, 0, img_w)
    x2 = np.clip(cx + cut_w // 2, 0, img_w)
    return y1, y2, x1, x2


def _rand_bbox_minmax(img_shape: Tuple, minmax: Sequence[float], count: int = None) -> Tuple:
    """Min-Max CutMix bounding-box.

    Inspired by Darknet cutmix impl, generates a random rectangular bbox
    based on min/max percent values applied to each dimension of the input image.

    Typical defaults for minmax are usually in the  .2-.3 for min and .8-.9 range for max.

    Args:
        img_shape: image shape as tuple.
        minmax: min and max bbox ratios (as percent of image size).
        count: number of bbox to generate.

    Returns:
        (y1, y2, x1, x2) represents (up, down, left, righ) of bbox.
    """
    assert len(minmax) == 2
    img_h, img_w = img_shape
    cut_h = np.random.randint(int(img_h * minmax[0]), int(img_h * minmax[1]), size=count)
    cut_w = np.random.randint(int(img_w * minmax[0]), int(img_w * minmax[1]), size=count)
    y1 = np.random.randint(0, img_h - cut_h, size=count)
    x1 = np.random.randint(0, img_w - cut_w, size=count)
    y2 = y1 + cut_h
    x2 = x1 + cut_w
    return y1, y2, x1, x2


def _cutmix_bbox_and_lam(
    img_shape: Tuple,
    lam: float,
    ratio_minmax: Sequence[float] = None,
    correct_lam: bool = True,
    count: int = None,
) -> Tuple:
    """Generate bbox and apply lambda correction."""
    if ratio_minmax is not None:
        y1, y2, x1, x2 = _rand_bbox_minmax(img_shape, ratio_minmax, count=count)
    else:
        y1, y2, x1, x2 = _rand_bbox(img_shape, lam, count=count)
    if correct_lam or ratio_minmax is not None:
        bbox_area = (y2 - y1) * (x2 - x1)
        lam = 1.0 - bbox_area / float(img_shape[-2] * img_shape[-1])
    return (y1, y2, x1, x2), lam


class MixupCutmixTransform(T.VisionTransform):
    """Implement Mixup and CutMix as VisionTransform.

    .. note::

        When composed in :py:class:`~megengine.data.transform.Compose` ,
        ``batch_compose`` must be set to ``True``.

    Args:
        mixup_alpha: mixup alpha value, mixup is active if > 0. Default: ``1.0``
        cutmix_alpha: cutmix alpha value, cutmix is active if > 0. Default: ``0.0``
        cutmix_minmax: cutmix min/max image ratio, cutmix is active and uses this vs alpha
            if not None. Default: ``None``
        prob: probability of applying mixup or cutmix per batch or element. Default: ``1.0``
        switch_prob: probability of switching to cutmix instead of mixup when both are active.
            Default: 0.5
        mode: how to apply mixup/cutmix params, supports  ``"batch"``, ``"pair"``
            (pair of elements) and ``"elem"`` (element). Default: ``"batch"``
        data_format: ``"CHW"`` or ``"HWC"``, use ``"HWC"`` if use this transform before
            ``T.ToMode()``. Default: ``"HWC"``
        num_classes: number of classes for target. Default: ``1000``
        calibrate_cutmix_lambda: apply lambda correction when cutmix bbox clipped by image borders.
            Correction is based on clipped area for cutmix. Default: ``True``
        calibrate_mixup_lambda: enforce mixup lambda to be greater than 0.5, only make difference
            in ``"elem"`` mode. Default: ``False``
        permute: whether mixup with permuted samples instead of flipped samples. Default: ``False``
    """

    def __init__(
        self,
        mixup_alpha: float = 1.0,
        cutmix_alpha: float = 0.0,
        cutmix_minmax: List[float] = None,
        prob: float = 1.0,
        switch_prob: float = 0.5,
        mode: str = "batch",
        data_format: str = "HWC",
        num_classes: int = 1000,
        calibrate_cutmix_lambda: bool = True,
        calibrate_mixup_lambda: bool = False,
        permute: bool = False,
        *,
        order=None,
    ):
        super().__init__(order)
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_minmax = cutmix_minmax
        if self.cutmix_minmax is not None:
            assert len(self.cutmix_minmax) == 2
            # force cutmix alpha == 1.0 when minmax active to keep logic simple & safe
            self.cutmix_alpha = 1.0
        self.mix_prob = prob
        self.switch_prob = switch_prob
        if mode not in ("batch", "pair", "elem"):
            raise ValueError(f"Mixup/CutMix mode '{mode}' not supported")
        self.mode = mode
        self.calibrate_cutmix_lambda = calibrate_cutmix_lambda
        if data_format not in ("CHW", "HWC"):
            raise ValueError(f"Data format '{data_format}' not supported")
        self.data_format = data_format
        self.num_classes = num_classes
        self.calibrate_mixup_lambda = calibrate_mixup_lambda
        self.permute = permute

    def _params_per_elem(self, batch_size: int):
        lam = np.ones(batch_size, dtype=np.float32)
        use_cutmix = np.zeros(batch_size, dtype=np.bool)
        if self.mixup_alpha == 0.0 and self.cutmix_alpha == 0.0:
            return lam, use_cutmix
        if self.mixup_alpha > 0.0 and self.cutmix_alpha > 0.0:
            use_cutmix = np.random.rand(batch_size) < self.switch_prob
            lam_mix = np.where(
                use_cutmix,
                np.random.beta(self.cutmix_alpha, self.cutmix_alpha, size=batch_size),
                np.random.beta(self.mixup_alpha, self.mixup_alpha, size=batch_size),
            )
        elif self.mixup_alpha > 0.0:
            lam_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha, size=batch_size)
        elif self.cutmix_alpha > 0.0:
            use_cutmix = np.ones(batch_size, dtype=np.bool)
            lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha, size=batch_size)
        else:
            raise ValueError(
                "One of mixup_alpha > 0.0, cutmix_alpha > 0.0,"
                "cutmix_minmax not None should be true."
            )
        if self.calibrate_mixup_lambda:
            lam_mix = np.where(use_cutmix, lam_mix, np.maximum(lam_mix, 1 - lam_mix))
        lam = np.where(np.random.rand(batch_size) < self.mix_prob, lam_mix.astype(np.float32), lam)
        return lam, use_cutmix

    def _params_per_batch(self):
        lam = 1.0
        use_cutmix = False
        if (self.mixup_alpha > 0.0 or self.cutmix_alpha > 0.0) and np.random.rand() < self.mix_prob:
            if self.mixup_alpha > 0.0 and self.cutmix_alpha > 0.0:
                use_cutmix = np.random.rand() < self.switch_prob
                lam_mix = (
                    np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
                    if use_cutmix
                    else np.random.beta(self.mixup_alpha, self.mixup_alpha)
                )
            elif self.mixup_alpha > 0.0:
                lam_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            elif self.cutmix_alpha > 0.0:
                use_cutmix = True
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            else:
                raise ValueError(
                    "One of mixup_alpha > 0.0, cutmix_alpha > 0.0,"
                    "cutmix_minmax not None should be true."
                )
            lam = float(lam_mix)
        return lam, use_cutmix

    def _mix_elem(self, x, perm: Sequence[int]):
        batch_size = len(x)
        lam_batch, use_cutmix = self._params_per_elem(batch_size)
        x_ori = x.copy()  # need to keep an unmodified original for mixing source
        for i in range(batch_size):
            j = perm[i]
            lam = lam_batch[i]
            if lam != 1.0:
                if use_cutmix[i]:
                    (y1, y2, x1, x2), lam = _cutmix_bbox_and_lam(
                        x.shape[-2:] if self.data_format == "CHW" else x.shape[1:3],
                        lam,
                        ratio_minmax=self.cutmix_minmax,
                        correct_lam=self.calibrate_cutmix_lambda,
                    )
                    if self.data_format == "CHW":
                        x[i, :, y1:y2, x1:x2] = x_ori[j, :, y1:y2, x1:x2]
                    else:
                        x[i, y1:y2, x1:x2] = x_ori[j, y1:y2, x1:x2]
                    lam_batch[i] = lam
                else:
                    x[i] = x[i] * lam + x_ori[j] * (1.0 - lam)
        return lam_batch[:, np.newaxis]

    def _mix_pair(self, x, perm: Sequence[int]):
        batch_size = len(x)
        lam_batch, use_cutmix = self._params_per_elem(batch_size // 2)
        x_ori = x.copy()  # need to keep an unmodified original for mixing source
        for i in range(batch_size // 2):
            j = perm[i]
            lam = lam_batch[i]
            if lam != 1.0:
                if use_cutmix[i]:
                    (y1, y2, x1, x2), lam = _cutmix_bbox_and_lam(
                        x.shape[-2:] if self.data_format == "CHW" else x.shape[1:3],
                        lam,
                        ratio_minmax=self.cutmix_minmax,
                        correct_lam=self.calibrate_cutmix_lambda,
                    )
                    if self.data_format == "CHW":
                        x[i, :, y1:y2, x1:x2] = x_ori[j, :, y1:y2, x1:x2]
                        x[j, :, y1:y2, x1:x2] = x_ori[i, :, y1:y2, x1:x2]
                    else:
                        x[i, y1:y2, x1:x2] = x_ori[j, y1:y2, x1:x2]
                        x[j, y1:y2, x1:x2] = x_ori[i, y1:y2, x1:x2]
                    lam_batch[i] = lam
                else:
                    x[i] = x[i] * lam + x_ori[j] * (1.0 - lam)
                    x[j] = x[j] * lam + x_ori[i] * (1.0 - lam)
        lam_batch = np.concatenate((lam_batch, lam_batch[perm]))
        return lam_batch[:, np.newaxis]

    def _mix_batch(self, x, perm: Sequence[int]):
        lam, use_cutmix = self._params_per_batch()
        if lam == 1.0:
            return 1.0
        if use_cutmix:
            (y1, y2, x1, x2), lam = _cutmix_bbox_and_lam(
                x.shape[-2:] if self.data_format == "CHW" else x.shape[1:3],
                lam,
                ratio_minmax=self.cutmix_minmax,
                correct_lam=self.calibrate_cutmix_lambda,
            )
            if self.data_format == "CHW":
                x[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
            else:
                x[:, y1:y2, x1:x2] = x[perm, y1:y2, x1:x2]
        else:
            x[:] = x * lam + x[perm] * (1.0 - lam)
        return lam

    def apply_batch(self, inputs: Sequence[Tuple]):
        images, targets = tuple(zip(*inputs))
        images = np.stack(images)
        targets = np.stack(targets)

        dtype = images.dtype
        images = images.astype(np.float32)

        batch_size = len(images)
        if self.mode == "pair":
            batch_size = batch_size // 2
        if self.permute:
            perm = np.random.permutation(batch_size)
        else:
            perm = np.arange(batch_size)[::-1]

        if self.mode == "elem":
            lam = self._mix_elem(images, perm=perm)
        elif self.mode == "pair":
            lam = self._mix_pair(images, perm=perm)
        else:
            lam = self._mix_batch(images, perm=perm)
        targets = _mixup_target(targets, self.num_classes, lam, perm=perm)

        return tuple(zip(images.clip(0, 255).astype(dtype), targets.astype(np.float32)))


class MixupCutmixCollator(data.Collator):
    """A faster version implemented as a collator."""

    def __init__(self, *args, **kwargs):
        self.collator = data.Collator()
        self.transform = MixupCutmixTransform(*args, data_format="CHW", **kwargs)

    def apply(self, inputs: Sequence[Tuple]):
        inputs = self.transform.apply_batch(inputs)
        inputs = self.collator.apply(inputs)
        return inputs
