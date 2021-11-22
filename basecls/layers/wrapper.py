#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2020 Ross Wightman
# This file has been modified by Megvii ("Megvii Modifications").
# All Megvii Modifications are Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import functools
import itertools
import math
import warnings
from typing import List, Sequence, Tuple, Union

import megengine as mge
import megengine.distributed as dist
import megengine.functional as F
import megengine.module as M
import numpy as np
from basecore.config import ConfigDict
from basecore.utils import all_reduce
from megengine.module.batchnorm import _BatchNorm
from megengine.utils.module_stats import register_flops

from basecls.data import DataLoaderType

__all__ = [
    "NORM_TYPES",
    "Preprocess",
    "adjust_block_compatibility",
    "calculate_fan_in_and_fan_out",
    "compute_precise_bn_stats",
    "init_weights",
    "init_vit_weights",
    "trunc_normal_",
    "lecun_normal_",
    "make_divisible",
]

NORM_TYPES = (_BatchNorm, M.GroupNorm, M.InstanceNorm, M.LayerNorm)


class Preprocess(M.Module):
    def __init__(self, mean: Union[float, Sequence[float]], std: Union[float, Sequence[float]]):
        super().__init__()
        self.mean = mge.Tensor(np.array(mean, dtype=np.float32).reshape(1, -1, 1, 1))
        self.std = mge.Tensor(np.array(std, dtype=np.float32).reshape(1, -1, 1, 1))

    def forward(self, inputs: Sequence[np.ndarray]) -> Tuple[mge.Tensor, mge.Tensor]:
        samples, targets = [mge.Tensor(x) for x in inputs]
        samples = (samples - self.mean) / self.std
        return samples, targets


def adjust_block_compatibility(
    ws: Sequence[int], bs: Sequence[float], gs: Sequence[int]
) -> Tuple[List[int], ...]:
    """Adjusts the compatibility of widths, bottlenecks and groups.

    Args:
        ws: widths.
        bs: bottleneck multipliers.
        gs: group widths.

    Returns:
        The adjusted widths, bottlenecks and groups.
    """
    assert len(ws) == len(bs) == len(gs)
    assert all(w > 0 and b > 0 and g > 0 for w, b, g in zip(ws, bs, gs))
    assert all(b < 1 or b % 1 == 0 for b in bs)
    vs = [int(max(1, w * b)) for w, b in zip(ws, bs)]
    gs = [int(min(g, v)) for g, v in zip(gs, vs)]
    ms = [np.lcm(g, int(b)) if b > 1 else g for g, b in zip(gs, bs)]
    vs = [max(m, int(round(v / m) * m)) for v, m in zip(vs, ms)]
    ws = [int(v / b) for v, b in zip(vs, bs)]
    assert all(w * b % g == 0 for w, b, g in zip(ws, bs, gs))
    return ws, bs, gs


def calculate_fan_in_and_fan_out(tensor: mge.Tensor, pytorch_style: bool = False):
    """Fixed :py:func:`megengine.module.init.calculate_fan_in_and_fan_out` for group conv2d.

    Note:
        The group conv2d kernel shape in MegEngine is ``(G, O/G, I/G, K, K)``. This function
        calculates ``fan_out = O/G * K * K`` as default, but PyTorch uses ``fan_out = O * K * K``.

    Args:
        tensor: tensor to be initialized.
        pytorch_style: utilize pytorch style init for group conv. Default: ``False``
    """
    if len(tensor.shape) not in (2, 4, 5):
        raise ValueError(
            "fan_in and fan_out can only be computed for tensor with 2/4/5 " "dimensions"
        )
    if len(tensor.shape) == 5:
        # `GOIKK` to `OIKK`
        tensor = tensor.reshape(-1, *tensor.shape[2:]) if pytorch_style else tensor[0]

    num_input_fmaps = tensor.shape[1]
    num_output_fmaps = tensor.shape[0]
    receptive_field_size = 1
    if len(tensor.shape) > 2:
        receptive_field_size = functools.reduce(lambda x, y: x * y, tensor.shape[2:], 1)
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out


def compute_precise_bn_stats(cfg: ConfigDict, model: M.Module, dataloader: DataLoaderType):
    """Computes precise BN stats on training data.

    References: https://github.com/facebookresearch/pycls/blob/main/pycls/core/net.py

    Args:
        cfg: config for precising BN.
        model: model for precising BN.
        dataloader: dataloader for precising BN.
    """
    # Prepare the preprocessor
    preprocess = Preprocess(cfg.preprocess.img_mean, cfg.preprocess.img_std)
    # Compute the number of minibatches to use
    num_iter = int(cfg.bn.num_samples_precise / cfg.batch_size / dist.get_world_size())
    num_iter = min(num_iter, len(dataloader))
    # Retrieve the BN layers
    bns = [m for m in model.modules() if isinstance(m, _BatchNorm)]
    # Initialize BN stats storage for computing mean(mean(batch)) and mean(var(batch))
    running_means = [F.zeros_like(bn.running_mean) for bn in bns]
    running_vars = [F.zeros_like(bn.running_var) for bn in bns]
    # Remember momentum values
    momentums = [bn.momentum for bn in bns]
    # Set momentum to 0.0 to compute BN stats that only reflect the current batch
    for bn in bns:
        bn.momentum = 0.0
    # Average the BN stats for each BN layer over the batches
    for data in itertools.islice(dataloader, num_iter):
        samples, _ = preprocess(data)
        model(samples)
        for i, bn in enumerate(bns):
            running_means[i] += bn.running_mean / num_iter
            running_vars[i] += bn.running_var / num_iter
    # Sync BN stats across GPUs (no reduction if 1 GPU used)
    running_means = [all_reduce(x, mode="mean") for x in running_means]
    running_vars = [all_reduce(x, mode="mean") for x in running_vars]
    # Set BN stats and restore original momentum values
    for i, bn in enumerate(bns):
        bn.running_mean = running_means[i]
        bn.running_var = running_vars[i]
        bn.momentum = momentums[i]


def init_weights(m: M.Module, pytorch_style: bool = False, zero_init_final_gamma: bool = False):
    """Performs ResNet-style weight initialization.

    About zero-initialize:
    Zero-initialize the last BN in each residual branch, so that the residual branch starts
    with zeros, and each residual block behaves like an identity.
    This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677.

    References: https://github.com/facebookresearch/pycls/blob/main/pycls/models/blocks.py

    Args:
        m: module to be initialized.
        pytorch_style: utilize pytorch style init for group conv. Default: ``False``
        zero_init_final_gamma: enable zero-initialize or not. Default: ``False``
    """

    if isinstance(m, M.Conv2d):
        _, fan_out = calculate_fan_in_and_fan_out(m.weight, pytorch_style)
        std = math.sqrt(2 / fan_out)
        M.init.normal_(m.weight, 0, std)
        if getattr(m, "bias", None) is not None:
            fan_in, _ = calculate_fan_in_and_fan_out(m.weight, pytorch_style)
            bound = 1 / math.sqrt(fan_in)
            M.init.uniform_(m.bias, -bound, bound)
    elif isinstance(m, NORM_TYPES):
        M.init.fill_(
            m.weight, 0.0 if getattr(m, "final_bn", False) and zero_init_final_gamma else 1.0
        )
        M.init.zeros_(m.bias)
    elif isinstance(m, M.Linear):
        M.init.normal_(m.weight, std=0.01)
        if getattr(m, "bias", None) is not None:
            M.init.zeros_(m.bias)


def init_vit_weights(module: M.Module):
    """Initialization for Vision Transformer (ViT).

    References:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

    Args:
        m: module to be initialized.
    """
    if isinstance(module, M.Linear):
        if module.name and module.name.startswith("head"):
            M.init.zeros_(module.weight)
            M.init.zeros_(module.bias)
        elif module.name and module.name.startswith("pre_logits"):
            lecun_normal_(module.weight)
            M.init.zeros_(module.bias)
        else:
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                M.init.zeros_(module.bias)
    elif isinstance(module, M.Conv2d):
        M.init.msra_uniform_(module.weight, a=math.sqrt(5))
        if module.bias is not None:
            fan_in, _ = M.init.calculate_fan_in_and_fan_out(module.weight)
            bound = 1 / math.sqrt(fan_in)
            M.init.uniform_(module.bias, -bound, bound)
    elif isinstance(module, (M.LayerNorm, M.GroupNorm, M.BatchNorm2d)):
        M.init.zeros_(module.bias)
        M.init.ones_(module.weight)


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    # Values are generated by using a truncated uniform distribution and
    # then using the inverse CDF for the normal distribution.
    # Get upper and lower cdf values
    lo = norm_cdf((a - mean) / std)
    up = norm_cdf((b - mean) / std)

    # Uniformly fill tensor with values from [l, u], then translate to
    # [2l-1, 2u-1].
    M.init.uniform_(tensor, 2 * lo - 1, 2 * up - 1)
    # tensor.uniform_(2 * l - 1, 2 * u - 1)

    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal
    tensor._reset(M.Elemwise("erfinv")(tensor))
    # tensor.erfinv_()

    # Transform to proper mean, std
    tensor *= std * math.sqrt(2.0)
    # tensor.mul_(std * math.sqrt(2.))
    tensor += mean
    # tensor.add_(mean)

    # Clamp to ensure it's in the proper range
    tensor._reset(F.clip(tensor, lower=a, upper=b))
    # tensor.clamp_(min=a, max=b)
    return tensor


def lecun_normal_(tensor):
    fan_in, _ = calculate_fan_in_and_fan_out(tensor)
    std = 1 / math.sqrt(fan_in) / 0.87962566103423978
    # constant is stddev of standard normal truncated to (-2, 2)
    trunc_normal_(tensor, std=std)


def make_divisible(value, divisor: int = 8, min_value: int = None, round_limit: float = 0.0):
    min_value = min_value or divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < round_limit * value:
        new_value += divisor
    return new_value


# FIXME: adaptive pool on 1x1 feature map lead to div_zero error
@register_flops(M.AdaptiveAvgPool2d, M.AdaptiveMaxPool2d)
def flops_adaptivePool(module: M.AdaptiveAvgPool2d, inputs, outputs):
    stride_h = (
        np.floor(inputs[0].shape[2] / (inputs[0].shape[2] - 1)) if (inputs[0].shape[2] - 1) else 0
    )
    kernel_h = inputs[0].shape[2] - (inputs[0].shape[2] - 1) * stride_h
    stride_w = (
        np.floor(inputs[0].shape[3] / (inputs[0].shape[3] - 1)) if (inputs[0].shape[3] - 1) else 0
    )
    kernel_w = inputs[0].shape[3] - (inputs[0].shape[3] - 1) * stride_w
    return np.prod(outputs[0].shape) * kernel_h * kernel_w
