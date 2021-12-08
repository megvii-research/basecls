#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import numbers
import re
from collections import defaultdict
from typing import Dict, Iterable, Optional, Sequence, Tuple, Type, Union

import megengine.module as M
from loguru import logger
from megengine.tensor import Tensor

__all__ = ["get_param_groups"]

PatternType = Union[str, Type, Sequence[Type]]


def get_param_groups(
    module: M.Module, weight_decay_policy: Union[float, Sequence[Tuple[float, PatternType]]]
) -> Dict[float, Iterable[Tensor]]:
    """Directly get optimizer's param_groups with different weight decays
    given policy.

    ``cfg.solver.weight_decay`` can be a float or a sequence of weight decay policies.

    For example:

    .. code::

        cfg.solver.weight_decay = 1e-5

    is equivalent to

    .. code::

        cfg.solver.weight_decay = [
            1e-5
        ]

    Weight decay policy works in sequential order, i.e., for each parameter,
    we try patterns from the beginning to the end. If unmatched, default
    weight decay (-1) will be applied. For example:

    .. code::

        from basecls.layers import NORM_TYPES
        cfg.solver.weight_decay = [
            (1e-5, "bias"),
            (0, NORM_TYPES),
            1e-4,
        ]

    The parameter will first match ``(1e-5, "bias")`` then ``(0, NORM_TYPES)``,
    so any bias parameter, including the bias of normalization layers, will have
    weight decay ``1e-5``.

    For mobile models, e.g. mobilenet and shufflenet, you may want to disable weight
    decay for normalization layers and any bias. This can be achieved by the following:

    .. code::

        from basecls.layers import NORM_TYPES
        cfg.solver.weight_decay = [
            (0, "bias"),
            (0, NORM_TYPES),
            4e-5,
        ]

    Args:
        module: training model
        weight_decay_policy: weight decay policy defined in ``cfg.solver.weight_decay``
    """
    weight_decay_policy = _parse_weight_decay_policy(weight_decay_policy)

    # build mapping from weight_decay to parameters
    wd2params = defaultdict(list)
    wd2param_names = defaultdict(list)
    for n, p, m in module.named_parameters(with_parent=True):
        wd = _get_weight_decay(n, m, weight_decay_policy)
        if wd < 0:
            wd = weight_decay_policy[-1]
        assert wd >= 0, "weight decay should be non-negative"
        wd2params[wd].append(p)
        wd2param_names[wd].append(n)

    if len(wd2param_names) > 1:  # more than one param groups
        for wd, param_names in wd2param_names.items():
            logger.info(f"Following params has weight_decay = {wd}\n\t{param_names}")

    optim_params = []
    for group_wd, params in wd2params.items():
        optim_params.append({"params": params, "weight_decay": group_wd})
    return optim_params


def _parse_weight_decay_policy(policy):
    """Parse and verify weight decay policy"""
    if isinstance(policy, numbers.Number):
        policy = [policy]

    for p in policy[:-1]:
        assert (
            isinstance(p, Sequence) and len(p) == 2
        ), "weight decay policy should be a tuple of (value, pattern), get {p}"
        assert isinstance(p[0], numbers.Number), "value should be a number" f", get {p[0]}"
        assert isinstance(
            p[1], (str, Type, Sequence)
        ), "pattern should be string, type, or list of types, get {p[1]}"

    p = policy[-1]
    assert isinstance(
        p, numbers.Number
    ), f"weight decay should fallback into default value e.g. 1e-4, get {p}"

    return policy


def _get_weight_decay(
    name: str,
    module: M.Module,
    policy: Sequence[Tuple[Optional[float], Union[PatternType, Sequence[PatternType]]]],
) -> Optional[float]:
    """Get weight decay based on given weight decay policy.

    Args:
        name: name of parameter
        module: the module owning the parameter
        policy: weight decay policy defined in ``cfg.solver.weight_decay``
    """

    for p in policy:
        if isinstance(p, numbers.Number):
            # touch fallback
            return p
        else:
            wd, pattern = p

        if isinstance(pattern, str):
            if re.findall(pattern, name):
                return wd
        elif isinstance(pattern, (Type, Sequence)):
            if isinstance(module, pattern):
                return wd
        else:
            raise TypeError(f"Unexpected pattern {pattern}")
    raise RuntimeError(f"Why parameter {name} excapes weight decay assignment?")
