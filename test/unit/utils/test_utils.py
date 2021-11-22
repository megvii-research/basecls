#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import pytest

from basecls.utils import recursive_update


@pytest.mark.parametrize(
    "d,u,t",
    [
        (dict(a=1, b=2), dict(a=3, c=4), dict(a=3, b=2, c=4)),
        (dict(a=1, b=dict(a=2, b=3)), dict(a=4, b=5), dict(a=4, b=5)),
        (dict(a=1, b=2), dict(a=3, b=dict(a=4)), dict(a=3, b=dict(a=4))),
        (
            dict(a=1, b=dict(a=2, b=3)),
            dict(a=4, b=dict(a=5, c=6)),
            dict(a=4, b=dict(a=5, b=3, c=6)),
        ),
        (dict(a=1, b=[2, 3, 4]), dict(b=[5]), dict(a=1, b=[5])),
    ],
)
def test_recursive_update(d, u, t):
    assert recursive_update(d, u) == t
