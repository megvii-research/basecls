#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import os
import pathlib
import pickle
from dataclasses import dataclass
from datetime import datetime
from typing import Union

import megengine as mge
from megfile import smart_open

import basecls


@dataclass
class Meta:
    uid: str
    series: str
    name: str

    create_time: str = datetime.now().strftime("%Y/%m/%d,%H:%M:%S")
    basecls_version: str = basecls.__version__

    flops: int = None
    params: int = None
    activations: int = None
    img_size: int = None

    acc1: float = None
    acc5: float = None


def get_series_and_name_and_id(p: Union[str, pathlib.PosixPath]) -> dict:
    if isinstance(p, str):
        p = pathlib.Path(p)
    series = p.absolute().resolve().parent.name
    name = p.with_suffix("").name
    return dict(series=series, name=name, uid="/".join([series, name]))


def purify_weight(src: str, dst: str = None) -> str:
    with smart_open(src, "rb") as f:
        state_dict = mge.load(f)
    if "model" in state_dict:
        state_dict = state_dict["model"]
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    if dst is None:
        dst = "_dump".join(os.path.splitext(src))
    with smart_open(dst, "wb") as f:
        mge.save(state_dict, f, pickle_protocol=pickle.DEFAULT_PROTOCOL)
    return dst
