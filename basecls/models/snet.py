#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
"""Shufflenet Series

ShufflenetV2: `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
<https://arxiv.org/abs/1807.11164>`_

References:
    https://github.com/megvii-model/ShuffleNet-Series/blob/master/ShuffleNetV2/network.py
"""
from typing import Any, Callable, Mapping, Sequence

import megengine.functional as F
import megengine.hub as hub
import megengine.module as M

from basecls.layers import SE, DropPath, activation, conv2d, norm2d
from basecls.layers.heads import build_head
from basecls.models.resnet import SimpleStem
from basecls.utils import recursive_update, registers

__all__ = ["SNV2Block", "SNV2XceptionBlock", "SNetV2"]


def channel_shuffle(x):
    batchsize, num_channels, height, width = x.shape
    assert num_channels % 4 == 0
    x = x.reshape(batchsize * num_channels // 2, 2, height * width)
    x = x.transpose(1, 0, 2)
    x = x.reshape(2, -1, num_channels // 2, height, width)
    return x[0], x[1]


class SNV2Block(M.Module):
    """ShuffleNet V2 Block"""

    def __init__(
        self,
        w_in: int,
        w_out: int,
        w_mid: int,
        *,
        kernel: int,
        stride: int,
        norm_name: str,
        act_name: str,
        se_r: float,
        drop_path_prob: float,
        **kwargs,
    ):
        super().__init__()
        assert not kwargs

        self.stride = stride

        w_main_out = w_out - w_in
        w_se = int(w_main_out * se_r)

        self.branch_main = M.Sequential(
            # pw1
            conv2d(w_in, w_mid, 1),
            norm2d(norm_name, w_mid),
            activation(act_name),
            # dw
            conv2d(w_mid, w_mid, kernel, stride=stride, groups=w_mid),
            norm2d(norm_name, w_mid),
            # pw2
            conv2d(w_mid, w_main_out, 1),
            norm2d(norm_name, w_main_out),
            activation(act_name),
        )
        # NOTE: se mid_act is RELU
        self.se = SE(w_main_out, w_se, act_name) if w_se else None
        self.drop_path = DropPath(drop_path_prob) if drop_path_prob > 0 else None

        self.branch_proj = (
            M.Sequential(
                # dw
                conv2d(w_in, w_in, kernel, stride=stride, groups=w_in),
                norm2d(norm_name, w_in),
                # pw
                conv2d(w_in, w_in, 1),
                norm2d(norm_name, w_in),
                activation(act_name),
            )
            if stride == 2
            else None
        )

    def forward(self, old_x):
        if self.stride == 1:
            x_proj, x = channel_shuffle(old_x)

        elif self.stride == 2:
            x_proj = x = old_x

        if self.branch_proj:
            x_proj = self.branch_proj(x_proj)

        x = self.branch_main(x)
        if self.se:
            x = self.se(x)
        if self.drop_path:
            x = self.drop_path(x)

        return F.concat((x_proj, x), 1)


class SNV2XceptionBlock(SNV2Block):
    """ShuffleNet V2 Xception stype block used in ShuffleNet V2+"""

    def __init__(
        self,
        w_in: int,
        w_out: int,
        w_mid: int,
        *,
        kernel: int,
        stride: int,
        norm_name: str,
        act_name: str,
        se_r: float,
        drop_path_prob: float,
        **kwargs,
    ):
        if isinstance(kernel, int):
            super().__init__(
                w_in,
                w_out,
                w_mid,
                kernel=kernel,
                stride=stride,
                norm_name=norm_name,
                act_name=act_name,
                se_r=se_r,
                drop_path_prob=drop_path_prob,
                **kwargs,
            )
        elif kernel == "x":
            super().__init__(
                w_in,
                w_out,
                w_mid,
                kernel=3,
                stride=stride,
                norm_name=norm_name,
                act_name=act_name,
                se_r=se_r,
                drop_path_prob=drop_path_prob,
                **kwargs,
            )
            w_main_out = w_out - w_in
            self.branch_main = M.Sequential(
                # dw
                conv2d(w_in, w_in, 3, stride=stride, groups=w_in),
                norm2d(norm_name, w_in),
                # pw
                conv2d(w_in, w_mid, 1),
                norm2d(norm_name, w_mid),
                activation(act_name),
                # dw
                conv2d(w_mid, w_mid, 3, stride=1, groups=w_mid),
                norm2d(norm_name, w_mid),
                # pw
                conv2d(w_mid, w_mid, 1),
                norm2d(norm_name, w_mid),
                activation(act_name),
                # dw
                conv2d(w_mid, w_mid, 3, stride=1, groups=w_mid),
                norm2d(norm_name, w_mid),
                # pw
                conv2d(w_mid, w_main_out, 1),
                norm2d(norm_name, w_main_out),
                activation(act_name),
            )
        else:
            raise ValueError(f"unidentified kernel={kernel}, should be (3, 5, 7, 'x')")


@registers.models.register()
class SNetV2(M.Module):
    """ShufflenetV2 model.

    Args:
        block: building block to use, ``SNV2XceptionBlock`` for v2+.
        stem_w: width for stem layer.
        depths: depth for each stage (number of blocks in the stage).
        widths: width for each stage (width of each block in the stage).
        strides: strides for each stage (applies to the first block of each stage).
        kernels: kernel sizes for each stage.
        use_maxpool: whether use maxpool stride 2 after stem. Default: ``True``
        se_r: Squeeze-and-Excitation (SE) ratio. Default: ``0.0``
        drop_path_prob: drop path probability. Default: ``0.0``
        norm_name: normalization function. Default: ``"BN"``
        act_name: activation function. Default: ``"relu6"``
        head: head args. Default: ``None``
    """

    def __init__(
        self,
        block: Callable,
        stem_w: int,
        depths: Sequence[int],
        widths: Sequence[int],
        strides: Sequence[int],
        kernels: Sequence[int],
        use_maxpool: bool = True,
        se_r: float = 0.0,
        drop_path_prob: float = 0.0,
        norm_name: str = "BN",
        act_name: str = "relu",
        head: Mapping[str, Any] = None,
    ):
        super().__init__()

        self.stem = SimpleStem(
            3,
            stem_w,
            norm_name,
            act_name,
        )
        self.maxpool = M.MaxPool2d(3, 2, 1) if use_maxpool else None

        prev_w = stem_w

        # stochastic depth
        iblock = 0
        total_blocks = sum(depths)

        self.num_stages = len(depths)

        for idxstage in range(self.num_stages):
            blocks = []
            depth = depths[idxstage]
            width = widths[idxstage]
            stride = strides[idxstage]
            kernel = kernels[idxstage]
            if isinstance(kernel, int):
                kernel = [kernel] * depth

            for i in range(depth):
                iblock += 1
                blocks.append(
                    block(
                        prev_w // 2 if i else prev_w,
                        width,
                        width // 2,
                        kernel=kernel[i],
                        stride=1 if i else stride,
                        norm_name=norm_name,
                        act_name=act_name if idxstage >= 1 else "relu",
                        se_r=se_r if idxstage >= 2 else 0.0,
                        drop_path_prob=drop_path_prob * iblock / total_blocks,
                    )
                )
                prev_w = width
            setattr(self, f"stage{idxstage}", M.Sequential(*blocks))

        self.head = build_head(
            prev_w,
            head_args=head,
            norm_name=norm_name,
            act_name=act_name,
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, M.Conv2d):
                if "stem" in name or "head.se" in name:
                    M.init.normal_(m.weight, 0, 0.01)
                else:
                    M.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    M.init.fill_(m.bias, 0)
            elif isinstance(m, M.batchnorm._BatchNorm):
                M.init.fill_(m.weight, 1)
                if m.bias is not None:
                    M.init.fill_(m.bias, 0.0001)
                M.init.fill_(m.running_mean, 0)
            elif isinstance(m, M.Linear):
                M.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    M.init.fill_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        if self.maxpool:
            x = self.maxpool(x)

        for i in range(self.num_stages):
            stage = getattr(self, f"stage{i}")
            x = stage(x)

        x = self.head(x)
        return x


def _build_snetv2(**kwargs):
    model_args = dict(
        block=SNV2Block,
        stem_w=24,
        depths=[4, 8, 4],
        strides=[2, 2, 2],
        kernels=[3, 3, 3],
        head=dict(name="ClsHead", width=1024, bias=False),
    )
    recursive_update(model_args, kwargs)
    return SNetV2(**model_args)


def _build_snetv2_plus(**kwargs):
    model_args = dict(
        block=SNV2XceptionBlock,
        stem_w=16,
        use_maxpool=False,
        depths=[4, 4, 8, 4],
        strides=[2, 2, 2, 2],
        kernels=[
            [3, 3, "x", 5],
            [5, 5, 3, 3],
            [7, 3, 7, 5, 5, 3, 7, 3],
            [7, 5, "x", 7],
        ],
        se_r=0.25,
        act_name="hswish",
        head=dict(
            name="MBV3Head",
            w_h=1280,
            width=1280,
            se_r=0.25,
            dropout_prob=0.2,
            bias=False,
        ),
    )
    recursive_update(model_args, kwargs)
    return SNetV2(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/snet/snetv2_x050/snetv2_x050.pkl"
)
def snetv2_x050(**kwargs):
    model_args = dict(widths=[48, 96, 192])
    recursive_update(model_args, kwargs)
    return _build_snetv2(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/snet/snetv2_x100/snetv2_x100.pkl"
)
def snetv2_x100(**kwargs):
    model_args = dict(widths=[116, 232, 464])
    recursive_update(model_args, kwargs)
    return _build_snetv2(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/snet/snetv2_x150/snetv2_x150.pkl"
)
def snetv2_x150(**kwargs):
    model_args = dict(widths=[176, 352, 704])
    recursive_update(model_args, kwargs)
    return _build_snetv2(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/snet/snetv2_x200/snetv2_x200.pkl"
)
def snetv2_x200(**kwargs):
    model_args = dict(widths=[244, 488, 976], head=dict(dropout_prob=0.2, width=2048))
    recursive_update(model_args, kwargs)
    return _build_snetv2(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/snet/snetv2p_x075/snetv2p_x075.pkl"
)
def snetv2p_x075(**kwargs):
    model_args = dict(widths=[36, 104, 208, 416])
    recursive_update(model_args, kwargs)
    return _build_snetv2_plus(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/snet/snetv2p_x100/snetv2p_x100.pkl"
)
def snetv2p_x100(**kwargs):
    model_args = dict(widths=[48, 128, 256, 512])
    recursive_update(model_args, kwargs)
    return _build_snetv2_plus(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/snet/snetv2p_x125/snetv2p_x125.pkl"
)
def snetv2p_x125(**kwargs):
    model_args = dict(widths=[68, 168, 336, 672])
    recursive_update(model_args, kwargs)
    return _build_snetv2_plus(**model_args)
