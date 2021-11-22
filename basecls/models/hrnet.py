#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
"""HRNet Series

HRNet: `"Deep High-Resolution Representation Learning for Visual Recognition"
<https://arxiv.org/abs/1908.07919>`_
"""
from collections import OrderedDict
from functools import partial
from typing import Any, List, Mapping, Optional

import megengine.functional as F
import megengine.module as M

from basecls.layers import build_head, conv2d, init_weights, norm2d
from basecls.layers.activations import activation
from basecls.utils import recursive_update, registers

from .resnet import ResBasicBlock, ResBottleneckBlock

__all__ = [
    "UpsampleNearest",
    "HRFusion",
    "HRModule",
    "HRTrans",
    "HRStage",
    "HRMerge",
    "HRNet",
]


class UpsampleNearest(M.Module):
    """Nearest upsample block

    Args:
        scale_factor: Upsample scale factor.
    """

    def __init__(self, scale_factor: int):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.repeat(F.repeat(x, self.scale_factor, axis=2), self.scale_factor, axis=3)


block_dict = {
    "basic": (partial(ResBasicBlock, stride=1, bot_mul=1, se_r=0, avg_down=False), 1),
    "bottleneck": (
        lambda w_out, **kwargs: ResBottleneckBlock(
            w_out=w_out * 4, stride=1, bot_mul=0.25, group_w=w_out, se_r=0, avg_down=False, **kwargs
        ),
        4,
    ),
}


class HRFusion(M.Module):
    """HRNet fusion block.

    Args:
        channels: Fusion channels.
        multi_scale_output: Whether output multi-scale features.
        norm_name: Normalization layer.
        act_name: Activation function.
    """

    def __init__(
        self,
        channels: List[int],
        multi_scale_output: bool,
        norm_name: str,
        act_name: str,
    ):
        super().__init__()
        self.multi_scale_output = multi_scale_output
        num_branches = len(channels)
        for f_o in range(num_branches if multi_scale_output else 1):
            for f_i in range(num_branches):
                fuse_layer = None
                if f_i < f_o:
                    fuse_layer = []
                    for i in range(f_o - f_i):
                        w_out = channels[f_i if i < f_o - f_i - 1 else f_o]
                        fuse_layer.extend(
                            [
                                (
                                    f"conv_{i + 1}",
                                    conv2d(w_in=channels[f_i], w_out=w_out, k=3, stride=2),
                                ),
                                (f"norm_{i + 1}", norm2d(norm_name, w_out)),
                            ]
                        )
                        if i < f_o - f_i - 1:
                            fuse_layer.append((f"act_{i + 1}", activation(act_name)))
                    fuse_layer = M.Sequential(OrderedDict(fuse_layer))
                elif f_i > f_o:
                    fuse_layer = M.Sequential(
                        OrderedDict(
                            [
                                ("conv_1", conv2d(w_in=channels[f_i], w_out=channels[f_o], k=1)),
                                ("norm_1", norm2d(norm_name, channels[f_o])),
                                ("upsample_1", UpsampleNearest(scale_factor=2 ** (f_i - f_o))),
                            ]
                        )
                    )
                setattr(self, f"fuse_{f_i + 1}_{f_o + 1}", fuse_layer)
        self.act = activation(act_name)

    def forward(self, x_list):
        x_fuse = []
        for f_o in range(len(x_list) if self.multi_scale_output else 1):
            x_sum = None
            for f_i, x in enumerate(x_list):
                fuse_layer = getattr(self, f"fuse_{f_i + 1}_{f_o + 1}", None)
                if fuse_layer:
                    x = fuse_layer(x)
                x_sum = x if x_sum is None else x_sum + x
            x_sum = self.act(x_sum)
            x_fuse.append(x_sum)
        return x_fuse


class HRModule(M.Module):
    """HRNet module.

    Args:
        block_name: Branch block type.
        num_blocks: Number of blocks.
        in_channels: Input channels.
        channels: Output channels.
        multi_scale_output: Whether output multi-scale features.
        norm_name: Normalization layer.
        act_name: Activation function.
    """

    def __init__(
        self,
        block_name: str,
        num_blocks: List[int],
        in_channels: List[int],
        channels: List[int],
        multi_scale_output: bool,
        norm_name: str,
        act_name: str,
    ):
        super().__init__()
        for i, (w_in, w_out, num_block) in enumerate(zip(in_channels, channels, num_blocks)):
            branch = self._make_branch(
                w_in=w_in,
                w_out=w_out,
                block_name=block_name,
                num_block=num_block,
                norm_name=norm_name,
                act_name=act_name,
            )
            setattr(self, f"branch{i + 1}", branch)
        self.fusion = None
        if len(channels) > 1:
            _, out_mul = block_dict[block_name]
            fusion_channels = [out_mul * c for c in channels]
            self.fusion = HRFusion(
                channels=fusion_channels,
                multi_scale_output=multi_scale_output,
                norm_name=norm_name,
                act_name=act_name,
            )

    def _make_branch(
        self,
        w_in: int,
        w_out: int,
        block_name: str,
        num_block: int,
        norm_name: str,
        act_name: str,
    ):
        block_fn, out_mul = block_dict[block_name]
        return M.Sequential(
            OrderedDict(
                [
                    (
                        f"block{i + 1}",
                        block_fn(
                            w_in=w_out * out_mul if i else w_in,
                            w_out=w_out,
                            norm_name=norm_name,
                            act_name=act_name,
                        ),
                    )
                    for i in range(num_block)
                ]
            )
        )

    def forward(self, x_list):
        x_list = [getattr(self, f"branch{i + 1}")(x) for i, x in enumerate(x_list)]
        if self.fusion:
            x_list = self.fusion(x_list)
        return x_list


class HRTrans(M.Module):
    """HRNet transition block.

    Args:
        in_chs: Input channels.
        out_chs: Output channels.
        norm_name: Normalization layer.
        act_name: Activation function.
    """

    def __init__(
        self,
        in_chs: List[int],
        out_chs: List[int],
        norm_name: str,
        act_name: str,
    ):
        super().__init__()
        n_in, n_out = len(in_chs), len(out_chs)
        self.num_trans = n_out
        for t_o in range(n_out):
            if t_o < n_in:
                trans_layer = (
                    M.Sequential(
                        OrderedDict(
                            [
                                ("conv_1", conv2d(w_in=in_chs[t_o], w_out=out_chs[t_o], k=3)),
                                ("norm_1", norm2d(norm_name, out_chs[t_o])),
                                ("act_1", activation(act_name)),
                            ]
                        )
                    )
                    if in_chs[t_o] != out_chs[t_o]
                    else None
                )
            else:
                trans_layer = []
                for i in range(t_o - n_in + 1):
                    w_out = out_chs[t_o] if i == t_o - n_in else in_chs[-1]
                    trans_layer.extend(
                        [
                            (f"conv_{i + 1}", conv2d(w_in=in_chs[-1], w_out=w_out, k=3, stride=2)),
                            (f"norm_{i + 1}", norm2d(norm_name, w_out)),
                            (f"act_{i + 1}", activation(act_name)),
                        ]
                    )
                trans_layer = M.Sequential(OrderedDict(trans_layer))
            setattr(self, f"trans_{t_o + 1}", trans_layer)

    def forward(self, x_list):
        x_trans = []
        x_list = x_list + [x_list[-1]] * (self.num_trans - len(x_list))
        for t_o, x in enumerate(x_list):
            trans_layer = getattr(self, f"trans_{t_o + 1}", None)
            if trans_layer:
                x = trans_layer(x)
            x_trans.append(x)
        return x_trans


class HRStage(M.Module):
    """HRNet stage.

    Args:
        num_modules: Number of modules.
        num_blocks: Number of blocks for each module.
        block_name: Branch block type.
        pre_channels: Channels of previous stage (an empty list for the first stage).
        cur_channels: Channels of current stage.
        multi_scale_output: Whether output multi-scale features.
        w_fst: Width of stem for the first stage (``None`` for other stages).
        norm_name: Normalization layer.
        act_name: Activation function.
    """

    def __init__(
        self,
        num_modules: int,
        num_blocks: List[int],
        block_name: str,
        pre_channels: List[int],
        cur_channels: List[int],
        multi_scale_output: bool,
        w_fst: Optional[int],
        norm_name: str,
        act_name: str,
    ):
        super().__init__()
        self.transition = None
        _, out_mul = block_dict[block_name]
        mid_channels = [out_mul * c for c in cur_channels]
        if w_fst:
            fst_channels = [w_fst]
        else:
            self.transition = HRTrans(
                in_chs=pre_channels,
                out_chs=mid_channels,
                norm_name=norm_name,
                act_name=act_name,
            )
        self.num_modules = num_modules
        for i in range(num_modules):
            module = HRModule(
                block_name=block_name,
                num_blocks=num_blocks,
                in_channels=fst_channels if w_fst and i == 0 else mid_channels,
                channels=cur_channels,
                multi_scale_output=multi_scale_output or i < num_modules - 1,
                norm_name=norm_name,
                act_name=act_name,
            )
            setattr(self, f"module{i + 1}", module)

    def forward(self, x_list):
        if self.transition:
            x_list = self.transition(x_list)
        for i in range(self.num_modules):
            module = getattr(self, f"module{i + 1}")
            x_list = module(x_list)
        return x_list


class HRMerge(M.Module):
    """HRNet merge block.

    Args:
        block_name: Head block type.
        pre_channels: Channels of the last stage.
        channels: Channels of each scale to merge.
        norm_name: Normalization layer.
        act_name: Activation function.
    """

    def __init__(
        self,
        block_name: str,
        pre_channels: List[int],
        channels: List[int],
        norm_name: str,
        act_name: str,
    ):
        super().__init__()
        block_fn, out_mul = block_dict[block_name]
        for i, (w_in, w_out) in enumerate(zip(pre_channels, channels)):
            branch = block_fn(
                w_in=w_in,
                w_out=w_out,
                norm_name=norm_name,
                act_name=act_name,
            )
            setattr(self, f"branch{i + 1}", branch)
        _, out_mul = block_dict[block_name]
        for i, (w_in, w_out) in enumerate(zip(channels[:-1], channels[1:])):
            dnsample = M.Sequential(
                OrderedDict(
                    [
                        (
                            "conv",
                            conv2d(
                                w_in=w_in * out_mul,
                                w_out=w_out * out_mul,
                                k=3,
                                stride=2,
                                bias=True,
                            ),
                        ),
                        ("norm", norm2d(norm_name, w_out * out_mul)),
                        ("act", activation(act_name)),
                    ]
                )
            )
            setattr(self, f"dnsample{i + 1}", dnsample)

    def forward(self, x_list):
        x = getattr(self, "branch1")(x_list[0])
        for i in range(len(x_list) - 1):
            dnsample = getattr(self, f"dnsample{i + 1}")
            branch = getattr(self, f"branch{i + 2}")
            x = dnsample(x) + branch(x_list[i + 1])
        return x


@registers.models.register()
class HRNet(M.Module):
    """HRNet model.

    Args:
        stage_modules: Number of modules for each stage.
        stage_blocks: Number of blocks for each module in stages.
        stage_block_names: Branch block types for each stage.
        stage_channels: Number of channels for each stage.
        w_stem: Stem width. Default: ``64``
        multi_scale_output: Whether output multi-scale features. Default: ``True``
        merge_block_name: Merge block type. Default: ``"bottleneck"``
        merge_channels: Channels of each scale in merge block. Default: ``[32, 64, 128, 256]``
        norm_name: Normalization layer. Default: ``"BN"``
        act_name: Activation function. Default: ``"relu"``
        head: head args. Default: ``None``
    """

    def __init__(
        self,
        stage_modules: List[int],
        stage_blocks: List[List[int]],
        stage_block_names: List[str],
        stage_channels: List[List[int]],
        w_stem: int = 64,
        multi_scale_output: bool = True,
        merge_block_name: str = "bottleneck",
        merge_channels: List[int] = [32, 64, 128, 256],
        norm_name: str = "BN",
        act_name: str = "relu",
        head: Mapping[str, Any] = None,
        **kwargs,
    ):
        super().__init__()
        self.stem = M.Sequential(
            OrderedDict(
                [
                    ("conv_1", conv2d(w_in=3, w_out=w_stem, k=3, stride=2)),
                    ("norm_1", norm2d(norm_name, w_in=w_stem)),
                    ("act_1", activation(act_name)),
                    ("conv_2", conv2d(w_in=64, w_out=w_stem, k=3, stride=2)),
                    ("norm_2", norm2d(norm_name, w_in=w_stem)),
                    ("act_2", activation(act_name)),
                ]
            )
        )

        self.num_stages = len(stage_modules)
        pre_channels = []
        for i in range(self.num_stages):
            stage = HRStage(
                num_modules=stage_modules[i],
                num_blocks=stage_blocks[i],
                block_name=stage_block_names[i],
                pre_channels=pre_channels,
                cur_channels=stage_channels[i],
                multi_scale_output=multi_scale_output,
                w_fst=None if i else w_stem,
                norm_name=norm_name,
                act_name=act_name,
            )
            setattr(self, f"stage{i + 1}", stage)
            _, out_mul = block_dict[stage_block_names[i]]
            pre_channels = [out_mul * c for c in stage_channels[i]]
        self.merge = HRMerge(
            block_name=merge_block_name,
            pre_channels=pre_channels,
            channels=merge_channels,
            norm_name=norm_name,
            act_name=act_name,
        )
        w_merge = merge_channels[-1] * block_dict[merge_block_name][1]
        self.head = build_head(w_merge, head, norm_name, act_name)
        self.apply(init_weights)

    def forward(self, x):
        x = self.stem(x)
        x_list = [x]
        for i in range(self.num_stages):
            stage = getattr(self, f"stage{i + 1}")
            x_list = stage(x_list)
        x = self.merge(x_list)
        if getattr(self, "head", None) is not None:
            x = self.head(x)
        return x


def _build_hrnet(**kwargs):
    model_args = dict(
        stage_modules=[1, 1, 4, 3],
        stage_blocks=[[4], [4, 4], [4, 4, 4], [4, 4, 4, 4]],
        stage_block_names=["bottleneck", "basic", "basic", "basic"],
        head=dict(name="ClsHead", width=2048),
    )
    recursive_update(model_args, kwargs)
    return HRNet(**model_args)


@registers.models.register()
def hrnet_w18_small_v1(**kwargs):
    model_args = dict(
        stage_modules=[1, 1, 1, 1],
        stage_blocks=[[1], [2, 2], [2, 2, 2], [2, 2, 2, 2]],
        stage_channels=[[32], [16, 32], [16, 32, 64], [16, 32, 64, 128]],
    )
    recursive_update(model_args, kwargs)
    return _build_hrnet(**model_args)


@registers.models.register()
def hrnet_w18_small_v2(**kwargs):
    model_args = dict(
        stage_modules=[1, 1, 3, 2],
        stage_blocks=[[2], [2, 2], [2, 2, 2], [2, 2, 2, 2]],
        stage_channels=[[64], [18, 36], [18, 36, 72], [18, 36, 72, 144]],
    )
    recursive_update(model_args, kwargs)
    return _build_hrnet(**model_args)


@registers.models.register()
def hrnet_w18(**kwargs):
    model_args = dict(
        stage_channels=[[64], [18, 36], [18, 36, 72], [18, 36, 72, 144]],
    )
    recursive_update(model_args, kwargs)
    return _build_hrnet(**model_args)


@registers.models.register()
def hrnet_w30(**kwargs):
    model_args = dict(
        stage_channels=[[64], [30, 60], [30, 60, 120], [30, 60, 120, 240]],
    )
    recursive_update(model_args, kwargs)
    return _build_hrnet(**model_args)


@registers.models.register()
def hrnet_w32(**kwargs):
    model_args = dict(
        stage_channels=[[64], [32, 64], [32, 64, 128], [32, 64, 128, 256]],
    )
    recursive_update(model_args, kwargs)
    return _build_hrnet(**model_args)


@registers.models.register()
def hrnet_w40(**kwargs):
    model_args = dict(
        stage_channels=[[64], [40, 80], [40, 80, 160], [40, 80, 160, 320]],
    )
    recursive_update(model_args, kwargs)
    return _build_hrnet(**model_args)


@registers.models.register()
def hrnet_w44(**kwargs):
    model_args = dict(
        stage_channels=[[64], [44, 88], [44, 88, 176], [44, 88, 176, 352]],
    )
    recursive_update(model_args, kwargs)
    return _build_hrnet(**model_args)


@registers.models.register()
def hrnet_w48(**kwargs):
    model_args = dict(
        stage_channels=[[64], [48, 96], [48, 96, 192], [48, 96, 192, 384]],
    )
    recursive_update(model_args, kwargs)
    return _build_hrnet(**model_args)


@registers.models.register()
def hrnet_w64(**kwargs):
    model_args = dict(
        stage_channels=[[64], [64, 128], [64, 128, 256], [64, 128, 256, 512]],
    )
    recursive_update(model_args, kwargs)
    return _build_hrnet(**model_args)
