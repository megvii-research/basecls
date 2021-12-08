#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
"""RepVGG Series

RegVGG: `"RepVGG: Making VGG-style ConvNets Great Again" <https://arxiv.org/abs/2101.03697>`_
"""
from numbers import Real
from typing import Any, List, Mapping, Sequence, Union

import megengine
import megengine.functional as F
import megengine.hub as hub
import megengine.module as M
import numpy as np

from basecls.layers import SE, activation, init_weights
from basecls.layers.heads import build_head
from basecls.utils import recursive_update, registers

__all__ = ["RepVGGBlock", "RepVGG"]


class RepVGGBlock(M.Module):
    """RepVGG Reparamed Block.

    Args:
        w_in: number of input channels.
        w_out: number of output channels.
        stride: stride of the 2D conv. Default: ``1``
        groups: number of groups of the 2D conv. Default: ``1``
        se_r: Squeeze-and-Excitation (SE) ratio. Default: ``0.0``
        act_name: activation function. Default: ``"relu"``
        deploy: fuse branches into a plain ``Conv2d`` layer. Default: ``False``
    """

    def __init__(
        self,
        w_in: int,
        w_out: int,
        stride: int = 1,
        groups: int = 1,
        se_r: float = 0.0,
        act_name: str = "relu",
        deploy: bool = False,
    ):
        super().__init__()
        self.w_in = w_in
        self.w_out = w_out
        self.stride = stride
        self.groups = groups
        self.act_name = act_name
        self.se_r = se_r
        self.deploy = deploy

        self.nonlinearity = activation(act_name)

        self.se = SE(w_out, int(w_out * se_r), act_name) if se_r > 0 else M.Identity()

        if deploy:
            self.rbr_reparam = M.Conv2d(w_in, w_out, 3, stride, 1, groups=groups, bias=True)
        else:
            self.rbr_identity = M.BatchNorm2d(w_in) if w_out == w_in and stride == 1 else None
            self.rbr_dense = M.ConvBn2d(w_in, w_out, 3, stride, 1, groups=groups, bias=False)
            self.rbr_1x1 = M.ConvBn2d(w_in, w_out, 1, stride, 0, groups=groups, bias=False)

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

    def _get_equivalent_kernel_bias(self):
        """
        This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
        You can get the equivalent kernel and bias at any time and do whatever you want,
        for example, apply some penalties or constraints during training, just like you do
        to the other models. May be useful for quantization or pruning.
        """
        kernel_3x3, bias_3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel_1x1, bias_1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernel_id, bias_id = self._fuse_bn_tensor(self.rbr_identity)
        return (
            kernel_3x3 + self._pad_1x1_to_3x3_tensor(kernel_1x1) + kernel_id,
            bias_3x3 + bias_1x1 + bias_id,
        )

    def _pad_1x1_to_3x3_tensor(self, kernel_1x1):
        if kernel_1x1 is None:
            return 0
        else:
            kernel = F.zeros((*kernel_1x1.shape[:-2], 3, 3))
            kernel[..., 1:2, 1:2] = kernel_1x1
            return kernel  # torch.nn.functional.pad(kernel_1x1, [1,1,1,1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, M.ConvBn2d):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, M.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.w_in // self.groups
                kernel_value = np.zeros((self.w_in, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.w_in):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                if self.groups > 1:
                    kernel_value = kernel_value.reshape(self.groups, -1, *kernel_value.shape[-3:])
                self.id_tensor = megengine.tensor(kernel_value)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = F.sqrt(running_var + eps)
        t = (gamma / std).reshape(*kernel.shape[:-3], 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    @classmethod
    def convert_to_deploy(cls, module) -> M.Module:
        """Convert/fuse reparameterized :py:class:`RepvggBlock` for deploy"""
        module_output = module
        if isinstance(module, RepVGGBlock):
            if module.deploy:  # already in deploy
                return module_output
            module_output = RepVGGBlock(
                module.w_in,
                module.w_out,
                module.stride,
                module.groups,
                module.se_r,
                module.act_name,
                deploy=True,
            )
            kernel, bias = module._get_equivalent_kernel_bias()
            module_output.rbr_reparam.weight[:] = kernel
            module_output.rbr_reparam.bias[:] = bias
            if isinstance(module.se, SE):
                module_output.se.load_state_dict(module.se.state_dict())
        for name, child in module.named_children():
            setattr(module_output, name, cls.convert_to_deploy(child))
        del module
        return module_output


@registers.models.register()
class RepVGG(M.Module):
    """RepVGG Model.

    Use :py:meth:`RepVGG.convert_to_deploy` to convert a training :py:class:`RepVGG` to deploy:

    .. code::

        model = RepVGG(..., deploy=False)
        model.load_state_dict(...)
        _ = RepVGG.convert_to_deploy(model)

    Args:
        num_blocks: RepVGG depths.
        width_multiplier: RepVGG widths, ``base_width`` is ``[64, 128, 256, 512]``.
        head: head args. Default: ``None``
        groups: number of groups for blocks. Default: ``1``
        se_r: Squeeze-and-Excitation (SE) ratio. Default: ``0.0``
        act_name: activation function. Default: ``"relu"``
        deploy: switch a reparamed RepVGG into deploy mode. Default: ``False``
    """

    def __init__(
        self,
        num_blocks: Sequence[int],
        width_multiplier: Sequence[int],
        head: Mapping[str, Any] = None,
        groups: Union[int, List[Union[int, List[int]]]] = 1,
        se_r: float = 0.0,
        act_name: str = "relu",
        deploy: bool = False,
    ):
        super().__init__()

        assert len(width_multiplier) == 4

        self.deploy = deploy
        self.se_r = se_r

        if isinstance(groups, Real):
            groups = [groups] * len(num_blocks)

        self.w_in = min(64, int(64 * width_multiplier[0]))

        self.stage0 = RepVGGBlock(3, self.w_in, 2, 1, self.se_r, act_name, self.deploy)
        self.stage1 = self._make_stage(
            int(64 * width_multiplier[0]), num_blocks[0], 2, groups[0], act_name
        )
        self.stage2 = self._make_stage(
            int(128 * width_multiplier[1]), num_blocks[1], 2, groups[1], act_name
        )
        self.stage3 = self._make_stage(
            int(256 * width_multiplier[2]), num_blocks[2], 2, groups[2], act_name
        )
        self.stage4 = self._make_stage(
            int(512 * width_multiplier[3]), num_blocks[3], 2, groups[3], act_name
        )

        self.head = build_head(int(512 * width_multiplier[3]), head, act_name=act_name)

        self.apply(init_weights)

    def _make_stage(self, w_out, num_blocks, stride, groups, act_name):
        blocks = []
        if isinstance(groups, Real):
            groups = [groups] * num_blocks
        for i in range(num_blocks):
            blocks.append(
                RepVGGBlock(self.w_in, w_out, stride, groups[i], self.se_r, act_name, self.deploy)
            )
            stride = 1
            self.w_in = w_out
        return M.Sequential(*blocks)

    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        if self.head:
            out = self.head(out)
        return out

    @classmethod
    def convert_to_deploy(cls, module) -> M.Module:
        return RepVGGBlock.convert_to_deploy(module)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/repvgg/repvgg_a0/repvgg_a0.pkl"
)
def repvgg_a0(**kwargs):
    model_args = dict(
        num_blocks=[2, 4, 14, 1],
        width_multiplier=[0.75, 0.75, 0.75, 2.5],
        head=dict(name="ClsHead"),
    )
    recursive_update(model_args, kwargs)
    return RepVGG(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/repvgg/repvgg_a1/repvgg_a1.pkl"
)
def repvgg_a1(**kwargs):
    model_args = dict(
        num_blocks=[2, 4, 14, 1],
        width_multiplier=[1, 1, 1, 2.5],
        head=dict(name="ClsHead"),
    )
    recursive_update(model_args, kwargs)
    return RepVGG(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/repvgg/repvgg_a2/repvgg_a2.pkl"
)
def repvgg_a2(**kwargs):
    model_args = dict(
        num_blocks=[2, 4, 14, 1],
        width_multiplier=[1.5, 1.5, 1.5, 2.75],
        head=dict(name="ClsHead"),
    )
    recursive_update(model_args, kwargs)
    return RepVGG(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/repvgg/repvgg_b0/repvgg_b0.pkl"
)
def repvgg_b0(**kwargs):
    model_args = dict(
        num_blocks=[4, 6, 16, 1],
        width_multiplier=[1, 1, 1, 2.5],
        head=dict(name="ClsHead"),
    )
    recursive_update(model_args, kwargs)
    return RepVGG(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/repvgg/repvgg_b1/repvgg_b1.pkl"
)
def repvgg_b1(**kwargs):
    model_args = dict(
        num_blocks=[4, 6, 16, 1],
        width_multiplier=[2, 2, 2, 4],
        head=dict(name="ClsHead"),
    )
    recursive_update(model_args, kwargs)
    return RepVGG(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/repvgg/repvgg_b1g2/repvgg_b1g2.pkl"
)
def repvgg_b1g2(**kwargs):
    model_args = dict(
        num_blocks=[4, 6, 16, 1],
        width_multiplier=[2, 2, 2, 4],
        groups=[
            [1, 2, 1, 2],
            [1, 2, 1, 2, 1, 2],
            [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            [1],
        ],
        head=dict(name="ClsHead"),
    )
    recursive_update(model_args, kwargs)
    return RepVGG(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/repvgg/repvgg_b1g4/repvgg_b1g4.pkl"
)
def repvgg_b1g4(**kwargs):
    model_args = dict(
        num_blocks=[4, 6, 16, 1],
        width_multiplier=[2, 2, 2, 4],
        groups=[
            [1, 4, 1, 4],
            [1, 4, 1, 4, 1, 4],
            [1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4],
            [1],
        ],
        head=dict(name="ClsHead"),
    )
    recursive_update(model_args, kwargs)
    return RepVGG(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/repvgg/repvgg_b2/repvgg_b2.pkl"
)
def repvgg_b2(**kwargs):
    model_args = dict(
        num_blocks=[4, 6, 16, 1],
        width_multiplier=[2.5, 2.5, 2.5, 5],
        head=dict(name="ClsHead"),
    )
    recursive_update(model_args, kwargs)
    return RepVGG(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/repvgg/repvgg_b2g2/repvgg_b2g2.pkl"
)
def repvgg_b2g2(**kwargs):
    model_args = dict(
        num_blocks=[4, 6, 16, 1],
        width_multiplier=[2.5, 2.5, 2.5, 5],
        groups=[
            [1, 2, 1, 2],
            [1, 2, 1, 2, 1, 2],
            [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            [1],
        ],
        head=dict(name="ClsHead"),
    )
    recursive_update(model_args, kwargs)
    return RepVGG(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/repvgg/repvgg_b2g4/repvgg_b2g4.pkl"
)
def repvgg_b2g4(**kwargs):
    model_args = dict(
        num_blocks=[4, 6, 16, 1],
        width_multiplier=[2.5, 2.5, 2.5, 5],
        groups=[
            [1, 4, 1, 4],
            [1, 4, 1, 4, 1, 4],
            [1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4],
            [1],
        ],
        head=dict(name="ClsHead"),
    )
    recursive_update(model_args, kwargs)
    return RepVGG(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/repvgg/repvgg_b3/repvgg_b3.pkl"
)
def repvgg_b3(**kwargs):
    model_args = dict(
        num_blocks=[4, 6, 16, 1],
        width_multiplier=[3, 3, 3, 5],
        head=dict(name="ClsHead"),
    )
    recursive_update(model_args, kwargs)
    return RepVGG(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/repvgg/repvgg_b3g2/repvgg_b3g2.pkl"
)
def repvgg_b3g2(**kwargs):
    model_args = dict(
        num_blocks=[4, 6, 16, 1],
        width_multiplier=[3, 3, 3, 5],
        groups=[
            [1, 2, 1, 2],
            [1, 2, 1, 2, 1, 2],
            [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            [1],
        ],
        head=dict(name="ClsHead"),
    )
    recursive_update(model_args, kwargs)
    return RepVGG(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/repvgg/repvgg_b3g4/repvgg_b3g4.pkl"
)
def repvgg_b3g4(**kwargs):
    model_args = dict(
        num_blocks=[4, 6, 16, 1],
        width_multiplier=[3, 3, 3, 5],
        groups=[
            [1, 4, 1, 4],
            [1, 4, 1, 4, 1, 4],
            [1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4],
            [1],
        ],
        head=dict(name="ClsHead"),
    )
    recursive_update(model_args, kwargs)
    return RepVGG(**model_args)


@registers.models.register()
@hub.pretrained(
    "https://data.megengine.org.cn/research/basecls/models/repvgg/repvgg_d2/repvgg_d2.pkl"
)
def repvgg_d2(**kwargs):
    model_args = dict(
        num_blocks=[8, 14, 24, 1],
        width_multiplier=[2.5, 2.5, 2.5, 5],
        se_r=0.0625,
        head=dict(name="ClsHead"),
    )
    recursive_update(model_args, kwargs)
    return RepVGG(**model_args)
