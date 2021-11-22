#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import megengine as mge
import megengine.module as M
import pytest

from basecls.models.vit import FFN, Attention, EncoderBlock, PatchEmbed


@pytest.mark.parametrize("img_size", [16])
@pytest.mark.parametrize("patch_size", [1, 2, 3])
@pytest.mark.parametrize("in_chans", [3])
@pytest.mark.parametrize("embed_dim", [8])
@pytest.mark.parametrize("flatten,norm_name", [(True, M.LayerNorm), (True, "LN"), (False, None)])
def test_patchembed(img_size, patch_size, in_chans, embed_dim, flatten, norm_name):
    m = PatchEmbed(
        img_size,
        patch_size,
        in_chans,
        embed_dim,
        flatten,
        norm_name,
    )
    assert isinstance(m, M.Module)

    m(mge.random.normal(size=(2, 3, 16, 16)))


@pytest.mark.parametrize("dim", [8])
@pytest.mark.parametrize("num_heads", [2])
@pytest.mark.parametrize("qkv_bias", [True, False])
@pytest.mark.parametrize("attn_drop", [0, 0.05])
@pytest.mark.parametrize("proj_drop", [0, 0.1])
def test_attention(dim, num_heads, qkv_bias, attn_drop, proj_drop):
    m = Attention(
        dim=dim,
        num_heads=num_heads,
        qkv_bias=qkv_bias,
        attn_drop=attn_drop,
        proj_drop=proj_drop,
    )
    assert isinstance(m, M.Module)

    m(mge.random.normal(size=(2, 16, 8)))


@pytest.mark.parametrize("in_features", [4])
@pytest.mark.parametrize("hidden_features", [None, 32])
@pytest.mark.parametrize("out_features", [None, 8])
@pytest.mark.parametrize("act_name", ["gelu", "relu"])
@pytest.mark.parametrize("drop", [0, 0.05])
def test_ffn(in_features, hidden_features, out_features, act_name, drop):
    m = FFN(
        in_features=in_features,
        hidden_features=hidden_features,
        out_features=out_features,
        act_name=act_name,
        drop=drop,
    )
    assert isinstance(m, M.Module)

    m(mge.random.normal(size=(2, 16, 4)))


@pytest.mark.parametrize("dim", [8])
@pytest.mark.parametrize("num_heads", [2])
@pytest.mark.parametrize("ffn_ratio", [0.3, 4])
@pytest.mark.parametrize("qkv_bias", [True, False])
@pytest.mark.parametrize("attn_drop", [0, 0.05])
@pytest.mark.parametrize("drop", [0, 0.1])
@pytest.mark.parametrize("drop_path", [0, 0.2])
@pytest.mark.parametrize("norm_name", ["LN", M.LayerNorm])
@pytest.mark.parametrize("act_name", ["gelu", "relu"])
def test_encoderblock(
    dim,
    num_heads,
    ffn_ratio,
    qkv_bias,
    attn_drop,
    drop,
    drop_path,
    norm_name,
    act_name,
):
    m = EncoderBlock(
        dim=dim,
        num_heads=num_heads,
        ffn_ratio=ffn_ratio,
        qkv_bias=qkv_bias,
        attn_drop=attn_drop,
        drop=drop,
        drop_path=drop_path,
        norm_name=norm_name,
        act_name=act_name,
    )
    assert isinstance(m, M.Module)

    m(mge.random.normal(size=(2, 16, 8)))
