import megengine as mge
import megengine.module as M
import pytest

from basecls.models.snet import SNV2Block, SNV2XceptionBlock


@pytest.mark.parametrize("w_in", [32, 48])
@pytest.mark.parametrize("w_out", [64])
@pytest.mark.parametrize("w_mid", [32, 24])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("kernel", [3, 5])
@pytest.mark.parametrize("se_r", [0.0, 0.25])
@pytest.mark.parametrize("drop_path_prob", [0.0, 0.1])
@pytest.mark.parametrize("norm_name", ["BN"])
@pytest.mark.parametrize("act_name", ["relu"])
def test_block(
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
):
    m = SNV2Block(
        w_in,
        w_out,
        w_mid,
        kernel=kernel,
        stride=stride,
        norm_name=norm_name,
        act_name=act_name,
        se_r=se_r,
        drop_path_prob=drop_path_prob,
    )
    assert isinstance(m, M.Module)

    m(mge.random.normal(size=(2, w_in * 2 // stride, 8, 8)))


@pytest.mark.parametrize("w_in", [32])
@pytest.mark.parametrize("w_out", [64])
@pytest.mark.parametrize("w_mid", [32])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("kernel", [7, "x"])
@pytest.mark.parametrize("se_r", [0.25])
@pytest.mark.parametrize("drop_path_prob", [0.1])
@pytest.mark.parametrize("norm_name", ["BN"])
@pytest.mark.parametrize("act_name", ["relu"])
def test_x_block(
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
):
    m = SNV2XceptionBlock(
        w_in,
        w_out,
        w_mid,
        kernel=kernel,
        stride=stride,
        norm_name=norm_name,
        act_name=act_name,
        se_r=se_r,
        drop_path_prob=drop_path_prob,
    )
    assert isinstance(m, M.Module)

    m(mge.random.normal(size=(2, w_in * 2 // stride, 8, 8)))
