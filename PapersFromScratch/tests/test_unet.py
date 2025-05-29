import torch
import pytest

from models.unet.unet_parts import DoubleConv, Down, Up
from models.unet.unet import UNet

#  from models.unet.unets_part2 import DoubleConv, Down, Up
#  from models.unet.unet2 import UNet


"""
- At each downsampling step we double the number of feature
  channels

"""


@pytest.fixture
def example_image():
    return torch.randint(1, 1000, size=(1, 3, 572, 572)).float()


@pytest.fixture
def example_up_tensor():
    #  return torch.randint(1, 1000, size=(5, 64, 1024, 1024)).float()
    return torch.randint(1, 1000, size=(1, 64, 128, 128)).float()
    #  return torch.randn(1, 64, 128, 128).float()


def test_double_conv(example_image):
    B, C, H, W = example_image.size()
    OUTCHANNELS = 64
    double_conv = DoubleConv(in_channels=3, out_channels=OUTCHANNELS)
    output = double_conv(example_image)
    assert output.size() == (B, OUTCHANNELS, H, W)


def test_down(example_image):
    B, C, H, W = example_image.size()
    down = Down(C, C * 2)
    output = down(example_image)
    assert output.size() == (B, C * 2, H // 2, W // 2)


def test_up_bilinear(example_up_tensor):  # FIXME
    """
    feature map and skip connection have the same shape (?)
    """
    B, C, H, W = example_up_tensor.size()
    skip = example_up_tensor.clone().detach()
    up = Up(W, W // 2, True)
    output = up(example_up_tensor, skip)
    assert output.size() == (B, C, H, W)


def test_up_transpose(example_up_tensor):
    """ """
    B, C, H, W = example_up_tensor.size()
    skip = torch.randint(1, 100, size=(B, C // 2, H * 2, W * 2)).float()
    up = Up(C, C // 2, False)
    output = up(example_up_tensor, skip)
    assert output.size() == (B, C // 2, H * 2, W * 2)


def test_unet_transpose(example_image):
    NUM_CLASSES = 10
    unet = UNet(n_channels=3, n_classes=NUM_CLASSES, bilinear=False)
    output = unet(example_image)
