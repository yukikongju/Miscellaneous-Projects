import torch
import pytest

from models.unet.unet_parts import DoubleConv, Down, Up


"""
- At each downsampling step we double the number of feature
  channels

"""


@pytest.fixture
def example_image():
    return torch.randint(1, 1000, size=(1, 3, 572, 572)).float()


def test_double_conv(example_image):
    OUTCHANNELS = 64
    double_conv = DoubleConv(in_channels=3, out_channels=OUTCHANNELS)
    output = double_conv(example_image)
    assert output.size()[1] == OUTCHANNELS
    assert output.size()[2] == example_image.size()[2]
    assert output.size()[3] == example_image.size()[3]


def test_down(example_image):
    B, C, H, W = example_image.size()
    down = Down(C, C * 2)
    output = down(example_image)
    assert output.size()[0] == B
    assert output.size()[1] == C * 2
    assert output.size()[2] == H // 2
    assert output.size()[3] == W // 2
