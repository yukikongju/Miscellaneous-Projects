import torch
import pytest

from models.unet.unet_parts import DoubleConv


@pytest.fixture
def example_tensor():
    return torch.randint(1, 1000, size=(1, 3, 572, 572)).float()


def test_double_conv(example_tensor):
    double_conv = DoubleConv(in_channels=3, out_channels=64)
    print(double_conv(example_tensor))
