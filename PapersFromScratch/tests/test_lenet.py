import pytest
import torch

from models.lenet.lenet import LeNet, Block


@pytest.fixture
def tensor() -> torch.Tensor:
    return torch.randint(1, 255, size=(5, 1, 32, 32)).float()
    #  return torch.randint(1, 255, size=(5, 1, 64, 64)).float()


def test_block(tensor: torch.Tensor):
    B, C, H, W = tensor.size()
    OUTPUT_CHANNELS = 6
    block = Block(C, OUTPUT_CHANNELS)
    output = block(tensor)
    kernel_size = block.layer[0].kernel_size[0]
    stride = block.layer[0].stride[0]
    padding = block.layer[0].padding[0]
    dilation = block.layer[0].dilation[0]

    H_conv = (H + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    W_conv = (W + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    H_OUTPUT = H_conv // 2
    W_OUTPUT = W_conv // 2

    #  assert output.size() == (B, OUTPUT_CHANNELS, 14, 14)
    assert output.size() == (B, OUTPUT_CHANNELS, H_OUTPUT, W_OUTPUT)


def test_lenet(tensor):
    B, C, H, W = tensor.size()
    conv_features = [6, 16]
    fc_features = [400, 120, 84]

    NUM_CLASSES = 10
    lenet = LeNet(C, NUM_CLASSES, conv_features, fc_features)
    output = lenet(tensor)
    assert output.size() == (B, NUM_CLASSES)
