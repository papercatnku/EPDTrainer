
'''
有改动的resnet版本,其中去掉了远跳连接后的relu
首个maxpooling使用卷积代替,卷积层用ConvBNReLU实现。
改动以利于QAT支持
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchvision

from torchvision.models.resnet import BasicBlock, Bottleneck

from epdtrainer.netutils.convbasics import ConvBlock, ConvBNReLU, ConvBN, BNReLU
from typing import Type, Any, Callable, Union, List, Optional


class BasicBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups: int = 1, base_width: int = 32,
                 dilation: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,):
        super(BasicBlock, self).__init__()
        self.cbr1 = ConvBNReLU(inplanes, planes, stride=stride)
        self.cb1 = ConvBN(planes, planes)
        # self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.quant_func = nn.quantized.FloatFunctional()

    def forward(self, x):
        residual = x
        out = self.cbr1(x)
        out = self.cb1(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.quant_func.add(residual, out)
        # out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 32,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 32.0)) * groups

        self.cbr1 = ConvBNReLU(
            inplanes, width, kernel_size=1, pad=0, groups=groups, dilation=dilation)
        self.cbr2 = ConvBNReLU(width, width, kernel_size=3, pad=1,
                               stride=stride, groups=groups, dilation=dilation)
        self.cb1 = ConvBN(width, planes * self.expansion,
                          kernel_size=1, pad=0, groups=groups, dilation=dilation)
        # self.relu = nn.ReLU6()
        self.downsample = downsample
        self.stride = stride
        self.quant_func = nn.quantized.FloatFunctional()

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.cbr1(x)
        out = self.cbr2(out)
        out = self.cb1(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.quant_func.add(out, identity)
        # out = self.relu(out)

        return out


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


class resnet_backbone(nn.Module):
    def __init__(
        self,
        layers,
        channels_ls,
        strides,
        in_channels=3,
        block='bottleneck',
        groups=1,
        width_per_group=32,
        norm_layer=None,
    ) -> None:
        super().__init__()
        assert block in ['basic', 'bottleneck']
        if block == 'basic':
            block = BasicBlock
        elif block == 'bottleneck':
            block = Bottleneck
        else:
            raise NotImplementedError
        self.in_channels = in_channels

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = channels_ls[0]
        self.dilation = 1

        self.bckb = nn.Sequential(
            ConvBNReLU(self.in_channels, self.inplanes//2,
                       kernel_size=3, stride=2, pad=1, bias=False),
            ConvBNReLU(self.inplanes//2, self.inplanes,
                       kernel_size=3, stride=2, pad=1, bias=False),
        )

        self.groups = groups
        self.base_width = width_per_group

        for i, (channel, num, stride) in enumerate(zip(channels_ls[1:], layers, strides)):
            self.bckb.add_module(
                str(i+2), self._make_layer(block, channel, num, stride=stride))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        block: BasicBlock,
        planes: int,
        blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if stride is not 1 or self.inplanes is not planes * block.expansion:
            downsample = nn.Sequential(
                ConvBN(self.inplanes, planes * block.expansion,
                       kernel_size=1, stride=stride, pad=0)
            )
        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.bckb(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


if __name__ == "__main__":
    import torchsummary
    import numpy as np
    from thop import profile, clever_format

    n, c, h, w = 1, 3, 32, 32

    backbone = resnet_backbone(
        layers=[3, 3, ],
        channels_ls=[32, 64, 128],
        strides=[2, 2, ],
        in_channels=c,
        block='bottleneck',
        groups=1,
        width_per_group=32,
        norm_layer=nn.BatchNorm2d
    )

    dummy_data = torch.as_tensor(
        np.random.normal(0, 1, (n, c, h, w)),
        dtype=torch.float32)
    macs, params = profile(backbone, inputs=(dummy_data, ))
    macs, params = clever_format([macs, params], "%.3f")

    print('resnet backbone macs: ' + macs + ', # of params: '+params)

    dummy_out = backbone(dummy_data)
    print(dummy_out.shape)
    torch.onnx.export(
        backbone,
        (dummy_data,),
        'backbone.onnx',
        export_params=True,
        verbose=False,
        opset_version=11
    )
