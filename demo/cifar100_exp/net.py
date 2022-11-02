import torch
import torch.nn as nn
import torch.nn.functional as F
from netutils.shufflenetV2 import ShuffleNetV2BackBone
import torchvision
from torchvision.models.resnet import Bottleneck, ResNet


def creat_cls_inf_net(config):
    net = nn.Sequential(
        *[
            ShuffleNetV2BackBone(
                in_channels=config.in_channels,
                out_channels=config.out_channels,
                stage_channels=config.stage_channels,
                stage_repeats=config.stage_repeats
            ),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(config.get('drop_out', 0.5)),
            nn.Linear(config.out_channels, config.num_class)
        ]
    )
    return net


def resnet(config):
    net = ResNet(
        Bottleneck,
        [3, 4, 6, 3],
        num_classes=config.num_class)
    return net


if __name__ == '__main__':
    from easydict import EasyDict

    config = EasyDict(
        in_channels=3,
        out_channels=256,
        stage_repeats=[4, 8, 4],
        stage_channels=[24, 116, 232, 464],
        num_class=100,
    )
    net = creat_cls_inf_net(config)

    pass
