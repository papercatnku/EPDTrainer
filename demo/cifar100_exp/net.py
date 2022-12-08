import torch
import torch.nn as nn
import torch.nn.functional as F
import epdtrainer
from epdtrainer.netutils.shufflenetV2 import ShuffleNetV2BackBone
from epdtrainer.netutils.resnet import resnet_backbone
import torchvision
# from torchvision.models.resnet import Bottleneck, ResNet


def shufflenet_cls_prototype(config):
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


def resne_cls_prototypet(config):
    net = nn.Sequential(
        *[
            resnet_backbone(
                layers=config.layers,
                channels_ls=config.channels_ls,
                strides=config.strides,
                in_channels=config.in_channels,
                block=config.block,
            ),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(config.get('drop_out', 0.5)),
            nn.Linear(config.out_channels, config.num_class)
        ]
    )
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
    net = shufflenet_cls_prototype(config)

    pass
