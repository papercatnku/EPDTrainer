import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import copy
from netutils.convbasics import ConvBlock, ConvBNReLU, ConvBN, BNReLU


class ShuffleV2BlockS1(nn.Module):
    def __init__(self, inp, oup, mid_channels, *, ksize, stride):
        super(ShuffleV2BlockS1, self).__init__()
        self.stride = stride
        assert stride == 1

        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp

        outputs = oup - inp

        branch_main = [
            # pw
            ConvBNReLU(inp, mid_channels, 1, 1, 0, bias=False),
            # dw
            ConvBN(mid_channels, mid_channels, ksize, stride,
                   pad, groups=mid_channels, bias=False),
            # pw-linear
            ConvBNReLU(mid_channels, outputs, 1, 1, 0, bias=False)
        ]
        self.branch_main = nn.Sequential(*branch_main)

        self.branch_proj = None

    def forward(self, old_x):
        x_proj, x = self.channel_shuffle(old_x)
        return torch.cat((x_proj, self.branch_main(x)), 1)

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert (num_channels % 4 == 0)
        x = x.reshape(batchsize * num_channels // 2, 2, height * width)
        x = x.permute(1, 0, 2)
        x = x.reshape(2, -1, num_channels // 2, height, width)
        return x[0], x[1]


class ShuffleV2BlockS2(nn.Module):
    def __init__(self, inp, oup, mid_channels, *, ksize, stride):
        super(ShuffleV2BlockS2, self).__init__()
        self.stride = stride
        assert stride == 2

        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp

        outputs = oup - inp

        branch_main = [
            # pw
            ConvBNReLU(inp, mid_channels, 1, 1, 0, bias=False),
            # dw
            ConvBN(mid_channels, mid_channels, ksize, stride,
                   pad, groups=mid_channels, bias=False),
            # pw-linear
            ConvBNReLU(mid_channels, outputs, 1, 1, 0, bias=False)
        ]

        self.branch_main = nn.Sequential(*branch_main)

        branch_proj = [
            # dw
            ConvBN(inp, inp, ksize, stride,
                   pad, groups=inp, bias=False),
            # pw-linear
            ConvBNReLU(inp, inp, 1, 1, 0, bias=False),
        ]

        self.branch_proj = nn.Sequential(*branch_proj)

    def forward(self, old_x):
        x_proj = old_x
        x = old_x
        return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert (num_channels % 4 == 0)
        x = x.reshape(batchsize * num_channels // 2, 2, height * width)
        x = x.permute(1, 0, 2)
        x = x.reshape(2, -1, num_channels // 2, height, width)
        return x[0], x[1]


class ShuffleNetV2BackBone(nn.Module):

    def __init__(
            self,
            in_channels=3,
            out_channels=128,
            stage_channels=[24, 48, 96, 192],
            stage_repeats=[4, 8, 4]
    ):

        super().__init__()

        assert(len(stage_channels) == len(stage_repeats) + 1)
        self.stage_repeats = stage_repeats
        self.stage_out_channels = stage_channels
        self.first_conv = ConvBNReLU(in_channels, stage_channels[0], 3, 2, 1)
        self.stages = []
        input_channel = stage_channels[0]

        for idxstage in range(len(stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]

            output_channel = self.stage_out_channels[idxstage+1]

            for i in range(numrepeat):
                if i == 0:
                    self.stages.append(ShuffleV2BlockS2(input_channel, output_channel,
                                                        mid_channels=output_channel // 2, ksize=3, stride=2))
                else:
                    self.stages.append(ShuffleV2BlockS1(input_channel // 2, output_channel,
                                                        mid_channels=output_channel // 2, ksize=3, stride=1))
                input_channel = output_channel
        self.stages = nn.Sequential(*self.stages)
        self.last_conv = ConvBNReLU(input_channel, out_channels, 1, 1, 0)
        return

    def forward(self, x):
        fea = self.first_conv(x)
        fea = self.stages(fea)
        fea = self.last_conv(fea)
        return fea

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def fuse_model(self):
        for m in self.modules():
            if type(m) == ConvBNReLU:
                torch.ao.quantization.fuse_modules_qat(
                    m, ['0', '1', '2'], inplace=True)
            if type(m) == BNReLU or type(m) == ConvBN:
                torch.ao.quantization.fuse_modules_qat(
                    m, ['0', '1'], inplace=True)


if __name__ == "__main__":
    net = ShuffleNetV2BackBone(
        in_channels=3,
        out_channels=96,
        stage_channels=[32, 64, 96],
        stage_repeats=[4, 8],
    )
    net.cuda()
    net.eval()
    import torchsummary

    torchsummary.summary(net, (3, 32, 32))

    pass
