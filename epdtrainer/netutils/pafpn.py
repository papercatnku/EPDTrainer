import torch
import torch.nn as nn
import torch.nn.functional as F
from epdtrainer.netutils.convbasics import DWConv, BaseConv, CSPLayer, SPPFBottleneck
from epdtrainer.netutils.miscellaneous import UpSample


class PAFPN3In3Out(nn.Module):
    def __init__(
        self,
        in_channels,
        depthwise=False,
        act="silu",
        depth=1,
        qat=False
    ):
        super().__init__()
        self.upsample = UpSample(scale_factor=2, mode="bilinear")
        self.quant_func = nn.quantized.FloatFunctional()

        self.in_num = len(in_channels)
        assert(self.in_num == 3)

        Conv = DWConv if depthwise else BaseConv

        self.lateral_conv0 = BaseConv(
            int(in_channels[2]), int(in_channels[1]), 1, 1, act=act
        )

        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1]),
            int(in_channels[1]),
            depth,
            False,
            depthwise=depthwise,
            act=act,
            qat=qat
        )  # cat
        self.reduce_conv1 = BaseConv(
            int(in_channels[1]), int(in_channels[0]), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0]),
            int(in_channels[0]),
            depth,
            False,
            depthwise=depthwise,
            act=act,
            qat=qat
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[0]), int(in_channels[0]), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0]),
            int(in_channels[1]),
            depth,
            False,
            depthwise=depthwise,
            act=act,
            qat=qat
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[1]), int(in_channels[1]), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1]),
            int(in_channels[2]),
            depth,
            False,
            depthwise=depthwise,
            act=act,
            qat=qat
        )

        return

    def forward(self, x2, x1, x0):
        #  = x
        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = self.quant_func.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = self.quant_func.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = self.quant_func.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = self.quant_func.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = (pan_out2, pan_out1, pan_out0)
        return outputs


if __name__ == '__main__':
    import numpy as np
    import torchsummary
    from thop import profile, clever_format

    # n, h, w = 1, 52, 52  # 416x416 -> 8x downsize
    n, h, w = 1, 32, 48  # 256x384 -> 8x downsize

    in_channels = [64, 128, 256]

    pafpn = PAFPN3In3Out(in_channels, act='relu')

    dummy_data_ls = [
        torch.as_tensor(
            np.random.normal(0, 1, (n, ic, h//stride, w//stride)),
            dtype=torch.float32) for ic, stride in zip(in_channels, [1, 2, 4])
    ]

    dummy_out = pafpn(*dummy_data_ls)

    in_size_tuple = [
        (ic, h//stride, w//stride) for ic, stride in zip(in_channels, [1, 2, 4])
    ]
    # in_size_tuple=tuple(in_size_tuple)

    torchsummary.summary(pafpn, in_size_tuple, device='cpu')

    macs, params = profile(pafpn, inputs=dummy_data_ls)
    macs, params = clever_format([macs, params], "%.3f")
    print('pafpn macs: ' + macs + ', # of params: '+params)

    torch.onnx.export(
        pafpn,
        tuple(dummy_data_ls),
        'pafpn_example.onnx',
        export_params=True,
        verbose=False,
        opset_version=11
    )

    pass
