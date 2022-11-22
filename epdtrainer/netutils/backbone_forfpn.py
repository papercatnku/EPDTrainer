import torch
import torch.nn as nn
from epdtrainer.netutils.darknet import CSPDarknet
from epdtrainer.netutils.convbasics import BaseConv, CSPLayer, DWConv, SPPFBottleneck, SPPBottleneck, Focus


class DarknetFPNBackbone(nn.Module):
    def __init__(
            self,
            in_channels=3,
            base_channels=32,
            depth=1,
            out_features=('dark3', 'dark4', 'dark5'),
            depthwise=False,
            act='silu',
            enable_sppf=False,
            qat=False):
        super().__init__()
        self.quant_func = nn.quantized.FloatFunctional()
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv
        base_depth = max(depth, 1)

        self.stem = Focus(
            in_channels,
            base_channels,
            ksize=3,
            act=act)

        # dark2
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(
                base_channels * 2,
                base_channels * 2,
                n=base_depth,
                depthwise=depthwise,
                act=act
            )
        )
        # dark3
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(
                base_channels * 4,
                base_channels * 4,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act
            ),
        )

        # dark4
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(
                base_channels * 8,
                base_channels * 8,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act
            ),
        )
        if (enable_sppf):
            self.dark5 = nn.Sequential(
                Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
                SPPFBottleneck(base_channels * 16, base_channels *
                               16, activation=act, qat=qat),
                CSPLayer(
                    base_channels * 16,
                    base_channels * 16,
                    n=base_depth,
                    shortcut=False,
                    depthwise=depthwise,
                    act=act
                ),
            )
        else:
            self.dark5 = nn.Sequential(
                Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
                CSPLayer(
                    base_channels * 16,
                    base_channels * 16,
                    n=base_depth,
                    shortcut=False,
                    depthwise=depthwise,
                    act=act
                ),
            )

        return

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return tuple([outputs[k] for k in self.out_features])


if __name__ == '__main__':
    import numpy as np
    import torchsummary
    from thop import profile, clever_format

    yolo_fpn_stem = DarknetFPNBackbone(
        in_channels=1,
        base_channels=16,
        act='lrelu'
    )

    n, c, h, w = 1, 1, 416, 416

    dummy_data = torch.as_tensor(np.random.normal(
        0, 1, (n, c, h, w)), dtype=torch.float32)
    dummy_outputs = yolo_fpn_stem(dummy_data)

    torchsummary.summary(yolo_fpn_stem, (c, h, w), device='cpu')

    macs, params = profile(yolo_fpn_stem, inputs=(dummy_data,))
    macs, params = clever_format([macs, params], "%.3f")
    print('yolo_fpn_stem macs: ' + macs + ', # of params: '+params)

    torch.onnx.export(
        yolo_fpn_stem,
        dummy_data,
        'yolo_fpn_stem_example.onnx',
        export_params=True,
        verbose=False,
        opset_version=11
    )
    pass
