import torch
import torch.nn as nn
import torch.nn.functional as F


class UpSample(nn.Module):

    def __init__(self, scale_factor=2, mode="bilinear"):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
