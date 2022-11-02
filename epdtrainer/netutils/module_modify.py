import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


def _replace_relu(module):
    reassign = {}
    for name, mod in module.named_children():
        _replace_relu(mod)
        # Checking for explicit type instead of instance
        # as we only want to replace modules of the exact type
        # not inherited classes
        if type(mod) == nn.ReLU or type(mod) == nn.ReLU6:
            reassign[name] = nn.ReLU(inplace=False)
    for key, value in reassign.items():
        logger.info(f"replace {key} with {value}", )
        module._modules[key] = value
