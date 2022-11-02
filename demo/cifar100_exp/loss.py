import torch
import torch.nn as nn
import torch.nn.functional as F


class clsification_loss(nn.Module):
    def __init__(self,):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
        return

    def forward(self, net_predict, data_dict):
        loss = self.loss(
            net_predict['cls_pred'],
            data_dict['label']
        )

        loss_dict = {
            'ce_loss': loss
        }
        return loss_dict
