import torch
import torch.nn.functional as F


class clsification_topk_decoder:
    def __init__(self, topk=3):
        self.topk = topk
        return

    def __call__(self, net_predict):
        sofmatmax_prob = F.softmax(net_predict['cls_pred'], dim=-1)
        topk_vals, topk_indices = torch.topk(sofmatmax_prob, k=self.topk)
        # topk_vals: n x k
        # topk_indices: n x k
        out_dict = {
            'pred_probs': topk_vals,
            'topk_indices': topk_indices
        }

        return out_dict
