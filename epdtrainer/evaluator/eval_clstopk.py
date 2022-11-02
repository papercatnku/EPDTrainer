import torch
from evaluator.evaluator_base import evaluator_base

import torch
import numpy as np
from evaluator.evaluator_base import evaluator_base


class cls_withtopk_eval(evaluator_base):
    def __init__(self, nm='', topk=1):
        super().__init__(nm)
        self.topk = topk
        self.reset()
        return

    def feed_data(self, decoderes_dict, data_dict):
        # decode_res_dict keys:
        # topk_ids
        self.total_num += data_dict['label'].shape[0]
        top1_match_num = sum(data_dict['label']
                             == decoderes_dict['topk_indices'][:, 0])
        topk_match_num = top1_match_num + sum(
            [(data_dict['label'] == decoderes_dict['topk_indices'][:, i]).sum() for i in range(1, self.topk)])
        self.matched_num += top1_match_num
        self.topk_matched_num += topk_match_num
        return

    def reset(self):
        self.matched_num = 0
        self.topk_matched_num = 0
        self.total_num = 0
        return

    def get_stastics_output(self):
        scalar_stastics_dict = {}
        if self.total_num == 0:
            scalar_stastics_dict[f'{self.nm}accuracy'] = 0
            scalar_stastics_dict[f'{self.nm}accuracy_top-{self.topk}'] = 0
        scalar_stastics_dict[f'{self.nm}accuracy'] = float(
            self.matched_num) / float(self.total_num)
        scalar_stastics_dict[f'{self.nm}accuracy_top-{self.topk}'] = float(
            self.topk_matched_num) / float(self.total_num)
        return scalar_stastics_dict

    def write_tblog(self, sw, tag='eval', step=None):
        scalar_stastics_dict = self.get_stastics_output()
        sw.add_scalars(tag, scalar_stastics_dict, global_step=step)
        return scalar_stastics_dict
