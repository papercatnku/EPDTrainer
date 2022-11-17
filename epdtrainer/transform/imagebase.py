import cv2
import numpy as np
from loguru import logger


class imgbase_transform_base:
    def __init__(
            self,
            img_wh,
            if_train,
            mean=0.0,
            std=1.0,
            if_gray=False):
        self.img_wh = img_wh
        self.if_gray = if_gray
        self.if_train = if_train

        self.norm_func = lambda x: (x-mean) / std

        return

    def img2tensor(self, srcimg):
        h, w = srcimg.shape[:2]
        rszimg = cv2.resize(srcimg, self.img_wh,
                            interpolation=cv2.INTER_LINEAR)
        if (self.if_gray):
            outimg = cv2.cvtColor(rszimg, cv2.COLOR_BGR2GRAY)
        else:
            outimg = cv2.cvtColor(rszimg, cv2.COLOR_BGR2RGB)
        tsr = self.norm_func(outimg.astype(np.float32))
        tsr = np.transpose(tsr, axes=(2, 0, 1))
        np.ascontiguousarray(tsr, dtype=np.float32)
        return tsr

    def __call__(self, data_dict):
        out_data_dict = {
            'img': data_dict['img'],  # keep for easy of visualize
            'img_tensor': self.img2tensor(data_dict['img']),
        }
        if self.if_train:
            tar_dict = self.tar_transform(data_dict)
            out_data_dict.update(tar_dict)
        else:
            gt_dict = self.gt_transform(data_dict)
            out_data_dict.update(gt_dict)

        return out_data_dict

    def tar_transform(self, data_dict):
        # to be override
        logger.warning("tar_transform shoulde be implemented in child class")
        tar_dict = {}
        return tar_dict

    def gt_transform(self, data_dict):
        # to be override
        logger.warning("gt_transform shoulde be implemented in child class")
        gt_dict = {}
        return gt_dict
