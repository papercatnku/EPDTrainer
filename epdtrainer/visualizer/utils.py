import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import cv2
import sys
import os

font_dir = os.path.dirname(__file__) + '/../../resources'
default_font = ImageFont.truetype(os.path.join(
    font_dir, "fangzhenghei.TTF"), size=14)

default_cls_color_pool = [
    (65, 105, 225),
    (67, 205, 128),
    (218, 165, 32),
    (205, 149, 12),
    (255, 250, 250),
    (28, 28, 28),
    (139, 0, 0),
    (0, 139, 139)
]


def cvimg2pillowimg(src_img, ifswap=True):
    # gray
    if len(src_img.shape) == 2:
        image = Image.fromarray(src_img, mode='L')
    # rgb
    elif ifswap:
        image = Image.fromarray(cv2.cvtColor(
            src_img, cv2.COLOR_BGR2RGB), mode='RGB')
    else:
        image = Image.fromarray(src_img)
    return image


def pillowimg2cvimg(src_img, ifswap=True):
    if(src_img.mode == 'L'):
        npimg = np.array(src_img)
    else:
        npimg = np.array(src_img)[:, :, :3]
        npimg = npimg.clip(0, 255).astype(np.uint8)
        if ifswap:
            npimg = cv2.cvtColor(npimg, cv2.COLOR_RGB2BGR)
    return npimg


class random_color_generator:
    def __init__(self,):
        return

    def __call__(self,):
        return tuple(np.random.randint(0, 255, (3,)).tolist())


class rcg_from_pool:
    def __init__(self, color_pool=[(65, 105, 225),
                                   (67, 205, 128),
                                   (218, 165, 32),
                                   (205, 149, 12),
                                   (255, 250, 250),
                                   (28, 28, 28),
                                   (139, 0, 0),
                                   (0, 139, 139)]
                 ):
        self.rcg = lambda: np.random.choice(color_pool) if color_pool else lambda: tuple(
            np.random.randint(0, 255, (3,)).tolist())

    def __call__(self,):
        return self.rcg()


if __name__ == '__main__':
    rcg = random_color_generator()
    print(rcg())

    print(os.path.dirname(__file__))
    pass
