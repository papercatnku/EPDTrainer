import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import cv2


class random_color_generator:
    def __init__(self,):
        return

    def __call__(self,):
        return tuple(np.random.randint(0, 255, (3,)).tolist())


if __name__ == '__main__':
    rcg = random_color_generator()
    print(rcg())
    pass
