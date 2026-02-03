import numpy as np
import scipy.stats as stats
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import random
import collections
import cv2
import torch
from torchvision import transforms

# # # 1. Augmentation for image

def hflip(img, mask, p=0.5):
    if random.random() < p:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return img,mask

def wflip(img, mask,p=0.5):
    if random.random() < p:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
    return img,mask



def label_weak_aug(img, mask):

    augmentation_methods = {
        'hflip': (lambda img, mask: hflip(img, mask), 0.6),
        'wflip': (lambda img, mask: wflip(img,mask), 0.6)
    }

    # 依次应用选定的图像增强方法
    for method, (func, prob) in augmentation_methods.items():
        if random.random() < prob:
            img , mask= func(img , mask)


    return img, mask

from PIL import Image

# 1. 加载图像和掩码
img_path = r"D:\python-learn\MLPMatch-main\crack_split\JPEGImages\023_768_1536.jpg"  # 替换为你的图像路径
mask_path = r"D:\python-learn\MLPMatch-main\crack_split\SegmentationClassAug\023_768_1536.PNG"  # 替换为你的掩码路径

img = Image.open(img_path).convert("RGB")  # 确保图像是 RGB 格式
mask = Image.open(mask_path).convert("L")  # 确保掩码是灰度格式

# 2. 调用增强函数
aug_img, aug_mask = label_weak_aug(img, mask)

# 3. 可视化结果
# aug_img.show(title="Augmented Image")
# aug_mask.show(title="Augmented Mask")

# 4. 保存结果（可选）
aug_img.save("augmented_image.jpg")
aug_mask.save("augmented_mask.png")



