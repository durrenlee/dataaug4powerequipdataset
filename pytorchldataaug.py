"""
Use pytorch torchvision package to do data aug
"""

import os
import sys
from PIL import Image
import torchvision
import random
import torch
import numpy as np


def resize_img(img):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize([224, 224])
    ])
    return transform(img)


def random_rotation(img):
    print("random_rotation")
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize([224, 224]),
        torchvision.transforms.RandomRotation(degrees=(-60, 60))
    ])
    return transform(img)


def random_horizontal_flip(img):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize([224, 224]),
        torchvision.transforms.RandomHorizontalFlip(p=1)
    ])
    return transform(img)


def random_vertical_flip(img):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize([224, 224]),
        torchvision.transforms.RandomVerticalFlip(p=1)
    ])
    return transform(img)


def gaussian_blur(img):
    kernel_sizes = [1, 3, 5, 7, 9, 11]
    size = random.choice(kernel_sizes)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize([224, 224]),
        torchvision.transforms.GaussianBlur(kernel_size=size)
    ])
    return transform(img)


def center_crop(img):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize([224, 224]),
        torchvision.transforms.CenterCrop([224, 224])
    ])
    return transform(img)


def random_crop(img):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop([224, 224], [4, 4, 4, 4], True)
    ])
    return transform(img)


"""
source: https://blog.csdn.net/u013685264/article/details/122562509
"""


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


def do_dataaug(root_dir):
    fileList = os.listdir(root_dir)

    for fileName in fileList:
        print("------------------------------------------------------------------")
        print("目录名：" + str(fileName))
        filename_arr = fileName.split(" ")
        sub_dir = root_dir + "\\" + fileName
        if os.path.isdir(sub_dir):
            file_List_insub = os.listdir(sub_dir)
            print("文件总数：" + str(len(file_List_insub)))
            for a_file in file_List_insub:
                img_path = sub_dir + "\\" + a_file
                img_path = img_path.replace("\\", "/")
                print(img_path)
                with open(img_path, 'rb') as f:
                    img = Image.open(f).convert('RGB')
                    f1 = resize_img(img)
                    f1.save(img_path)

                    f2 = random_rotation(img)
                    f2.save(img_path.split('.')[0] + '_' + str(2) + '.jpg')

                    f3 = random_horizontal_flip(img)
                    f3.save(img_path.split('.')[0] + '_' + str(3) + '.jpg')

                    f4 = gaussian_blur(img)
                    f4.save(img_path.split('.')[0] + '_' + str(4) + '.jpg')

                    f5 = random_vertical_flip(f1)
                    f5 = torchvision.transforms.ToTensor()(f5)

                    n_holes = random.choice(list(range(1, 5)))
                    cut = Cutout(n_holes=n_holes, length=16)
                    f5 = cut(f5)
                    f5 = torchvision.transforms.ToPILImage()(f5)
                    f5.save(img_path.split('.')[0] + '_' + str(5) + '.jpg')

                    f6 = random_crop(img)
                    f6.save(img_path.split('.')[0] + '_' + str(6) + '.jpg')

    sys.stdin.flush()  # 刷新


root_dir = "D:\\tempImg"
# root_dir = "D:\\epimgtxtretrieval\\dataset\\test"

do_dataaug(root_dir)
