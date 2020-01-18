# coding:utf8
import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import pickle


class DogCat(data.Dataset):

    def __init__(self, root, train=True, test=False):
        """
        主要目标： 获取所有图片的地址，并根据训练，验证，测试划分数据
        """
        self.test = test
        imgs = [os.path.join(root, img) for img in os.listdir(root)]

        # test1: data/test1/8973.jpg
        # train: data/train/cat.10004.jpg


        imgs_num = len(imgs)

        if self.test:
            self.imgs = imgs
        elif train:
            self.imgs = imgs[:int(0.7 * imgs_num)]
        else:
            self.imgs = imgs[int(0.7 * imgs_num):]

        if self.test:
            self.transforms = T.Compose([
                T.Resize(224),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.059702], std=[0.113304])
            ])
        else:
            self.transforms = T.Compose([
                T.Resize(224),
                T.CenterCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.060238], std=[0.113941])
            ])

    def __getitem__(self, index):
        """
        一次返回一张图片的数据
        """
        img_path = self.imgs[index]

        # label = 0 if 'SMCI' in img_path.split('/')[-1] else 1
        if 'NC' in img_path.split('/')[-1]:
            label = 0
        elif 'AD' in img_path.split('/')[-1]:
            label = 1
        elif 'SMCI' in img_path.split('/')[-1]:
            label = 2
        else:
            label = 3


        data = Image.open(img_path)
        data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)

# import ipdb
# val_data = DogCat('/home/shimy/FusionData/gray_mri/validation', test=True)
# for data, label, path in val_data:
#     print( path, label)
#     # ipdb.set_trace()

