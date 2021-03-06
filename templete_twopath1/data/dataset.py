# coding:utf8
import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import pickle


class MriPet(data.Dataset):

    def __init__(self, root_mri, root_pet, train=True, test=False):
        """
        主要目标： 获取所有图片的地址，并根据训练，验证，测试划分数据
        """
        self.test = test
        imgs_mri = [os.path.join(root_mri, img) for img in os.listdir(root_mri)]
        imgs_pet = [os.path.join(root_pet, img) for img in os.listdir(root_pet)]
        imgs_num = len(imgs_mri)

        if self.test:
            self.imgs_mri = imgs_mri
            self.imgs_pet = imgs_pet
        elif train:
            self.imgs_mri = imgs_mri[:int(0.7 * imgs_num)]
            self.imgs_pet = imgs_pet[:int(0.7 * imgs_num)]
        else:
            self.imgs_mri = imgs_mri[int(0.7 * imgs_num):]
            self.imgs_pet = imgs_pet[int(0.7 * imgs_num):]

        self.imgs_mri_mean = []
        self.imgs_mri_std = []
        self.imgs_pet_mean = []
        self.imgs_pet_std = []

        if self.test:
            self.transforms_mri = T.Compose([
                T.Resize(224),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.059702], std=[0.113304])
            ])
            self.transforms_pet = T.Compose([
                T.Resize(224),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.335203], std=[0.237316])
            ])
        else:
            self.transforms_mri = T.Compose([
                T.Resize(224),
                T.CenterCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.060238], std=[0.113941])
            ])
            self.transforms_pet = T.Compose([
                T.Resize(224),
                T.CenterCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.334345], std=[0.237282])
            ])



    def __getitem__(self, index):
        """
        一次返回一张图片的数据
        """
        mri_path = self.imgs_mri[index]
        pet_path = self.imgs_pet[index]

        # label = 0 if 'AD' in mri_path.split('/')[-1] else 1
        if 'NC' in mri_path.split('/')[-1]:
            label = 0
        elif 'AD' in mri_path.split('/')[-1]:
            label = 1
        elif 'SMCI' in mri_path.split('/')[-1]:
            label = 2
        else:
            label = 3

        data_mri = Image.open(mri_path)
        data_mri = self.transforms_mri(data_mri)
        data_pet = Image.open(pet_path)
        data_pet = self.transforms_pet(data_pet)

        return data_mri, data_pet, label
        # return data_mri, mri_path,data_pet,pet_path, label

    def __len__(self):
        return len(self.imgs_mri)

# import pdb
# root1 = '/home/shimy/FusionData/data_mri/train'
# root2 = '/home/shimy/FusionData/data_pet/train'
# val_data = MriPet(root1, root2, train=False)
# for data, path,data2,path2, label in val_data:
#     print(data.size(), data2.size(), label)
#     print(path,'\n',path2)
#     pdb.set_trace()

