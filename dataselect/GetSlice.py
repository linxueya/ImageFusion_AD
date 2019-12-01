# -*- coding: utf-8 -*-
"""
Created on  20190905

@author: linxueya
"""
import os
import nibabel as nib
import numpy as np
from skimage import transform
from PIL import Image


def getslice(save_path, file_name, label):
    id_list = []
    with open(file_name) as f:
        files = f.readlines()
        for line in files:
            id_list.append(line.strip())

    slice_num = []
    #
    # if label == 'NC':
    #     id_list = id_list[0:-40]
    for idx, id in enumerate(id_list):
        nii = nib.load(id).get_data()
        img_3d_max = np.amax(nii)
        img_3d = nii / img_3d_max   # 对6所求的像素进行归一化变成0-1范围,这里就是三维数据
        imgn = np.array(img_3d)
        img = np.squeeze(imgn)
        print(idx, id)

        # if save_path.split('/')[-1] == "pet_bmp":
        #     imgr = transform.resize(img, (180, 180, 180))   # 将3D图像重采样到指定大小,pet原始图像为53 63 52
        # else:
        #     imgr = img[:-1, 19:-18, :-1]    # 原始图像大小181 217 181

        # pet原始图像为53 63 52
        # mri原始图像大小181 217 181
        imgr = transform.resize(img, (180, 180, 180),mode='constant')

        x, y, z = imgr.shape
        slice_num.append(x)

        sbj = label + "%03d" % idx
        sbj_path = os.path.join(save_path, sbj)

        slicex1 = x//2-30
        slicex2 = x//2+30
        slicey1 = y//2-30
        slicey2 = y//2+30
        slicez1 = z//2-30
        slicez2 = z//2+30

        for i in range(slicez1, slicez2):  # 对Z轴切片进行循环
            img_2d = imgr[:, :, i]  # 取出一张图像
            img_2d = img_2d * 255
            im = Image.fromarray(img_2d)
            im = im.convert('L')
            im = im.rotate(90)

            slice_path = os.path.join(sbj_path, "Zslice")
            im_name = "Z%03d" % i
            if not os.path.exists(slice_path):
                os.makedirs(slice_path)  # 创建路径
            save_name = os.path.join(slice_path, im_name) + '.bmp'
            if not os.path.isfile(save_name):
                im.save(save_name)

        for i in range(slicey1, slicey2):  # 对Y轴切片进行循环
            # print("the resize shape of {}= {}".format(i,imgr.shape))
            img_2d = imgr[:, i, :]  # 取出一张图像
            img_2d = img_2d * 255
            im = Image.fromarray(img_2d)
            im = im.convert('L')
            im = im.rotate(90)

            slice_path = os.path.join(sbj_path, "Yslice")
            im_name = "Y%03d" % i
            if not os.path.exists(slice_path):
                os.makedirs(slice_path)  # 创建路径
            save_name = os.path.join(slice_path, im_name) + '.bmp'
            if not os.path.isfile(save_name):
                im.save(save_name)

        for i in range(slicex1, slicex2):  # 对X轴切片进行循环
            img_2d = imgr[i, :, :]  # 取出一张图像
            img_2d = img_2d * 255
            im = Image.fromarray(img_2d)
            im = im.convert('L')
            im = im.rotate(90)

            slice_path = os.path.join(sbj_path, "Xslice")
            im_name = "X%03d" % i
            if not os.path.exists(slice_path):
                os.makedirs(slice_path)  # 创建路径
            save_name = os.path.join(slice_path, im_name) + '.bmp'
            if not os.path.isfile(save_name):
                im.save(save_name)
        print('save successful' + str(idx))


if __name__ == "__main__":
    save_path = '/home/shimy/FusionData/Subject_NC/mri_bmp'
    file_name = '/home/shimy/FusionData/Subject_NC/NC_mri.txt'
    label = "NC"
    getslice(save_path, file_name, label)

    save_path = '/home/shimy/FusionData/Subject_AD/mri_bmp'
    file_name = '/home/shimy/FusionData/Subject_AD/AD_mri.txt'
    label = "AD"
    getslice(save_path, file_name, label)

    # save_path = '/home/shimy/FusionData/Subject_EMCI/mri_bmp'
    # file_name = '/home/shimy/FusionData/Subject_EMCI/EMCI_mri.txt'
    # label = "EMCI"
    # getslice(save_path, file_name, label)
    #
    # save_path = '/home/shimy/FusionData/Subject_LMCI/mri_bmp'
    # file_name = '/home/shimy/FusionData/Subject_LMCI/LMCI_mri.txt'
    # label = "LMCI"
    # getslice(save_path, file_name, label)

    # save_path = '/home/shimy/FusionData/Subject_NC/pet_bmp'
    # file_name = '/home/shimy/FusionData/Subject_NC/NC_pet.txt'
    # label = "NC"
    # getslice(save_path, file_name, label)
    #
    # save_path = '/home/shimy/FusionData/Subject_AD/pet_bmp'
    # file_name = '/home/shimy/FusionData/Subject_AD/AD_pet.txt'
    # label = "AD"
    # getslice(save_path, file_name, label)
    #
    # save_path = '/home/shimy/FusionData/Subject_EMCI/pet_bmp'
    # file_name = '/home/shimy/FusionData/Subject_EMCI/EMCI_pet.txt'
    # label = "EMCI"
    # getslice(save_path, file_name, label)
    #
    # save_path = '/home/shimy/FusionData/Subject_LMCI/pet_bmp'
    # file_name = '/home/shimy/FusionData/Subject_LMCI/LMCI_pet.txt'
    # label = "LMCI"
    # getslice(save_path, file_name, label)