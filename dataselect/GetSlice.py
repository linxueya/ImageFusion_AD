# -*- coding: utf-8 -*-
"""
Created on  20190905

@author: linxueya
"""
import os
import pdb
import nibabel as nib
import numpy as np
from skimage import transform
import matplotlib.pyplot as plt
from PIL import Image


ID_list=[]
save_path='D:\Master\FusionData\AD_JPG'
file_name='AD_original.txt'
label="AD"
with open(file_name) as f:
    files=f.readlines()
    for line in files:
        ID_list.append(line.strip())

for idx,id in enumerate(ID_list):
    nii = nib.load(id).get_data()
    img_3d_max = np.amax(nii)
    img_3d = nii / img_3d_max   # 对6所求的像素进行归一化变成0-1范围,这里就是三维数据
    imgn = np.array(img_3d)
    img = np.squeeze(imgn)
    imgr = transform.resize(img, (64, 64, 64)) #将3D图像重采样到指定大小

    x,y,z=imgr.shape

    sbj = "S%03d" %(idx)
    sbj_path=os.path.join(save_path,sbj)

    for i in range(imgr.shape[2]):  # 对Z轴切片进行循环
        img_2d = imgr[:, :, i]  # 取出一张图像
        img_2d = img_2d*255
        im = Image.fromarray(img_2d)
        im = im.convert('RGB')

        slice_path=os.path.join(sbj_path,"Zslice")
        im_name=label+"%03d" %(i)
        if not os.path.exists(slice_path):
            os.makedirs(slice_path)  # 创建路径
        save_name=os.path.join(slice_path, im_name) + '.jpg'
        if not os.path.isfile(save_name):
            im.save(save_name)

    for i in range(imgr.shape[1]):  # 对Y轴切片进行循环
        # print("the resize shape of {}= {}".format(i,imgr.shape))
        img_2d = imgr[:, i, :]  # 取出一张图像
        img_2d = img_2d * 255
        im = Image.fromarray(img_2d)
        im = im.convert('RGB')

        slice_path = os.path.join(sbj_path, "Yslice")
        im_name = label + "%03d" % (i)
        if not os.path.exists(slice_path):
            os.makedirs(slice_path)  # 创建路径
        save_name = os.path.join(slice_path, im_name) + '.jpg'
        if not os.path.isfile(save_name):
            im.save(save_name)

    for i in range(imgr.shape[0]):  # 对X轴切片进行循环
        img_2d = imgr[i, :, :]  # 取出一张图像
        img_2d = img_2d * 255
        im = Image.fromarray(img_2d)
        im = im.convert('RGB')

        slice_path = os.path.join(sbj_path, "Xslice")
        im_name = label + "%03d" % (i)
        if not os.path.exists(slice_path):
            os.makedirs(slice_path)  # 创建路径
        save_name = os.path.join(slice_path, im_name) + '.jpg'
        if not os.path.isfile(save_name):
            im.save(save_name)
    print('save successful '+ str(idx))

