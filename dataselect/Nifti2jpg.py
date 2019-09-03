# -*- coding: utf-8 -*-
"""
Created on  20190902

@author: linxueya
"""

import numpy as np
import os
import nibabel as nib
import pdb
import matplotlib.pyplot as plt
from PIL import Image

preserving_ratio = 0.25




root= 'D:\Master\FusionData\AD_PET'
save_path= 'D:\Master\FusionData\AD_JPG'
for img_id in os.listdir(root):
    img_paths=os.path.join(root, img_id)
    img_name=os.listdir(img_paths)[0]
    img_name=os.path.join(img_paths,img_name)
    img = nib.load(img_name).get_data()
    print(img.shape)

    img_3d_max = np.amax(img)
    img = img / img_3d_max * 255  # 对6所求的像素进行归一化变成0-255范围,这里就是三维数据

    for i in range(img.shape[2]):  # 对切片进行循环
        print(i)
        print(img.shape)
        imgx = img[:, :, i]  # 取出一张图像
        imgx = np.array(imgx)
        img_2d = np.squeeze(imgx)
        #plt.imshow(img_2d) #显示图像
        #plt.pause(0.001)
        # filter out 2d images containing < 10% non-zeros
        # print(np.count_nonzero(img_2d))
        # print("before process:", img_2d.shape)
        # if float(np.count_nonzero(img_2d)) / img_2d.size >= preserving_ratio:  # 表示一副图像非0个数超过整副图像的10%我们才把该图像保留下来
        #    img_2d = img_2d / 127.5 - 1  # 对最初的0-255图像进行归一化到[-1, 1]范围之内
        #    img_2d = np.transpose(img_2d, (1, 0))  # 这个相当于将图像进行旋转90度

        im = Image.fromarray(img_2d)
        im = im.convert('RGB')
        im_name='AD'+str(i+1)
        im_path=os.path.join(save_path,img_id)

        if not os.path.exists(im_path):
            os.makedirs(im_path)  # 创建路径
        save_name=os.path.join(im_path, im_name) + '.jpg'
        im.save(save_name)
        print('save sucsess '+ img_id)
            # plt.imshow(img_2d)
            # plt.pause(0.01)
