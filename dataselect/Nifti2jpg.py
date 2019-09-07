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
from skimage import transform

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
    img_3d = img / img_3d_max   # 对6所求的像素进行归一化变成0-1范围,这里就是三维数据
    imgn = np.array(img_3d)
    img = np.squeeze(imgn)
    imgr = transform.resize(img, (64, 64, 64))

    for i in range(imgr.shape[2]):  # 对切片进行循环
        #print("the resize shape of {}= {}".format(i,imgr.shape))
        img_2d = imgr[:, :, i]  # 取出一张图像
        img_2d = img_2d*255

        im = Image.fromarray(img_2d)
        im = im.convert('RGB')
        im_name='AD'+str(i+1)
        im_path=os.path.join(save_path,img_id)

        if not os.path.exists(im_path):
            os.makedirs(im_path)  # 创建路径
        save_name=os.path.join(im_path, im_name) + '.jpg'
        im.save(save_name)
        print('save sucsess '+ img_id)
        #plt.imshow(img_2d)
        #plt.pause(0.01)


