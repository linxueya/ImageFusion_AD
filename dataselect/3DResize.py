# -*- coding: utf-8 -*-
"""
Created on  20190905

@author: linxueya
"""
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from skimage import transform
import os
import pdb

root_path='D:/Master/FusionData/AD_PET/002_S_5018'
imgs=os.listdir(root_path)
img=imgs[0]
img_name=os.path.join(root_path,img)
nii=nib.load(img_name)
im=nii.get_data()
img_3d_max = np.amax(im)
img_3d = im / img_3d_max
imx = np.array(img_3d)
img_2d = np.squeeze(imx)
image=transform.resize(img_2d,(64,64,64))
plt.imshow(image[:,:,30])
plt.show()
pdb.set_trace()