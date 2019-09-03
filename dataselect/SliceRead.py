
# -*- coding: utf-8 -*-
"""
Created on  20190902

@author: linxueya
"""

import numpy as np
import os
import pdb
root= 'D:\Master\FusionData\AD_JPG'

imgs = os.listdir(root)

img_num=[]
for img_id in imgs:
    img_path=os.path.join(root,img_id)
    n=len(os.listdir(img_path))
    img_num.append(n)
    print(img_id,n)

result_dic={}
for item_str in img_num:
    if item_str not in result_dic:
        result_dic[item_str]=1
    else:
        result_dic[item_str]+=1
print(result_dic)