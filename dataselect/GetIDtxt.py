# -*- coding: utf-8 -*-
"""
Created on  20190905

@author: linxueya
"""

import os

root_path= 'D:\Master\FusionData\AD_PET'

files=os.listdir(root_path)
f=open('825_Subject_NC.txt','w')
for file in files:
    print(file)
    f.write(file+'\n')
f.close()