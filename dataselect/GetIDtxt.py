# -*- coding: utf-8 -*-
"""
Created on  20190905

@author: linxueya
"""

import os
import pdb
root_path= '/home/shimy/FusionData/NC_PET'

files=os.listdir(root_path)
pdb.set_trace()
f=open('tempnc.txt','w')
for file in files:
    print(file)
    f.write(file+'\n')
f.close()