# -*- coding: utf-8 -*-
"""
Created on  20190903

@author: linxueya
"""

import os,shutil
import pdb

def mymovefile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.move(srcfile,dstfile)          #移动文件
        print ("move %s -> %s"%( srcfile,dstfile))

def mycopyfile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.copyfile(srcfile,dstfile)      #复制文件
        print ("copy %s -> %s"%( srcfile,dstfile))



root_path = 'D:\Master\FusionData\FUSION_AD_PET\ADNI'
save_path= 'D:\Master\FusionData\AD_PET'
imgs = os.listdir(root_path)
img_num=[]
for img in imgs:
    img1 = os.path.join(root_path,img)
    img2 = os.listdir(img1)
    img3 = os.path.join(img1,img2[0])
    img4 = os.listdir(img3)
    img5 = os.path.join(img3,img4[0])
    img6 = os.listdir(img5)
    img7 = os.path.join(img5,img6[0])

    #os.chdir(img7)
    #n=len(os.listdir(img7))
    #img_num.append(n)
    #print(img,n)

    src_path=os.path.join(root_path,img7)
    dir_path=os.path.join(save_path,img)
    for pet in os.listdir(src_path):
        src_pet=os.path.join(src_path,pet)
        dir_pet=os.path.join(dir_path,pet)
        mycopyfile(src_pet,dir_pet)



