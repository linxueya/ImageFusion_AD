# -*- coding: utf-8 -*-
"""
Created on  20190911

@author: linxueya
"""


import os,pdb
import shutil
import random
import pdb
root_path= '/home/shimy/FusionData/data_mri'
NC_file='/home/shimy/FusionData/Slice/NCMRI_ZSlice.txt'
AD_file='/home/shimy/FusionData/Slice/ADMRI_ZSlice.txt'

# root_path= '/home/shimy/FusionData/data_pet'
# NC_file='/home/shimy/FusionData/Slice/NCPET_ZSlice.txt'
# AD_file='/home/shimy/FusionData/Slice/ADPET_ZSlice.txt'

train_percentage = 0.8
val_percentage = 0.2

def mymovefile(srcfile, dstfile):
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        fpath, fname = os.path.split(dstfile)  # 分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)  # 创建路径
        shutil.move(srcfile, dstfile)  # 移动文件
        #print("move %s -> %s" % (srcfile, dstfile))

def mycopyfile(srcfile, dstfile):
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        fpath, fname = os.path.split(dstfile)  # 分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)  # 创建路径
        shutil.copyfile(srcfile, dstfile)  # 复制文件
        #print("copy %s -> %s" % (srcfile, dstfile))

train_folder=os.path.join(root_path,"train")
if not os.path.exists(train_folder):
    os.mkdir(train_folder)
val_folder=os.path.join(root_path,"validation")
if not os.path.exists(val_folder):
    os.mkdir(val_folder)

def getpath(filename):
    ID_list=[]
    with open(filename) as f:
        files=f.readlines()
        for line in files:
            ID_list.append(line.strip())
    return ID_list

def creatdataset(IDs):
    for idx,img in enumerate(IDs):
        if idx<len(IDs)*train_percentage:
            path1=train_folder
        else:
            path1=val_folder
        path2 = 'AD' if "AD" in img.split('/')[-3] else 'NC'
        new_name=img.split('/')[-3]+img.split('/')[-1]
        save_path=os.path.join(path1,path2)
        save_name=os.path.join(save_path,new_name)
        mycopyfile(img,save_name)
        print("successful create ->"+save_name)



img_name=getpath(NC_file)
creatdataset(img_name)

img_name=getpath(AD_file)
creatdataset(img_name)






