#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on  20191130

@author: linxueya
"""
import os
import shutil
import glob
import ipdb


def mymovefile(srcfile, dstfile):
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        fpath, fname = os.path.split(dstfile)  # 分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)  # 创建路径
        shutil.move(srcfile, dstfile)  # 移动文件
        print("move %s -> %s" % (srcfile, dstfile))

def mycopyfile(srcfile, dstfile):
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        fpath, fname = os.path.split(dstfile)  # 分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)  # 创建路径
        shutil.copyfile(srcfile, dstfile)  # 复制文件
        print("copy %s -> %s" % (srcfile, dstfile))

def GetIdx(root, files, ranks, rank_top = 20):
    with open(files) as f:
        files_list = [file.strip() for file in f.readlines()]

    with open(ranks) as r:
        rank_list = [rank.strip() for rank in r.readlines()]


    top_list = rank_list[0 : rank_top]
    for file in files_list:
        file_ = file.split('/')[-1].split('.')[0]
        if file_ in top_list:
            file_name = file.split('/')[-1]
            new_name = file.split('/')[-3] + file_name
            save_name = os.path.join(root,new_name)
            mycopyfile(file, save_name)
        # print('{} save successful !!!'.format(file_name))

def GetTrainVal(root, ratio):
    train_folder = os.path.join(root, "train")
    if not os.path.exists(train_folder):
        os.mkdir(train_folder)
    val_folder = os.path.join(root, "validation")
    if not os.path.exists(val_folder):
        os.mkdir(val_folder)

    imgs = glob.glob(root + '*.bmp')
    for idx, img in enumerate(imgs):
        if idx<len(imgs)* ratio:
            path1=train_folder
        else:
            path1=val_folder
        name_ = img.split('/')[-1]
        save_name = os.path.join(root, path1, name_)
        mymovefile(img, save_name)


if __name__ == '__main__':
    root = '/home/shimy/FusionData/rank_mri/'
    NC_file = '/home/shimy/FusionData/Slice/NCMRI_Slice.txt'
    AD_file = '/home/shimy/FusionData/Slice/ADMRI_Slice.txt'
    rank_file = '/home/shimy/FusionData/Slice/acc_rank.txt'
    GetIdx(root, NC_file,rank_file, rank_top = 20)
    GetTrainVal(root, 0.8)
    GetIdx(root, AD_file,rank_file, rank_top = 20)
    GetTrainVal(root, 0.8)








