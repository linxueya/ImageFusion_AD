#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import shutil
import re
import datetime
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

#  /home/shimy/FusionData/Subject_AD/mri_bmp/AD000/Zslice/Z085.bmp
def GetIdx(root, files):
    with open(files) as f:
        files_list = [file.strip() for file in f.readlines()]
    for file in files_list:
        file_name = file.split('/')[-1]
        file_path = file_name.split('.')[0]
        new_name = file.split('/')[-3] + file_name
        save_name = os.path.join(root,file_path,new_name)
        mycopyfile(file, save_name)
        # print('{} save successful !!!'.format(file_name))


def GetDataSet(root, file1, file2):
    GetIdx(root, file1)
    GetIdx(root, file2)
    for root_path in os.listdir(root):
        files = os.listdir(os.path.join(root, root_path))
        for idx,file in enumerate(files):
            train_folder = os.path.join(root,root_path, "train")
            if not os.path.exists(train_folder):
                os.mkdir(train_folder)
            val_folder = os.path.join(root, root_path, "validation")
            if not os.path.exists(val_folder):
                os.mkdir(val_folder)

            if idx < len(files) * 0.8:
                path1 = train_folder
            else:
                path1 = val_folder

            path2 = 'AD' if "AD" in file.split('/')[-1] else 'NC'
            save_path = os.path.join(path1, path2)
            save_name = os.path.join(save_path, file)
            source_name = os.path.join(root,root_path,file)
            mymovefile(source_name, save_name)
            # print("successful create ->" + save_name)


if __name__ == '__main__':
    root = '/home/shimy/FusionData/ADNC'
    NC_file_Z = '/home/shimy/FusionData/Slice/NCMRI_ZSlice.txt'
    AD_file_Z = '/home/shimy/FusionData/Slice/ADMRI_ZSlice.txt'
    GetDataSet(root, NC_file_Z, AD_file_Z)

