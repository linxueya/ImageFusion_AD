#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import re
import datetime


def write_nii_addr(root_path, save_file, last_root_path):
    ### 参数解释
    # root_path: 各个模态的根目录, 如 825_Subject_NC, 该变量不随目录的递归而变化. 用于将.txt文档存放于模态的根目录
    # file_path: 该变量随目录的递归而变化, 直到找到.nii为止.
    # original_doc, gray_matter_doc, white_matter_doc, CSF_doc:
    # lable: 表示模态所属的类别, 包括AD, NC, pMCI, sMCI, uMCI

    # 遍历 file_path 下所有文件, 包括子目录
    files = os.listdir(root_path)
    # _file_path = root_path
    for file_name in files:
        next_root_path = os.path.join(root_path, file_name)
        if os.path.isdir(next_root_path):
            last_root_path = root_path
            root_path = next_root_path
            write_nii_addr(root_path, save_file, last_root_path)
            selected_path = select_slice_path(root_path)
            if (selected_path != "NONE"):
                save_file.writelines(selected_path + "\n")

            root_path = last_root_path  # 递归遍历的回馈 - feed-back




def select_slice_path(file_path):
    satisified_path = ['Xslice', 'Yslice', 'Zslice']
    target_file = "NONE"
    for item in satisified_path:
        if item in file_path:
            print("file_path = {}".format(file_path))
            target_file = file_path
            break

    return target_file


def execute(root_path, save_file_name):
    save_file_path = os.path.join(root_path, save_file_name)
    # print("save_file_path = {}".format(save_file_path))
    if os.path.exists(save_file_path):
        i = datetime.datetime.now()
        date = str(i.year) + str(i.month) + str(i.day) + str(i.hour) + str(i.minute) + str(i.second)
        new_name = save_file_path + date +".bak"
        os.rename(save_file_path, new_name)
        print("copied and deleted file, new_name = {}".format(new_name))
    # os.remove(save_file_path)

    with open(save_file_path, "a") as save_file:
        write_nii_addr(root_path, save_file, "")
    print("DONE... root_path = {}".format(root_path))


# 递归遍历/root目录下所有文件
if __name__ == "__main__":

    root_path = 'D:\Master\FusionData\AD_JPG'
    save_file_name =  'AD_gray_matter_Slices_path.txt'
    execute(root_path, save_file_name)

