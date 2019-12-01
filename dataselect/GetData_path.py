#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import re
import time
import datetime
import pdb


def write_nii_addr(root_path, file_path, mri_doc,pet_doc, lable):
    ### 参数解释
    # root_path: 各个模态的根目录, 如 825_Subject_NC, 该变量不随目录的递归而变化. 用于将.txt文档存放于模态的根目录
    # file_path: 该变量随目录的递归而变化, 直到找到.nii为止.
    # original_doc, gray_matter_doc, white_matter_doc, CSF_doc:
    # lable: 表示模态所属的类别, 包括AD, NC, pMCI, sMCI, uMCI

    # 遍历 file_path 下所有文件, 包括子目录
    files = os.listdir(file_path)
    for file_name in files:
        _file_path = os.path.join(file_path, file_name)
        if os.path.isdir(_file_path):
            write_nii_addr(root_path, _file_path, mri_doc, pet_doc, lable)
        else:
            postfix = file_name.split('.')[1]
            if (postfix == "nii"):
                pre_fix = file_name.split('.')[0]
                # gray_matter

                if pre_fix.find("mri")>=0:
                    _name = lable + "_mri.txt"
                    with open(os.path.join(root_path, _name), "a") as f:
                        f.writelines(_file_path + "\n")

                # original
                else:
                    _name = lable + "_pet.txt"
                    with open(os.path.join(root_path, _name), "a") as f:
                        f.writelines(_file_path + "\n")

            # print(os.path.join(file_path))


def create_modal_file(root_path, file_path, lable):
    # 文件夹：原图，灰质，白质，脑脊液
    # documents: original, gray_matter, white_matter, CSF
    # 原图：ADNI_002_S_0619_MR_MPR-R__GradWarp__N3_Br_20070411125307309_S15145_I48616.nii
    # 灰质：mwp1ADNI_002_S_0619_MR_MPR-R__GradWarp__N3_Br_20070411125307309_S15145_I48616.nii
    # 白质：mwp2ADNI_002_S_0619_MR_MPR-R__GradWarp__N3_Br_20070411125307309_S15145_I48616.nii
    # 脑脊液：wmADNI_002_S_0619_MR_MPR-R__GradWarp__N3_Br_20070411125307309_S15145_I48616.nii
    mri_doc = os.path.join(root_path, "mri_bmp")
    pet_doc = os.path.join(root_path, "pet_bmp")


    if not os.path.exists(mri_doc):
        print("Create file original_doc = {}".format(mri_doc))
        os.makedirs(mri_doc)

    if not os.path.exists(pet_doc):
        print("Create file gray_matter_doc = {}".format(pet_doc))
        os.makedirs(pet_doc)


    # 预先进行备份当前根目录下所有.txt文档并删除它们
    import shutil
    backup_file = os.path.join(root_path, "backup")
    i = datetime.datetime.now()
    date = str(i.year) + str(i.month) + str(i.day)
    if not os.path.exists(backup_file):
        print("Create file backup_file = {}".format(backup_file))
        os.makedirs(backup_file)
    files = os.listdir(root_path)
    for file in files:
        print("[backup] file = {}".format(file))
        if not os.path.isdir(file):
            if (len(file.split('.')) > 1):
                if (file.split('.')[1] == "txt"):
                    old_name = file
                    new_name = date + "_" + str(file)
                    print("old_name = {}".format(old_name))
                    print("new_name = {}".format(new_name))
                    os.rename(os.path.join(root_path, old_name), os.path.join(root_path, new_name))
                    source_dir = os.path.join(root_path, new_name)
                    target_dir = os.path.join(root_path, "backup")
                    shutil.copy(source_dir, target_dir)
                    os.remove(source_dir)

    # 逻辑程序
    write_nii_addr(root_path, file_path, mri_doc, pet_doc, lable)


# 递归遍历/root目录下所有文件
if __name__ == "__main__":
    root_path_AD = '/home/shimy/FusionData/Subject_AD'
    file_path = '/home/shimy/FusionData/MRI_postpro/AD_MRI'
    create_modal_file(root_path_AD, file_path,"AD")

    root_path_NC = '/home/shimy/FusionData/Subject_NC'
    file_path = '/home/shimy/FusionData/MRI_postpro/NC_MRI'
    create_modal_file(root_path_NC, file_path,"NC")

    root_path_EMCI = '/home/shimy/FusionData/Subject_EMCI'
    file_path = '/home/shimy/FusionData/MRI_postpro/EMCI_MRI'
    create_modal_file(root_path_EMCI, file_path,"EMCI")

    root_path_LMCI = '/home/shimy/FusionData/Subject_LMCI'
    file_path = '/home/shimy/FusionData/MRI_postpro/LMCI_MRI'
    create_modal_file(root_path_LMCI, file_path,"LMCI")

    # root_path_AD = '/home/shimy/FusionData/Subject_AD'
    # file_path = '/home/shimy/FusionData/subjects/AD_PET'
    # create_modal_file(root_path_AD, file_path,"AD")
    #
    # root_path_NC = '/home/shimy/FusionData/Subject_NC'
    # file_path = '/home/shimy/FusionData/subjects/NC_PET'
    # create_modal_file(root_path_NC, file_path,"NC")
    #
    # root_path_EMCI = '/home/shimy/FusionData/Subject_EMCI'
    # file_path = '/home/shimy/FusionData/subjects/EMCI_PET'
    # create_modal_file(root_path_EMCI, file_path,"EMCI")
    #
    # root_path_LMCI = '/home/shimy/FusionData/Subject_LMCI'
    # file_path = '/home/shimy/FusionData/subjects/LMCI_PET'
    # create_modal_file(root_path_LMCI, file_path,"LMCI")

