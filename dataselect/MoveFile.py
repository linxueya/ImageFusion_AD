# -*- coding: utf-8 -*-
"""
Created on  20190903

@author: linxueya

    文件夹：原图，灰质，白质，脑脊液
    documents: original, gray_matter, white_matter, CSF
    原图：ADNI_002_S_0619_MR_MPR-R__GradWarp__N3_Br_20070411125307309_S15145_I48616.nii
    灰质：mwp1ADNI_002_S_0619_MR_MPR-R__GradWarp__N3_Br_20070411125307309_S15145_I48616.nii
    白质：mwp2ADNI_002_S_0619_MR_MPR-R__GradWarp__N3_Br_20070411125307309_S15145_I48616.nii
    脑脊液：wmADNI_002_S_0619_MR_MPR-R__GradWarp__N3_Br_20070411125307309_S15145_I48616.nii
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


root_path = '/home/shimy/FusionData/MRI_postpro/NC'
save_path= '/home/shimy/FusionData/MRI_postpro/NC_MRI'
# root_path = '/home/shimy/FusionData/subjects/LMCI'
# save_path= '/home/shimy/FusionData/subjects/LMCI_PET'
imgs = os.listdir(root_path)

for img in imgs:
    img1 = os.path.join(root_path,img,'mri')
    # # the source file name of mri
    file='mwp1outputResult.nii'
    new_file='wmmrioutputResult.nii'

    # the source file name of pet
    # file='swp{}.nii'.format(img)
    # new_file='swppetoutputResult.nii'
    file_name=os.path.join(img1,file)
    #['mwp1outputResult.nii', 'mwp2outputResult.nii', 'p0outputResult.nii', 'wmoutputResult.nii']
    save_name=os.path.join(save_path,img,new_file)
    mycopyfile(file_name,save_name)




