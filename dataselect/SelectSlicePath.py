#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import shutil
import re
import datetime
import ipdb


file = './acc.txt'
save_file = './acc_rank.txt'
with open(file) as f:
    files_list = [file.strip() for file in f.readlines()]
    num = [file.split(' ')[-3] for file in files_list]
    acc = [file.split(' ')[-1] for file in files_list]
    num_acc = dict((num_,acc_) for num_,acc_ in zip(num, acc))
    num_rank = sorted(num_acc.items(),key= lambda x:x[1],reverse=True)
    num_rank20 = num_rank[0 : 20]

    num_rank30 = num_rank[0 : 30]

    for key, vaule in num_rank:
        with open(save_file, 'a') as r1:
            r1.write(key + '\n' )
    ipdb.set_trace()




