# -*- coding: utf-8 -*-
"""
Created on  20191017

@author: linxueya
@description: the custom combined Network
"""

import os,pdb,time,torch
import torch.nn as nn
from torch import optim
from alexnet1 import AlexNet1
from alexnet2 import AlexNet2
from torch.optim import lr_scheduler as LRS



class ALEXNet_com(AlexNet1,AlexNet2):
    def __init__(self):
        AlexNet1.__init__(self)
        AlexNet2.__init__(self)

        self.features1=AlexNet1().features
        self.avgpool1=AlexNet1().avgpool
        self.classifier1=AlexNet1().classifier

        self.features2=AlexNet2().features
        self.avgpool2=AlexNet2().avgpool
        self.classifier2=AlexNet2().classifier

    def forward(self,x1,x2):
        x1 = self.features1(x1)
        x1 = self.avgpool1(x1)
        x1 = x1.view(x1.size(0), 256 * 6 * 6)
        x1 = self.classifier1(x1)

        x2 = self.features2(x2)
        x2 = self.avgpool2(x2)
        x2 = x2.view(x2.size(0), 256 * 6 * 6)
        x2 = self.classifier2(x2)

        x = torch.add(x1,x2)
        return x


# model=ALEXNet_com()
# print(model.features1)



# input1=torch.rand(1,3,224,224 )
# input2=torch.rand(1,3,224,224 )
#
# model1=AlexNet1()
# out1=model1(input1)
#
#
# model = ALEXNet_com()
#
#
#
# cirterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.0002, momentum=0.9)
# lr_scheduler = LRS.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.6)
# label=torch.tensor([1])
# out=model.forward(input1,input2)
# pdb.set_trace()
# loss = cirterion(out, label)
# loss.backward()
# optimizer.step()
# print(out,out1)

