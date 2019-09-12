
import torch,os,torchvision
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
import cv2
import pdb



trainroot = '/home/shimy/FusionData/data_pet/train'
testroot = '/home/shimy/FusionData/data_pet/validation'
IMG_SIZE = 224 # resnet50的输入是224的所以需要将图片统一大小
BATCH_SIZE= 32 #这个批次大小需要占用4.6-5g的显存，如果不够的化可以改下批次，如果内存超过10G可以改为512
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]
CUDA=torch.cuda.is_available()
DEVICE = torch.device("cuda:1 " if CUDA else "cpu")
f=open('a.log','w+')

transform = transforms.Compose([
    transforms.Resize(224),  # 缩放图片(Image)，保持长宽比不变，最短边为224像素
    transforms.CenterCrop(224),  # 从图片中间切出224*224的图片
    transforms.ToTensor(),  # 将图片(Image)转成Tensor，归一化至[0, 1]
    transforms.Normalize(mean=[.5,.5,.5], std=[.5,.5,.5])  # 标准化至[-1, 1]，规定均值和标准差
])


class DogCat(Dataset):
    def __init__(self, root, transforms=None):
        imgs = os.listdir(root)
        self.imgs = [os.path.join(root, img) for img in imgs]
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = 0 if 'AD' in img_path.split('/')[-1] else 1
        data = Image.open(img_path)
        #pic = Image.merge('RGB', (data, data, data))#原始图像为灰度图时使用
        #data1 = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)

        if self.transforms:
            data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)




trainset = DogCat(trainroot,transforms=transform)

trainloader = DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)

testset = DogCat(testroot,transforms=transform)

testloader = DataLoader(testset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)



model_ft = models.vgg16(pretrained=True) # 这里自动下载官方的预训练模型，并且


model_ft.classifier = nn.Sequential(nn.Linear(25088, 4096),      #vgg16
                                 nn.ReLU(),
                                 nn.Dropout(p=0.5),
                                 nn.Linear(4096, 128),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.5),
                                 nn.Linear(128, 2))




model_ft=model_ft.to(DEVICE)# 放到设备中
print(model_ft) # 最后再打印一下新的模型
print(model_ft,file=f)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([    {'params':model_ft.classifier.parameters()}], lr=0.001)#指定 新加的fc层的学习率

def train(model,device, train_loader, epoch,log):
    model.train()
    for batch_idx, data in enumerate(train_loader):
        x,y= data
        x=x.to(device)
        y=y.to(device)
        optimizer.zero_grad()
        y_hat= model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
    print ('Train Epoch: {}\t Loss: {:.6f}'.format(epoch,loss.item()))
    print ('Train Epoch: {}\t Loss: {:.6f}'.format(epoch,loss.item()),file=log)

def test(model, device, test_loader,log):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for i,data in enumerate(test_loader):
            x,y= data
            x=x.to(device)
            y=y.to(device)
            optimizer.zero_grad()
            y_hat = model(x)
            test_loss += criterion(y_hat, y).item() # sum up batch loss
            pred = y_hat.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(y.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testset),
        100. * correct / len(testset)))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testset),
        100. * correct / len(testset)),file=log)

for epoch in range(1, 2000):
    train(model=model_ft,device=DEVICE, train_loader=trainloader,epoch=epoch,log=f)
    test(model=model_ft, device=DEVICE, test_loader=testloader,log=f)
    if epoch  % 100 == 99:
        model_name = 'model/model_{}'.format(epoch)
        torch.save(model_ft.state_dict(), model_name)

f.close()



