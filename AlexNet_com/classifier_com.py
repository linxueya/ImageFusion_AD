# -*- coding: utf-8 -*-
"""
Created on  20191018

@author: linxueya
@description: the custom combined Network
"""


import os, time, torch, pdb
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler as LRS
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from PIL import Image
import config
from custom1 import ALEXNet_com


CUDA = torch.cuda.is_available()
device = torch.device("cuda:1" if CUDA else "cpu")


class multi(Dataset):
    def __init__(self, root, root2, transforms=None):
        imgs = os.listdir(root)
        self.imgs = [os.path.join(root, img) for img in imgs]
        self.imgs2 = [os.path.join(root2, img) for img in imgs]
        self.transforms = transforms

    def __getitem__(self, index):
        img_path1 = self.imgs[index]
        img_path2 = self.imgs2[index]
        label = 0 if 'NC' in img_path1.split('/')[-1] else 1
        data1 = Image.open(img_path1)
        data2 = Image.open(img_path2)

        if self.transforms:
            data1, data2 = self.transforms(data1), self.transforms(data2)
        return data1, data2, label

    def __len__(self):
        return len(self.imgs)


class Classifier():
    def __init__(self, config):
        torch.cuda.empty_cache()
        self.config = config

        root1 = os.path.join(self.config.data_mri, 'train')
        root2 = os.path.join(self.config.data_mri, 'validation')
        root3 = os.path.join(self.config.data_pet, 'train')
        root4 = os.path.join(self.config.data_pet, 'validation')

        train_set = multi(root1, root3, transforms = T.Compose([T.RandomHorizontalFlip(),
                                                     T.Resize(self.config.input_size),
                                                     T.ToTensor(),
                                                     T.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225])]))

        val_set = multi(root2, root4, transforms = T.Compose([T.Resize(self.config.input_size),
                                                   T.ToTensor(),
                                                  T.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])]))

        self.train_loader = DataLoader(train_set,
                                       batch_size=self.config.batch_size,
                                       shuffle=True,
                                       num_workers=4,
                                       drop_last=False
                                       )

        self.val_loader = DataLoader(val_set,
                                     batch_size=self.config.batch_size,
                                     shuffle=False,
                                     num_workers=4,
                                     drop_last=False)

        print(time.asctime(time.localtime(time.time())))
        print('initing complete...')

    def train(self):
        num = self.config.num_print
        total = int(len(self.train_loader) / num) * num
        model = ALEXNet_com()
        print(model)
        model.to(device)
        cirterion = nn.CrossEntropyLoss()
        optimizer = optim.RMSprop(model.parameters(), lr=self.config.lr,
                                  weight_decay=1e-5, momentum=self.config.momentum, eps=1e-2)
        print(optimizer)
        lr_scheduler = LRS.MultiStepLR(optimizer, milestones=self.config.milestones, gamma=self.config.gamma)

        for epoch in range(self.config.epochs):
            lr = 0.0
            running_loss = 0.0
            lr_scheduler.step(epoch = epoch)
            for idx, data in enumerate(self.train_loader):
                optimizer.zero_grad()
                x1, x2, y = data
                x1 = x1.to(device)
                x2 = x2.to(device)
                y = y.to(device)
                y_hat = model(x1, x2)
                loss = cirterion(y_hat, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                lr = optimizer.param_groups[0]['lr']

                if idx % num == num-1:
                    localtime = time.strftime('%H:%M:%S', time.localtime(time.time()))
                    log = '{} Train set:Epoch[{}] Iter[{}/{}]  Loss:{:.10f}  Lr:{:.5f}'.format(
                        localtime, epoch,  idx+1, total, running_loss / num, lr)
                    print(log)
                    f = open('./log.txt', 'a')
                    f.write(log+'\n')
                    f.close()
                    running_loss = 0.0
            if epoch % 20 == 19:
                torch.save(model.state_dict(), './model/vgg_'+str(epoch+1)+'.pth')

    def evaluate(self):
        model = ALEXNet_com()
        model.load_state_dict(torch.load(self.config.model_path))
        model.to(device)
        model.eval()

        with torch.no_grad():
            num = 0
            correct = 0
            for data in self.val_loader:
                x1, x2, y = data
                num += len(y)
                x1 = x1.to(device)
                x2 = x2.to(device)
                y = y.to(device)
                y_hat = model(x1, x2)
                pred = y_hat.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(y.view_as(pred)).sum().item()
            print('\nTest set:  Accuracy: {}/{} ({:.0f}%)\n'.format(
                correct, num, 100. * correct / num))


if __name__ == '__main__':
    classifier = Classifier(config)
    classifier.train()
    classifier.evaluate()
