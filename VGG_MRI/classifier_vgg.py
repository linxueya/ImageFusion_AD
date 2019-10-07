import os
import time
import torch
import torch.nn as nn
from torchvision import models
from torch import optim
from torch.optim import lr_scheduler as LRS
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import config
from alexnet import AlexNet
from vgg_custom import VGG16Net

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
device = 'cuda'


class Classifier:
    def __init__(self, config):
        torch.cuda.empty_cache()
        self.config = config
        train_set = ImageFolder(os.path.join(self.config.data_dir, 'train'),
                                transform=T.Compose([T.RandomHorizontalFlip(),
                                                     T.Resize(self.config.input_size),
                                                     T.ToTensor(),
                                                     T.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225])]))

        val_set = ImageFolder(os.path.join(self.config.data_dir, 'validation'),
                              transform=T.Compose([T.Resize(self.config.input_size),
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
        num = self.config.num_batch_print
        total = int(len(self.train_loader)/num)*num
        model = VGG16Net()
        model.to(device)
        cirterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=self.config.lr, momentum=self.config.momentum)
        lr_scheduler = LRS.MultiStepLR(optimizer, milestones=self.config.milestones, gamma=self.config.gamma)

        for epoch in range(self.config.epochs):
            lr = 0.0
            running_loss = 0.0
            lr_scheduler.step(epoch=epoch)
            for i, data in enumerate(self.train_loader):
                optimizer.zero_grad()
                img, label = data
                img = img.to(device)
                label = label.to(device)
                oup = model(img)
                loss = cirterion(oup, label)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                for params_group in optimizer.param_groups:
                    lr = params_group['lr']

                if i % num == num-1:
                    localtime = time.strftime('%H:%M:%S', time.localtime(time.time()))
                    log = '{} Train Epoch[{}] Iteration[{}/{}]  Loss:{:.10f}  LR:{:.5f}'.format(
                        localtime, epoch,  i+1, total, running_loss / num, lr)
                    print(log)
                    f = open('./log.txt', 'a')
                    f.write(log+'\n')
                    f.close()
                    running_loss = 0.0
            if epoch % 20 == 19:
                torch.save(model.state_dict(), './model/vggcm2_'+str(epoch+1)+'.pth')

    def evaluate(self):
        model = VGG16Net()
        model.load_state_dict(torch.load(self.config.model_path, map_location='cpu'))
        model.to(device)
        model.eval()

        with torch.no_grad():
            correct = 0
            num = 0
            for data in self.val_loader:
                images, labels = data
                num += len(labels)
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predict = torch.max(outputs, 1)
                correct += sum(labels == predict).cpu().numpy()
            print('\nTest set:  Accuracy: {}/{} ({:.0f}%)\n'.format(
                 correct, num,100. * correct / num))

if __name__ == '__main__':

    classifier = Classifier(config)
    classifier.train()
    classifier.evaluate()
