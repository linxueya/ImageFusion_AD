import torch.nn as nn


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # self.a1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        # self.a2 = nn.ReLU(inplace=True)
        # self.a3 = nn.MaxPool2d(kernel_size=3, stride=2)
        # self.a4 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        # self.a5 = nn.ReLU(inplace=True)
        # self.a6 = nn.MaxPool2d(kernel_size=3, stride=2)
        # self.a7 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        # self.a8 = nn.ReLU(inplace=True)
        # self.a9 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        # self.a10 = nn.ReLU(inplace=True)
        # self.a11 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        # self.a12 = nn.ReLU(inplace=True)
        # self.a13 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128*7*3, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 128*7*3)
        x = self.classifier(x)
        return x

