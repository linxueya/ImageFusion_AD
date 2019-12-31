import models
import torch
import torch.nn as nn

input1 = torch.randn(3,1,224,224)
input2 = torch.randn(3,1,224,224)

pool = nn.AdaptiveAvgPool2d(1)
conv = nn.Conv2d(64, 32, kernel_size=1)
activation = nn.Sigmoid()  # todo modify the activation function
att = nn.Sequential(*[pool, conv, activation])
model = getattr(models, 'ACNet34cat')()
print(model)
output = model(input1,input2)

for opt in output:
    print(opt.size())

# input1 = torch.randn(1,3,224,224)
# model_ft = models.resnet50()
# output2 = model_ft(input1)
#
# print(output2.size())



# input = torch.randn(1, 16, 12, 12)
# downsample = nn.Conv2d(16, 16, 3, stride=2, padding=1)
# upsample = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1)
# output  = downsample(input)
# print(output.size())
# output2 = upsample(output)
# print(output2.size())
