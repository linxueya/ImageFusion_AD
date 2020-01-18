import models
import torch
import ipdb
import torch.nn as nn

input1 = torch.randn(1,3,224,224)
input2 = torch.randn(1,1,224,224)


model = getattr(models, 'AlexNet')()
print(input1)
output = model(input1)
ipdb.set_trace()
for opt in output:
    print(opt.size())


