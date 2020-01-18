import torch
from torch import nn
import math
import ipdb
import torch.utils.model_zoo as model_zoo
from .basic_module import BasicModule

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class ACNet34se(BasicModule):
    def __init__(self, num_classes=4, pretrained=False):
        super(ACNet34se, self).__init__()

        self.model_name = 'ACNet34se'

        layers = [3, 4, 6, 3]
        block = ResidualBlock
        # mri image branch
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2) # use PSPNet extractors
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # pet image branch
        self.inplanes = 64
        self.conv1_d = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1_d = nn.BatchNorm2d(64)
        self.relu_d = nn.ReLU(inplace=True)
        self.maxpool_d = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1_d = self._make_layer(block, 64, layers[0])
        self.layer2_d = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3_d = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_d = self._make_layer(block, 512, layers[3], stride=2)

        # attention branch
        self.atten_mri_0 = self.channel_attention(64)
        self.atten_pet_0 = self.channel_attention(64)
        self.atten_mri_1 = self.channel_attention(64*block.expansion)
        self.atten_pet_1 = self.channel_attention(64*block.expansion)
        self.atten_mri_2 = self.channel_attention(128*block.expansion)
        self.atten_pet_2 = self.channel_attention(128*block.expansion)
        self.atten_mri_3 = self.channel_attention(256*block.expansion)
        self.atten_pet_3 = self.channel_attention(256*block.expansion)
        self.atten_mri_4 = self.channel_attention(512*block.expansion)
        self.atten_pet_4 = self.channel_attention(512*block.expansion)

        # cat conv branch 
        self.catconv0 = self._make_catconv(64)
        self.catconv1 = self._make_catconv(64*block.expansion)
        self.catconv2 = self._make_catconv(128*block.expansion)
        self.catconv3 = self._make_catconv(256*block.expansion)
        self.catconv4 = self._make_catconv(512*block.expansion)

        # merge branch
        self.inplanes = 64
        self.maxpool_m = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1_m = self._make_layer(block, 64, layers[0])
        self.layer2_m = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3_m = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_m = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # weight initial
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if pretrained:
            self._load_resnet_pretrained()

    def forward(self, mri, pet, phase_checkpoint=False):
        mri = self.conv1(mri)
        mri = self.bn1(mri)
        mri = self.relu(mri)
        pet = self.conv1_d(pet)
        pet = self.bn1_d(pet)
        pet = self.relu_d(pet)
        attn_mri = self.atten_mri_0(mri)
        attn_pet = self.atten_pet_0(pet)
        m0 = self.netcat(None, self.catconv0, mri, attn_mri, pet, attn_pet)

        mri = self.maxpool(mri)
        pet = self.maxpool_d(pet)
        m = self.maxpool_m(m0)

        # block 1
        mri = self.layer1(mri)
        pet = self.layer1_d(pet)
        m = self.layer1_m(m)

        attn_mri = self.atten_mri_1(mri)
        attn_pet = self.atten_pet_1(pet)
        m1 = self.netcat(m, self.catconv1, mri, attn_mri, pet, attn_pet)

        # block 2
        mri = self.layer2(mri)
        pet = self.layer2_d(pet)
        m = self.layer2_m(m1)

        attn_mri = self.atten_mri_2(mri)
        attn_pet = self.atten_pet_2(pet)
        m2 = self.netcat(m, self.catconv2, mri, attn_mri, pet, attn_pet)

        # block 3
        mri = self.layer3(mri)
        pet = self.layer3_d(pet)
        m = self.layer3_m(m2)

        attn_mri = self.atten_mri_3(mri)
        attn_pet = self.atten_pet_3(pet)
        m3 = self.netcat(m, self.catconv3, mri, attn_mri, pet, attn_pet)

        # block 4
        mri = self.layer4(mri)
        pet = self.layer4_d(pet)
        m = self.layer4_m(m3)

        attn_mri = self.atten_mri_4(mri)
        attn_pet = self.atten_pet_4(pet)
        m4 = self.netcat(m, self.catconv4, mri, attn_mri, pet, attn_pet)

        # fc
        m5 = self.avgpool(m4)
        m5 = m5.reshape(m5.size(0), -1)
        m5 = self.fc(m5)
        return m5  # channel of m5 is 512

    def netcat(self, branch0, conv, branch1, atten1, branch2, atten2):
        branch = torch.cat([branch1.mul(atten1), branch2.mul(atten2)], 1)
        branch = conv(branch)
        if branch0 is not None:
            branch = branch + branch0
            branch = torch.cat([branch0, branch], 1)
            branch = conv(branch)
        return branch

    def _make_catconv(self, num_channel):
        cond1 = nn.Conv2d(num_channel * 2, num_channel, kernel_size=1)
        return cond1

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def channel_attention(self, num_channel, ablation=False):
        pool = nn.AdaptiveAvgPool2d(1)
        conv = nn.Conv2d(num_channel, num_channel, kernel_size=1)
        bn = nn.BatchNorm2d(num_channel)
        activation = nn.ReLU(inplace=True)
        return nn.Sequential(*[pool, conv, bn, activation])

    def _load_resnet_pretrained(self):
        pretrain_dict = model_zoo.load_url(model_urls['resnet34'])
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            # print('%%%%% ', k)
            if k in state_dict:
                if k.startswith('conv1'):
                    model_dict[k] = v
                    # print('##### ', k)
                    model_dict[k.replace('conv1', 'conv1_d')] = torch.mean(v, 1).data. \
                        view_as(state_dict[k.replace('conv1', 'conv1_d')])

                elif k.startswith('bn1'):
                    model_dict[k] = v
                    model_dict[k.replace('bn1', 'bn1_d')] = v
                elif k.startswith('layer'):
                    model_dict[k] = v
                    model_dict[k[:6]+'_d'+k[6:]] = v
                    model_dict[k[:6]+'_m'+k[6:]] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


class ResidualBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(ResidualBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        # main branch
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        # attention branch
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(in_features=planes, out_features=round(planes / 16))
        self.fc2 = nn.Linear(in_features=round(planes / 16), out_features=planes)
        self.sigmoid = nn.Sigmoid()
        if planes == 64:
            self.AvgPool = nn.AvgPool2d(56, stride=1)
        elif planes == 128:
            self.AvgPool = nn.AvgPool2d(28, stride=1)
        elif planes == 256:
            self.AvgPool = nn.AvgPool2d(14, stride=1)
        elif planes == 512:
            self.AvgPool = nn.AvgPool2d(7, stride=1)

    def forward(self, x):
        identity = x
        # main branch
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        original_out = out

        # attention branch
        out = self.AvgPool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0), out.size(1), 1, 1)
        out = out * original_out

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=dilation,
                               padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

