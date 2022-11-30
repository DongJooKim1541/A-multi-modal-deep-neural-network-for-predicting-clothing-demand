import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
import math

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNet, self).__init__()
        self.num_best_sex_class=3
        self.num_best_age_class = 7
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 2, stride=1)
        self.layer2 = self._make_layer(32, 2, stride=2)

        self.dropout1=nn.Dropout(0.5)

        self.bn1 = nn.BatchNorm1d(7200+3)

        self.linear1 = nn.Linear(7200+3, 2048)
        self.linear2 = nn.Linear(2048, 512)
        self.linear3 = nn.Linear(512, 32)
        self.linear_best_sex = nn.Linear(32, self.num_best_sex_class)
        self.linear_best_age = nn.Linear(32, self.num_best_age_class)
        self.linear_view = nn.Linear(32, 1)
        self.linear_sales = nn.Linear(32, 1)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self,image,sex,price,category):
        out = F.relu(self.bn1(self.conv1(image))) # torch.Size([16, 64, 250, 250])
        out = self.layer1(out) # torch.Size([16, 64, 250, 250])
        out = self.layer2(out) # torch.Size([16, 64, 250, 250])
        out = self.layer3(out) # torch.Size([16, 64, 250, 250])
        out = F.avg_pool2d(out, 4) # torch.Size([16, 512, 8, 8])

        conv_out = out.view(out.size(0), -1) # torch.Size([16, 32768])
        out_concat=torch.cat((conv_out,sex,price,category),dim=1) # torch.Size([16, 4099])
        #print(out_concat.shape)
        out_concat=self.bn1(out_concat)
        out = self.linear1(out_concat) # torch.Size([16, 512])
        out= self.dropout1(out)
        out = self.linear2(out)
        out = self.dropout1(out)
        out = self.linear3(out)
        out = self.dropout1(out)

        out_best_sex = self.linear_best_sex(out) # torch.Size([16, 3])
        out_best_age = self.linear_best_age(out) # torch.Size([16, 7])
        out_view=self.linear_view(out)
        out_sales = self.linear_sales(out) # torch.Size([16, 6])
        return out_best_sex,out_best_age,out_view, out_sales

def weights_init(m):
    classname = m.__class__.__name__
    if type(m) in [nn.Linear, nn.Conv2d, nn.Conv1d]:
        torch.nn.init.xavier_uniform_(m.weight) # weight initialization
        if m.bias is not None:
            m.bias.data.fill_(0.01) # initializes the bias to 0.01
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02) # initialize the weight to mean 1.0, deviation 0.02
        m.bias.data.fill_(0) # initializes the bias to 0