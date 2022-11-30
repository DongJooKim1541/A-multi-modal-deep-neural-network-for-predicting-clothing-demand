import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
import math


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.num_best_sex_class = 3
        self.num_best_age_class = 7
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(32)

        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)

        self.shortcut = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64))

        self.dropout1 = nn.Dropout2d(0.5)

        self.bn5 = nn.BatchNorm1d(14400 + 3)
        self.linear1 = nn.Linear(14400 + 3, 4096)
        self.linear2 = nn.Linear(4096, 512)
        self.linear3 = nn.Linear(512, 32)

        self.linear_best_sex = nn.Linear(32, self.num_best_sex_class)
        self.linear_best_age = nn.Linear(32, self.num_best_age_class)
        self.linear_view = nn.Linear(32, 1)
        self.linear_sales = nn.Linear(32, 1)

    def forward(self, image, sex, price, category):
        out = F.relu(self.bn1(self.conv1(image)))  # torch.Size([BATCH, 64, 125, 125])
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        out = F.relu(self.bn4(self.conv4(out)))

        out += self.shortcut(image)  # skip connection 구현
        out = F.relu(out)
        # print(out.shape)
        out = self.dropout1(out)
        out = F.avg_pool2d(out, 8)
        out = self.dropout1(out)

        conv_out = out.view(out.size(0), -1)
        out_concat = torch.cat((conv_out, sex, price, category), dim=1)
        # print(out_concat.shape)

        out_concat = self.bn5(out_concat)
        out = self.linear1(out_concat)
        out = self.dropout1(out)
        out = self.linear2(out)
        out = self.dropout1(out)
        out = self.linear3(out)
        out = self.dropout1(out)
        out_best_sex = self.linear_best_sex(out)
        out_best_age = self.linear_best_age(out)
        out_view = self.linear_view(out)
        out_sales = self.linear_sales(out)

        return out_best_sex, out_best_age, out_view, out_sales


def weights_init(m):
    classname = m.__class__.__name__
    if type(m) in [nn.Linear, nn.Conv2d, nn.Conv1d]:
        torch.nn.init.xavier_uniform_(m.weight)  # weight initialization
        if m.bias is not None:
            m.bias.data.fill_(0.01)  # initializes the bias to 0.01
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)  # initialize the weight to mean 1.0, deviation 0.02
        m.bias.data.fill_(0)  # initializes the bias to 0
