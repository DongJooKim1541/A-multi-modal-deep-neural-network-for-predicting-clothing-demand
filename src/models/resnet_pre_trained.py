import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from torchvision.models import resnet18, resnet34, vgg16

resnet = resnet18(pretrained=True)


class ResNet(nn.Module):
    """Multi-task ResNet18 fusion model for clothing demand prediction."""

    def __init__(self) -> None:
        super(ResNet, self).__init__()
        self.num_best_sex_class = 3
        self.num_best_age_class = 7
        self.num_sales_class = 7

        resnet_modules = list(resnet.children())[:-3]
        self.cnn1 = nn.Sequential(*resnet_modules)
        self.cnn2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm1d(512 + 768 + 3)
        self.linear1 = nn.Linear(512 + 768 + 3, 64)
        self.linear_best_sex = nn.Linear(64, self.num_best_sex_class)
        self.linear_best_age = nn.Linear(64, self.num_best_age_class)
        self.linear_view = nn.Linear(64, 1)
        self.linear_sales = nn.Linear(64, 1)

    def forward(self, image: torch.Tensor, sex: torch.Tensor, price: torch.Tensor,
                category: torch.Tensor, bert_feature: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Forward pass for multi-task prediction."""
        out = self.cnn1(image)
        out = self.cnn2(out)
        out = F.avg_pool2d(out, 4)
        conv_out = out.view(out.size(0), -1)
        out_concat = torch.cat((conv_out, sex, price, category, bert_feature), dim=1)
        out_concat = self.bn1(out_concat)
        out = self.linear1(out_concat)
        out = self.dropout(out)
        out_best_sex = self.linear_best_sex(out)
        out_best_age = self.linear_best_age(out)
        out_view = self.linear_view(out)
        out_sales = self.linear_sales(out)
        return out_best_sex, out_best_age, out_view, out_sales


def weights_init(m: nn.Module) -> None:
    """Initialize model weights using Xavier uniform for linear/conv layers."""
    classname = m.__class__.__name__
    if type(m) in [nn.Linear, nn.Conv2d, nn.Conv1d]:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)