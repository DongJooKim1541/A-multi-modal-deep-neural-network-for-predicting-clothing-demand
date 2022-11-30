import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, vgg16

resnet = resnet18(pretrained=True)

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.num_best_sex_class = 3
        self.num_best_age_class = 7
        self.num_sales_class = 7
        # self.num_sales_class = 6

        resnet_modules = list(resnet.children())[:-3]

        self.cnn1 = nn.Sequential(*resnet_modules)

        self.cnn2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.dropout=nn.Dropout(0.5)

        self.bn1 = nn.BatchNorm1d(512+768+3)

        self.linear1 = nn.Linear(512+768+3, 64)

        self.linear_best_sex = nn.Linear(64, self.num_best_sex_class)
        self.linear_best_age = nn.Linear(64, self.num_best_age_class)
        self.linear_view = nn.Linear(64, 1)
        self.linear_sales = nn.Linear(64, 1)


    def forward(self,image,sex,price,category, bert_feature):
        out = self.cnn1(image)
        out = self.cnn2(out)
        out = F.avg_pool2d(out, 4) # torch.Size([16, 512, 8, 8])

        conv_out = out.view(out.size(0), -1) # torch.Size(
        out_concat=torch.cat((conv_out,sex,price,category, bert_feature),dim=1) # torch.Size([16, 4099])
        #print(out_concat.shape)
        out_concat=self.bn1(out_concat)
        out = self.linear1(out_concat) # torch.Size([16, 512])
        #print(out.shape)
        out=self.dropout(out)
        out_best_sex = self.linear_best_sex(out) # torch.Size([16, 3])
        out_best_age = self.linear_best_age(out) # torch.Size([16, 7])
        out_view = self.linear_view(out) # torch.Size([16, 8])
        out_sales = self.linear_sales(out) # torch.Size([16, 6])
        return out_best_sex,out_best_age,out_view,out_sales

def weights_init(m):
    classname = m.__class__.__name__
    if type(m) in [nn.Linear, nn.Conv2d, nn.Conv1d]:
        torch.nn.init.xavier_uniform_(m.weight) # weight initialization
        if m.bias is not None:
            m.bias.data.fill_(0.01) # initializes the bias to 0.01
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02) # initialize the weight to mean 1.0, deviation 0.02
        m.bias.data.fill_(0) # initializes the bias to 0