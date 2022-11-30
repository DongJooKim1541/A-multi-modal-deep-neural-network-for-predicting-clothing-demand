import os
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import shoppingDataset_loader
from config import *
from models import resnet_pre_trained
from shoppingDataset_loader import *
from torch import nn

from transformers import BertTokenizer, BertModel

""" Device Confirmation """
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print('Using PyTorch version: ', torch.__version__)

# implement bert
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
net_bert = BertModel.from_pretrained("bert-base-multilingual-cased")


def get_bert_feature(clothing, tokenizer, net_bert):
    encoded_input = tokenizer(clothing, return_tensors='pt')
    output = net_bert(**encoded_input)
    # print(output[1].shape)

    output = output[1].clone().detach()
    # print(output.dtype)
    return output

# get bert features
csv_info = []
df = pd.read_csv(csv_path, encoding='cp949')
# slice[0]:  index
# slice[1]:  1
# slice[2]:  goodsNum
# slice[3]:  1220731
# slice[4]:  clothing_name
# slice[5]:  사피아노신세틱레더벨트
print("len(df): ", len(df))
for i in range(0, len(df)):
    slice = str(df.loc[i]).split()
    csv_info_dict = {}
    csv_info_dict[slice[1]] = get_bert_feature(slice[5], tokenizer, net_bert)
    #print(get_bert_feature(slice[5], tokenizer, net_bert).dtype)
    csv_info.append(csv_info_dict)
    if i % 1000 == 0:
        print(str(i) + "th file feature extraction")

"""Data preprocessing"""
trainset = ShoppingDataset(csv_info, train=True)
testset = ShoppingDataset(csv_info, train=False)
train_loader = DataLoader(trainset, batch_size=batch_size, num_workers=4, shuffle=True)
test_loader = DataLoader(testset, batch_size=batch_size, num_workers=0, shuffle=False)

"""Confirm data"""
print("len(train_loader), len(test_loader): ", len(train_loader), len(test_loader))

criterion = nn.CrossEntropyLoss()
criterion_for_focal = nn.CrossEntropyLoss(reduction='none')
criterion_regression = nn.MSELoss()

# train, test list 담기
train_epoch_acc = []
train_epoch_best_sex_acc = []
train_epoch_best_age_acc = []

train_epoch_loss = []
train_epoch_best_sex_loss = []
train_epoch_best_age_loss = []
train_epoch_view_loss = []
train_epoch_sales_loss = []

test_epoch_acc = []
test_epoch_best_sex_acc = []
test_epoch_best_age_acc = []

test_epoch_loss = []
test_epoch_best_sex_loss = []
test_epoch_best_age_loss = []
test_epoch_view_loss = []
test_epoch_sales_loss = []


def train(model, train_loader, optimizer):
    model.train()

    train_best_sex_loss_list = []
    train_best_age_loss_list = []
    train_view_loss_list = []
    train_sales_loss_list = []


    correct_best_sex = 0
    correct_best_age = 0


    for image, sex, label_best_sex, label_best_age, label_view, label_sales, price, category, clothing_feature, _ in train_loader:
        optimizer.zero_grad()
        sex = sex.clone().detach().unsqueeze(1)
        label_best_sex = label_best_sex.clone().detach().cuda()
        label_best_age = label_best_age.clone().detach().cuda()
        label_view = label_view.clone().detach().cuda().unsqueeze(1)
        label_sales = label_sales.clone().detach().cuda().unsqueeze(1)
        price = price.clone().detach().unsqueeze(1)
        category = category.clone().detach().unsqueeze(1)

        bert_feature_batch = get_bert_feature_by_batch(clothing_feature)
        #print("train bert_feature_batch: ",bert_feature_batch)
        preds = model(image.cuda(), sex.cuda(), price.cuda(), category.cuda(), bert_feature_batch.cuda())  # torch.Size([BATCH_SIZE, NUM_CLASSES])

        if analysis=="best_sex":
            best_sex_prediction = preds.max(1, keepdim=True)[1]
            correct_best_sex += best_sex_prediction.eq(label_best_sex.view_as(best_sex_prediction)).sum().item()

            train_iteration_best_sex_loss = criterion(preds, label_best_sex)

            train_iteration_total_loss = train_iteration_best_sex_loss

            train_best_sex_loss_list.append(train_iteration_best_sex_loss.item())
        elif analysis=="best_age":
            best_age_prediction = preds.max(1, keepdim=True)[1]
            correct_best_age += best_age_prediction.eq(label_best_age.view_as(best_age_prediction)).sum().item()

            train_iteration_best_age_loss = criterion_for_focal(preds, label_best_age)
            pt = torch.exp(-train_iteration_best_age_loss)
            train_iteration_best_age_loss = (alpha * (1 - pt) ** gamma * train_iteration_best_age_loss).mean()

            train_iteration_total_loss = train_iteration_best_age_loss

            train_best_age_loss_list.append(train_iteration_best_age_loss.item())
        elif analysis=="view":
            label_view = label_view.type(torch.FloatTensor)
            label_view = label_view.cuda()
            train_iteration_view_loss = criterion_regression(preds, label_view)

            train_iteration_total_loss = train_iteration_view_loss

            train_view_loss_list.append(train_iteration_view_loss.item())
        elif analysis=="sales":
            label_sales = label_sales.type(torch.FloatTensor)
            label_sales = label_sales.cuda()
            train_iteration_sales_loss = criterion_regression(preds, label_sales)

            train_iteration_total_loss = train_iteration_sales_loss

            train_sales_loss_list.append(train_iteration_sales_loss.item())

        iteration_total_loss = train_iteration_total_loss.detach().cpu()
        if np.isnan(iteration_total_loss) or np.isinf(iteration_total_loss):
            continue

        train_iteration_total_loss.backward()
        optimizer.step()

    acc=0
    loss=0
    if analysis == "best_sex":
        train_best_sex_acc = 100. * correct_best_sex / len(train_loader.dataset)
        loss = np.mean(train_best_sex_loss_list)
        acc = np.mean(train_best_sex_acc)
        print("Epoch:{}    acc:{}".format(epoch + 1, acc))
    elif analysis == "best_age":
        train_best_age_acc = 100. * correct_best_age / len(train_loader.dataset)
        loss = np.mean(train_best_age_loss_list)
        acc = np.mean(train_best_age_acc)
        print("Epoch:{}    acc:{}".format(epoch + 1, acc))
    elif analysis == "view":
        loss = np.mean(train_view_loss_list)

    elif analysis == "sales":
        loss = np.mean(train_sales_loss_list)


    print("Epoch:{}   loss:{}".format(epoch + 1,loss))

    train_epoch_acc.append(acc)

    train_epoch_loss.append(loss)

def get_bert_feature_by_batch(clothing_feature):
    for i in range(0,len(clothing_feature)):
        if i is 0:
            out_concat=clothing_feature[0]
        else:
            out_concat = torch.cat((out_concat, clothing_feature[i]), dim=0)
    return out_concat

def evaluate(model, test_loader):
    model.eval()

    test_best_sex_loss = 0
    test_best_age_loss = 0
    test_view_loss = 0
    test_sales_loss = 0

    correct_best_sex = 0
    correct_best_age = 0

    with torch.no_grad():
        for image, sex, label_best_sex, label_best_age, label_view, label_sales, price, category, clothing_feature, _ in test_loader:
            sex = sex.clone().detach().unsqueeze(1)
            label_best_sex = label_best_sex.clone().detach().cuda()
            label_best_age = label_best_age.clone().detach().cuda()
            label_view = label_view.clone().detach().cuda().unsqueeze(1)
            label_sales = label_sales.clone().detach().cuda().unsqueeze(1)
            price = price.clone().detach().unsqueeze(1)
            category = category.clone().detach().unsqueeze(1)

            bert_feature_batch = get_bert_feature_by_batch(clothing_feature)
            #print("test bert_feature_batch: ", bert_feature_batch)
            preds = model(image.cuda(), sex.cuda(), price.cuda(), category.cuda(),bert_feature_batch.cuda())  # torch.Size([BATCH_SIZE, NUM_CLASSES])

            if analysis == "best_sex":
                best_sex_prediction = preds.max(1, keepdim=True)[1]
                correct_best_sex += best_sex_prediction.eq(label_best_sex.view_as(best_sex_prediction)).sum().item()

                test_best_sex_loss += criterion(preds, label_best_sex).item()

                test_total_acc = 100. * correct_best_sex / len(test_loader.dataset)

                test_total_loss = test_best_sex_loss
                print("Epoch:{}   Test acc:{}".format(epoch + 1, test_total_acc))
                test_epoch_acc.append(test_total_acc)
            elif analysis == "best_age":
                best_age_prediction = preds.max(1, keepdim=True)[1]
                correct_best_age += best_age_prediction.eq(label_best_age.view_as(best_age_prediction)).sum().item()

                test_best_age_loss += criterion(preds, label_best_age).item()
                test_total_acc = 100. * correct_best_age / len(test_loader.dataset)

                test_total_loss = test_best_age_loss
                print("Epoch:{}   Test acc:{}".format(epoch + 1, test_total_acc))
                test_epoch_acc.append(test_total_acc)
            elif analysis == "view":
                label_view = label_view.type(torch.FloatTensor)
                label_view = label_view.cuda()

                test_view_loss += criterion_regression(preds, label_view)

                test_total_loss = test_view_loss
            elif analysis == "sales":
                label_sales = label_sales.type(torch.FloatTensor)
                label_sales = label_sales.cuda()

                test_sales_loss += criterion_regression(preds, label_sales)

                test_total_loss = test_sales_loss

            iteration_total_loss = test_total_loss

        test_total_loss /= len(test_loader.dataset)

        print("Epoch:{}   eval loss:{}".format(epoch + 1,test_total_loss))


        print("")



        test_epoch_loss.append(test_total_loss)


# 스크립트를 실행하려면 여백의 녹색 버튼을 누릅니다.
if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    # _net = Network.ConvNet().cuda()
    _net = resnet_pre_trained.ResNet().cuda()
    # _net = resnet.ResNet().cuda()
    net = nn.DataParallel(_net).to(device)

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    start_train_time = time.time()
    for epoch in range(num_epochs):
        train(net, train_loader, optimizer)
        evaluate(net, test_loader)

    total_train_evaluate_time = str((time.time() - start_train_time) / 60) + " min"

    # Save model
    torch.save(net.state_dict(), "./model_weights/analysis_" + str(analysis) +"_lr_" + str(lr) +"_epoch_" + str(num_epochs) + "_weight_decay_" + str(weight_decay) + "_clip_norm_" + str(
        clip_norm) + "_model_state__dict.pt")

    f = open('results/AccLoss2.txt', 'a')
    f.write("analysis: " + str(analysis) + "\n")
    f.write("num_epochs: " + str(num_epochs) + "\n")
    f.write("loss_alpha: " + str(loss_alpha) + "\n")
    f.write("loss_beta: " + str(loss_beta) + "\n")
    f.write("focal_alpha: " + str(alpha) + "\n")
    f.write("focal_gamma: " + str(gamma) + "\n")
    f.write("batch_size: " + str(batch_size) + "\n")
    f.write("lr: " + str(lr) + "\n")
    f.write("weight_decay: " + str(weight_decay) + "\n")
    f.write("clip_norm: " + str(clip_norm) + "\n")
    f.write("dir: " + str(shoppingDataset_loader.data_path) + "\n")
    f.write("total_train_time: " + str(total_train_evaluate_time) + "\n")
    f.write("\n")

    if analysis == "best_sex":
        f.write("test_epoch_best_sex_acc= ")
        f.write(str(test_epoch_acc) + "\n")
    elif analysis == "best_age":
        f.write("test_epoch_best_age_acc= ")
        f.write(str(test_epoch_acc) + "\n")
    elif analysis == "view":
        f.write("test_epoch_view_loss= ")
        f.write(str(test_epoch_loss) + "\n")
    elif analysis == "sales":
        f.write("test_epoch_sales_loss= ")
        f.write(str(test_epoch_loss) + "\n")
    f.write("\n")

    f.close()
