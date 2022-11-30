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
    train_total_loss_list = []

    correct_best_sex = 0
    correct_best_age = 0
    correct_view = 0
    correct_sales = 0
    correct_total = 0

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

        pred_best_sex, pred_best_age, pred_view, pred_sales = model(image.cuda(), sex.cuda(), price.cuda(),
                                                                    category.cuda(), bert_feature_batch.cuda())  # torch.Size([BATCH_SIZE, NUM_CLASSES])

        # 정확도 구하는 코드 짜야함
        # classification
        best_sex_prediction = pred_best_sex.max(1, keepdim=True)[1]
        correct_best_sex += best_sex_prediction.eq(label_best_sex.view_as(best_sex_prediction)).sum().item()
        best_age_prediction = pred_best_age.max(1, keepdim=True)[1]
        correct_best_age += best_age_prediction.eq(label_best_age.view_as(best_age_prediction)).sum().item()

        # train_best_age_prediction.append(best_age_prediction.item())

        label_view = label_view.type(torch.FloatTensor)
        label_sales = label_sales.type(torch.FloatTensor)

        label_view = label_view.cuda()
        label_sales = label_sales.cuda()

        best_sex_pred_list = best_sex_prediction.eq(label_best_sex.view_as(best_sex_prediction)).tolist()
        best_age_pred_list = best_age_prediction.eq(label_best_age.view_as(best_age_prediction)).tolist()

        for i in range(0, len(best_sex_pred_list)):
            if best_sex_pred_list[i] == [True] and best_age_pred_list[i] == [True]:
                correct_total += 1

        # loss 구하기
        train_iteration_best_sex_loss = criterion(pred_best_sex, label_best_sex)

        # train_iteration_best_age_loss = criterion(pred_best_age, label_best_age)

        # train_iteration_sales_loss = criterion(pred_sales, label_sales)

        # focal loss(classification)
        train_iteration_best_age_loss = criterion_for_focal(pred_best_age, label_best_age)
        pt = torch.exp(-train_iteration_best_age_loss)
        train_iteration_best_age_loss = (alpha * (1 - pt) ** gamma * train_iteration_best_age_loss).mean()

        # regression loss
        train_iteration_view_loss = criterion_regression(pred_view, label_view)
        train_iteration_sales_loss = criterion_regression(pred_sales, label_sales)

        # 전체 loss
        train_iteration_total_loss = train_iteration_best_sex_loss + train_iteration_best_age_loss + loss_alpha * train_iteration_view_loss + loss_beta * train_iteration_sales_loss
        iteration_total_loss = train_iteration_total_loss.detach().cpu()
        if np.isnan(iteration_total_loss) or np.isinf(iteration_total_loss):
            continue

        # 각각의 loss list
        train_best_sex_loss_list.append(train_iteration_best_sex_loss.item())
        train_best_age_loss_list.append(train_iteration_best_age_loss.item())
        train_view_loss_list.append(train_iteration_view_loss.item())
        train_sales_loss_list.append(train_iteration_sales_loss.item())
        train_total_loss_list.append(train_iteration_total_loss.item())
        """
        train_iteration_best_sex_loss.backward(retain_graph=True)
        train_iteration_best_age_loss.backward(retain_graph=True)
        train_iteration_sales_loss.backward(retain_graph=True)
        """
        train_iteration_total_loss.backward()
        optimizer.step()

    train_best_sex_acc = 100. * correct_best_sex / len(train_loader.dataset)
    train_best_age_acc = 100. * correct_best_age / len(train_loader.dataset)
    train_view_acc = correct_view / len(train_loader.dataset)
    train_sales_acc = correct_sales / len(train_loader.dataset)
    train_total_acc = 100. * correct_total / len(train_loader.dataset)

    # Mean the iteration loss to 1 epoch loss
    epoch_best_sex_loss = np.mean(train_best_sex_loss_list)
    epoch_best_age_loss = np.mean(train_best_age_loss_list)
    epoch_view_loss = np.mean(train_view_loss_list)
    epoch_sales_loss = np.mean(train_sales_loss_list)
    epoch_total_loss = np.mean(train_total_loss_list)

    # Mean the iteration loss to 1 epoch accuracy
    epoch_best_sex_acc = np.mean(train_best_sex_acc)
    epoch_best_age_acc = np.mean(train_best_age_acc)
    epoch_view_acc = np.mean(train_view_acc)
    epoch_sales_acc = np.mean(train_sales_acc)
    epoch_total_acc = np.mean(train_total_acc)

    print(
        "Epoch:{}    Train best sex loss:{}    Train best age loss:{}    Train view loss:{}    Train sales loss:{}    Total loss:{}".format(
            epoch + 1, epoch_best_sex_loss, epoch_best_age_loss, epoch_view_loss, epoch_sales_loss, epoch_total_loss))
    print(
        "Epoch:{}    Train best sex acc:{}    Train best age acc:{}    Total acc:{}".format(
            epoch + 1, epoch_best_sex_acc, epoch_best_age_acc, epoch_total_acc))

    train_epoch_acc.append(epoch_total_acc)
    train_epoch_best_sex_acc.append(epoch_best_sex_acc)
    train_epoch_best_age_acc.append(epoch_best_age_acc)

    train_epoch_loss.append(epoch_total_loss)
    train_epoch_best_sex_loss.append(epoch_best_sex_loss)
    train_epoch_best_age_loss.append(epoch_best_age_loss)
    train_epoch_view_loss.append(epoch_view_loss)
    train_epoch_sales_loss.append(epoch_sales_loss)

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
    correct_view = 0
    correct_sales = 0
    correct_total = 0

    with torch.no_grad():
        for image, sex, label_best_sex, label_best_age, label_view, label_sales, price, category, clothing_feature, _ in test_loader:
            sex = sex.clone().detach().unsqueeze(1)
            label_best_sex = label_best_sex.clone().detach().cuda()
            label_best_age = label_best_age.clone().detach().cuda()
            label_view = label_view.clone().detach().cuda().unsqueeze(1)
            label_sales = label_sales.clone().detach().cuda().unsqueeze(1)
            price = price.clone().detach().unsqueeze(1)
            category = category.clone().detach().unsqueeze(1)

            bert_feature = get_bert_feature_by_batch(clothing_feature)

            pred_best_sex, pred_best_age, pred_view, pred_sales = model(image.cuda(), sex.cuda(), price.cuda(),
                                                                        category.cuda(), bert_feature.cuda())  # torch.Size([BATCH_SIZE, NUM_CLASSES])

            # 정확도 구하는 코드 짜야함
            # classification
            best_sex_prediction = pred_best_sex.max(1, keepdim=True)[1]
            correct_best_sex += best_sex_prediction.eq(label_best_sex.view_as(best_sex_prediction)).sum().item()
            best_age_prediction = pred_best_age.max(1, keepdim=True)[1]
            correct_best_age += best_age_prediction.eq(label_best_age.view_as(best_age_prediction)).sum().item()

            # regression r2_score
            label_view = label_view.type(torch.FloatTensor)
            label_sales = label_sales.type(torch.FloatTensor)

            label_view = label_view.cuda()
            label_sales = label_sales.cuda()

            best_sex_pred_list = best_sex_prediction.eq(label_best_sex.view_as(best_sex_prediction)).tolist()
            best_age_pred_list = best_age_prediction.eq(label_best_age.view_as(best_age_prediction)).tolist()

            for i in range(0, len(best_sex_pred_list)):
                if best_sex_pred_list[i] == [True] and best_age_pred_list[i] == [True]:
                    correct_total += 1

            # loss 구하기
            test_best_sex_loss += criterion(pred_best_sex, label_best_sex).item()
            test_best_age_loss += criterion(pred_best_age, label_best_age).item()

            # regression loss
            test_view_loss += criterion_regression(pred_view, label_view)
            test_sales_loss += criterion_regression(pred_sales, label_sales)

            test_best_sex_acc = 100. * correct_best_sex / len(test_loader.dataset)
            test_best_age_acc = 100. * correct_best_age / len(test_loader.dataset)
            test_view_acc = correct_view / len(test_loader.dataset)
            test_sales_acc = correct_sales / len(test_loader.dataset)
            test_total_acc = 100. * correct_total / len(test_loader.dataset)

            # 전체 loss
            test_total_loss = test_best_sex_loss + test_best_age_loss + loss_alpha * test_view_loss + loss_beta * test_sales_loss

            iteration_total_loss = test_total_loss.detach().cpu()
            if np.isnan(iteration_total_loss) or np.isinf(iteration_total_loss):
                continue

        test_best_sex_loss /= len(test_loader.dataset)
        test_best_age_loss /= len(test_loader.dataset)
        test_view_loss /= len(test_loader.dataset)
        test_sales_loss /= len(test_loader.dataset)
        test_total_loss /= len(test_loader.dataset)

        print(
            "Epoch:{}    eval best sex loss:{}    eval best age loss:{}    eval view loss:{}    eval sales loss:{}    eval loss:{}".format(
                epoch + 1, test_best_sex_loss, test_best_age_loss, test_view_loss, test_sales_loss,
                test_total_loss))

        print(
            "Epoch:{}    Test best sex acc:{}    Test best age acc:{}   Test acc:{}".format(
                epoch + 1, test_best_sex_acc, test_best_age_acc, test_total_acc))
        print("")

        test_epoch_acc.append(test_total_acc)
        test_epoch_best_sex_acc.append(test_best_sex_acc)
        test_epoch_best_age_acc.append(test_best_age_acc)

        test_epoch_loss.append(test_total_loss)
        test_epoch_best_sex_loss.append(test_best_sex_loss)
        test_epoch_best_age_loss.append(test_best_age_loss)
        test_epoch_view_loss.append(test_view_loss)
        test_epoch_sales_loss.append(test_sales_loss)


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
    torch.save(net.state_dict(), "./model_weights/lr_" + str(lr) +"_epoch_" + str(num_epochs) + "_weight_decay_" + str(weight_decay) + "_clip_norm_" + str(
        clip_norm) + "_model_state__dict.pt")

    f = open('results/AccLoss2.txt', 'a')
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
    f.write("train_epoch_acc= ")
    f.write(str(train_epoch_acc) + "\n")
    f.write("train_epoch_best_sex_acc= ")
    f.write(str(train_epoch_best_sex_acc) + "\n")
    f.write("train_epoch_best_age_acc= ")
    f.write(str(train_epoch_best_age_acc) + "\n")
    f.write("\n")
    f.write("train_epoch_loss= ")
    f.write(str(train_epoch_loss) + "\n")
    f.write("train_epoch_best_sex_loss= ")
    f.write(str(train_epoch_best_sex_loss) + "\n")
    f.write("train_epoch_best_age_loss= ")
    f.write(str(train_epoch_best_age_loss) + "\n")
    f.write("train_epoch_view_loss= ")
    f.write(str(train_epoch_view_loss) + "\n")
    f.write("train_epoch_sales_loss= ")
    f.write(str(train_epoch_sales_loss) + "\n")
    f.write("\n")

    f.write("test_epoch_acc= ")
    f.write(str(test_epoch_acc) + "\n")
    f.write("test_epoch_best_sex_acc= ")
    f.write(str(test_epoch_best_sex_acc) + "\n")
    f.write("test_epoch_best_age_acc= ")
    f.write(str(test_epoch_best_age_acc) + "\n")
    f.write("\n")
    f.write("test_epoch_loss= ")
    f.write(str(test_epoch_loss) + "\n")
    f.write("test_epoch_best_sex_loss= ")
    f.write(str(test_epoch_best_sex_loss) + "\n")
    f.write("test_epoch_best_age_loss= ")
    f.write(str(test_epoch_best_age_loss) + "\n")
    f.write("test_epoch_view_loss= ")
    f.write(str(test_epoch_view_loss) + "\n")
    f.write("test_epoch_sales_loss= ")
    f.write(str(test_epoch_sales_loss) + "\n")
    f.write("\n")

    f.close()
