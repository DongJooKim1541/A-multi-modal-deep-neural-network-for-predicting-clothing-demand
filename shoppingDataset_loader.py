import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random
import math
from config import *


df=pd.read_csv(csv_path,encoding='cp949')
idx_list=df['index'].to_list()
file_list = os.listdir(data_path)
print("len(file_list): ",len(file_list))
label_best_sex_and_best_age=[]

for file_name in file_list:
    list = file_name.split(".png")[0]
    #print(list)
    list = list.split("_")
    label_best_sex_and_best_age.append(int(float(list[6])) * 1000 + int(float(list[3])) * 100 +int(float(list[4])) * 10 + int(float(list[8])))

#file_list_train,file_list_test=train_test_split(file_list, random_state=0, stratify=label_best_sex_and_best_age)
file_list_train,file_list_test=train_test_split(file_list, random_state=0)

# dataset random sampling, (file_list,data_size)
sampleList = random.sample(file_list_train, len(file_list_train))

print("len(sampleList), len(file_list_test): ", len(sampleList), len(file_list_test))

# define dataloader class
class ShoppingDataset(Dataset):
    def __init__(self, csv_info, train):
        self.train=train
        self.price=[]
        self.category=[]
        self.sex=[]
        self.label_sales = []
        self.label_best_sex=[]
        self.label_best_age = []
        self.label_view=[]
        self.idx=[]
        self.clothing_feature=[]
        self.file_list_train=sampleList
        self.file_list_test = file_list_test
        # print(csv_info[0]['0'].dtype) # torch.float32
        # {'goods_num': '2447685', 'sex': '2', 'best_sex': '1', 'best_age': '5', 'view': '3', 'sales': '0', 'price': '8', 'category': '4', 'Name': '0'}
        if self.train:
            print("Train")
            self.status="train"
            self.file_list=self.file_list_train
            print("file_list_train: ",len(self.file_list_train))
            for file_name in self.file_list_train:
                list=file_name.split(".png")[0]
                list=list.split("_")
                # print(list)# ['8711', '2680976', '0', '1', '5', '4100', '7', '100', '2', '69900', '1', '0']
                self.idx.append(int(float(list[0])))
                if int(list[0]) in idx_list:
                    index=idx_list[int(list[0])]
                    # print("index: ",index) # 12606
                    word_dict=csv_info[int(index)]
                    # print("word_dict: ",word_dict) # {'12606': '배색큐롯스커트(W)'}
                    words_feature=word_dict[str(index)]
                    #print(words_feature.dtype)
                    #print(words_feature.shape)
                    # print("words: ", words) # 배색큐롯스커트(W)
                    self.clothing_feature.append(words_feature)
                else:
                    zero_tensor=torch.zeros(1,768)
                    self.clothing_feature.append(zero_tensor)
                self.sex.append(int(float(list[2])))
                self.label_best_sex.append(int(float(list[3])))
                self.label_best_age.append(int(float(list[4])))
                if int(float(list[5])) != 0:
                    self.label_view.append(math.log(int(float(list[5]))))
                else:
                    self.label_view.append(int(float(list[5])))
                if int(float(list[7])) != 0:
                    self.label_sales.append(math.log(int(float(list[7]))))
                else:
                    self.label_sales.append(int(float(list[7])))
                self.price.append(int(float(list[9])))
                self.category.append(int(float(list[10])))
        elif not self.train:
            print("Test")
            self.status="test"
            self.file_list = self.file_list_test
            print("file_list_test: ",len(self.file_list_test))
            for file_name in self.file_list_test:
                list=file_name.split(".png")[0]
                list=list.split("_")

                self.idx.append(int(float(list[0])))
                if int(list[0]) in idx_list:
                    index = idx_list[int(list[0])]
                    word_dict = csv_info[int(index)]
                    words_feature = word_dict[str(index)]
                    self.clothing_feature.append(words_feature)
                else:
                    zero_tensor = torch.zeros(1, 768)
                    self.clothing_feature.append(zero_tensor)
                self.sex.append(int(float(list[2])))
                self.label_best_sex.append(int(float(list[3])))
                self.label_best_age.append(int(float(list[4])))
                if int(float(list[5])) != 0:
                    self.label_view.append(math.log(int(float(list[5]))))
                else:
                    self.label_view.append(int(float(list[5])))
                if int(float(list[7])) != 0:
                    self.label_sales.append(math.log(int(float(list[7]))))
                else:
                    self.label_sales.append(int(float(list[7])))
                self.price.append(int(float(list[8])))
                self.category.append(int(float(list[10])))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        if self.train:
            img_name = self.file_list_train[index]
            #print("getitem train")
            img = Image.open(data_path+img_name).convert('RGB')
            img = self.transform(img)
            return img, self.sex[index],self.label_best_sex[index],self.label_best_age[index],self.label_view[index],self.label_sales[index],self.price[index],self.category[index],self.clothing_feature[index],index
        elif not self.train:
            #print("getitem test")
            img_name = self.file_list_test[index]
            img = Image.open(data_path+img_name).convert('RGB')
            img = self.transform2(img)
            return img, self.sex[index],self.label_best_sex[index],self.label_best_age[index],self.label_view[index],self.label_sales[index],self.price[index],self.category[index],self.clothing_feature[index],index

    def transform(self, image):
        transform_ops = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomResizedCrop((125,125),scale=(0.1,1),ratio=(0.5,2)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        return transform_ops(image)

    def transform2(self, image):
        transform_ops = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        return transform_ops(image)

    def get_bert_feature(self, clothing, tokenizer, net_bert):
        encoded_input = tokenizer(clothing, return_tensors='pt')
        output = net_bert(**encoded_input)
        # print(output[1].shape)
        output = output[1].clone().detach().cuda()
        return output


