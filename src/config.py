"""Paramenter generalization"""
CUDA_VISIBLE_DEVICES=0
batch_size = 64
num_epochs = 3000 #3000
lr = 0.0001
weight_decay = 1e-5
clip_norm = 5

# focal loss weight
alpha = 1
gamma = 2

# loss weight
loss_alpha = 0.01
loss_beta = 0.01

data_path = "dataset/category_all_ver2_20221002_words_125_aug/"
#data_path = "dataset/aa/"
csv_path = "dataset/goodsNum_clothing_name_20221002.csv"
#csv_path = "dataset/goodsNum_clothing_name_20221002_sample.csv"