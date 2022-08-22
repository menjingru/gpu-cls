# -- coding: utf-8 --
from global_.global_path import *
from global_.global_parameters import *
from train.train import train
from test.test import test
from inference.inference import inference
from model.skvnet22 import SKVNet
from training.optimizer import *
from utils.utils import get_mask
from preprocessing.statistics import get_excel, get_txt
from preprocessing.preprocessing import preprocessing
import torch.nn as nn

# 总控


simple = True

model = SKVNet()  # 模型
model = model.to(DEVICE)  # 模型部署到gpu或cpu里
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
Loss = nn.CrossEntropyLoss()
Loss = Loss.to(DEVICE)

# 预处理
preprocess_or_not = False
if preprocess_or_not or simple:
    get_excel()
    get_txt()
    p = preprocessing(XLSX=XLSX, pre_npy=pre_npy)
    p.pre(save=True)

# 训练


train_or_not = False
if train_or_not or simple:
    train(model=model, optimizer=optimizer, Loss=Loss, val_or_not=False, early_stop=False)

# 测试


test_or_not = False
if test_or_not or simple:
    test(Loss=Loss, save=True)

# 推理


infer_or_not = True
if infer_or_not or simple:
    in_img = "E:\dataset\hcc_0\infer\inf_img\Z0000777SHI JUN_V.nii.gz"
    in_mask = ""
    if in_mask:
        pass
    else:
        in_hcc = r"E:\dataset\hcc_0\infer\inf_img\Z0000777SHI JUN_hcc.nii.gz"
        in_tumor = r"E:\dataset\hcc_0\infer\inf_img\Z0000777SHI JUN_tumor.nii.gz"
        in_mask = get_mask(hcc_path=in_hcc, tumor_path=in_tumor,save=True)
    inference(in_img_path=in_img, in_msk_path=in_mask, infer_model_path=r'E:\dataset\hcc_0\out\model\train_model.pth')
