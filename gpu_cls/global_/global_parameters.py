# -- coding: utf-8 --

import torch
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽通知和警告信息
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用gpu0

BATCH_SIZE = 4  # 2
EPOCH = 200  # 共跑200轮
lr = 0.001
momentum = 0.99
weight_decay = 1e-8
num_workers = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
