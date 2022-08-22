# -- coding: utf-8 --

import numpy as np
from tqdm import tqdm
import torch


def train_model(model, train_loader, optimizer, Loss, epoch):
    """
    训练模型
    :param model: 初始化模型/预训练模型
    :param train_loader: 训练集
    :param optimizer: 优化器
    :param Loss: loss function
    :param epoch:EPOCH
    :return: loss值
    """
    model.train()
    loss_need = []
    tqdr = tqdm(enumerate(train_loader))
    for batch_index, (data, target) in tqdr:
        # print(data.shape)
        # print(target.shape)
        # data = torch.unsqueeze(data, 1)
        data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = Loss(output.squeeze(), target.squeeze())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss = loss.item()
        loss_need.append(train_loss)
        tqdr.set_description("Train Epoch : {} \t train Loss : {:.6f} ".format(epoch, loss.item()))
    train_loss = np.mean(loss_need)

    del loss_need

    print("train_loss", train_loss)
    return train_loss
