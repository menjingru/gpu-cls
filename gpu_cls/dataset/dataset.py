import pandas as pd
import torch
import torch.utils.data
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path


class myDataset(Dataset):
    def __init__(self, pre_npy, lei=''):
        self.pre_npy = pre_npy
        self.lei = lei
        self.annos_img, self.annos_lbl = self.get_data(self.pre_npy, self.lei)

    def __getitem__(self, index):
        """
        加载数据集
        :param index:
        :return:
        """
        img = self.annos_img[index]
        lbl = self.annos_lbl[index]
        img = np.load(img)
        lbl = np.load(lbl)
        print(img.shape)
        print(lbl.shape)
        # img = torch.tensor(img.squeeze())  # 这一步应当挪到预处理
        lbl = torch.tensor(lbl)
        # img = img.type(torch.FloatTensor)
        # print(img.shape)
        # print(lbl.shape)
        torch.cuda.empty_cache()
        return img, lbl

    def __len__(self):
        return len(self.annos_img)

    @staticmethod
    def get_data(npy_path, lei):
        """
        分配数据集
        :param npy_path: 表格绝对地址
        :param lei: str（tr/val/te） 训练/验证/测试
        :return: 图片绝对地址list、标签绝对地址lise
        """
        img = sorted([str(i) for i in Path(npy_path + '/img').glob('*')])
        lbl = sorted([str(i) for i in Path(npy_path + '/lbl').glob('*')])
        if 'tr' in lei:
            img = img#[:len(img) * 4 // 6]
            lbl = lbl#[:len(lbl) * 4 // 6]
        elif 'val' in lei:
            img = img[len(img) * 4 // 6:len(img) * 5 // 6]
            lbl = lbl[len(lbl) * 4 // 6:len(lbl) * 5 // 6]
        elif 'ts' in lei:
            img = img[len(img) * 5 // 6:]
            lbl = lbl[len(lbl) * 5 // 6:]
        else:
            print('ERROR:no pattern')
        return img, lbl


