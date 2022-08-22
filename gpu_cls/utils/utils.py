# -- coding: utf-8 --
from gpu_cls.global_.global_path import *
import torch
import torch.utils.data
import numpy as np
from sklearn.metrics import confusion_matrix
from skimage import measure
import nibabel as nb
# from torchvision import transforms
import random


# 预处理用


def grey_use(mask):
    # 求最大连通域
    labels, num = measure.label(mask, background=0,
                                return_num=True)  # 这里返回的labels是一幅图像，不再是一副二值图像，有几个连通域，最大值就是几，num是连通域个数
    max_label = 0
    max_num = 0
    # 图像全黑，没有连通域num=0,或者是由一个连通域num=1，直接返回原图像
    if num == 0 or num == 1:
        return mask
    else:
        for i in range(1, num + 1):  # 注意这里的范围，为了与连通域的数值相对应
            # 计算面积，保留最大面积对应的索引标签，然后返回二值化最大连通域
            if np.sum(labels == i) > max_num:
                max_num = np.sum(labels == i)
                max_label = i
        lcc = (labels == max_label)
        return lcc + 0


class GaussianNoise():
    def __init__(self, ):
        pass

    def __call__(self, data):
        ra = random.uniform(0., 1.)
        if ra > 0.5:
            data = data + np.random.normal(0.0, 1, size=data.shape)
        return data


class reverse():  # 把这个放到resize前是否可以……
    def __init__(self, ):
        pass

    def __call__(self, data):
        x = random.uniform(0., 1.)
        if x > 0.5:
            data = np.flip(data, axis=0)
        y = random.uniform(0., 1.)
        if y > 0.5:
            data = np.flip(data, axis=1)
        z = random.uniform(0., 1.)
        if z > 0.5:
            data = np.flip(data, axis=2)
        return data


# 训练用


def early_or_not(valid_loss_list):
    if len(valid_loss_list) < 20:
        return False
    else:
        # 如果验证loss五轮内没有下降
        if valid_loss_list[-5] == min(valid_loss_list[-5:]):
            return True
        else:
            return False


# 验证用


class indicators():
    def __init__(self, data, label):
        """
        计算指标，[PA（准确率），P（精确率），R（召回率），F1（F1-score），ROC（概率阈值划分正负样本PR曲线），AUC（ROC曲线下面积）]
        """
        my_data = torch.squeeze(torch.squeeze(torch.squeeze(torch.argmax(data, dim=1), -1), -1), -1).cpu().numpy()
        my_label = torch.squeeze(torch.squeeze(torch.squeeze(label, -1), -1), -1).cpu().numpy()

        confuse = confusion_matrix(my_label, my_data, labels=[1, 0])
        self.PA, self.FPR, self.TPR = self.confuse_PR(confuse)

    def get_PA_P_R(self, ):
        """
        计算指标，[PA（准确率），P（精确率），R（召回率）]
        :param data: 图
        :param label: 标签
        :return:
        """
        return self.PA, self.FPR, self.TPR

    @staticmethod
    def confuse_PR(confuse):
        """
        根据混淆函数，计算二分类P R
        :param confuse: 混淆函数
        :return:
        """
        confuse = np.array(confuse)
        # 每个类别的（tp,fn,fp,tn）
        tp = confuse[0][0]
        fn = confuse[0][1]
        fp = confuse[1][0]
        tn = confuse[1][1]
        print('tp,fn,fp,tn:', tp, fn, fp, tn)
        if fp + tn > 0:
            FPR = tn / (fp + tn)
        else:
            FPR = 0
        if tp + fn > 0:
            TPR = tp / (tp + fn)
        else:
            TPR = 0

        if tp + fn + fp + tn > 0:
            PA = (tp + tn) / (tp + fn + fp + tn)
        else:
            PA = 0
        return PA, FPR, TPR


# 推理用

def get_mask(hcc_path, tumor_path, save=True, save_path=inference_file_pred):
    """
    读取hcc、tumor，合并成hcc=1，tumor=1（考卷）
    读取hcc、tumor，合并成hcc=1，tumor=2（标准答案）
    :param hcc_path:
    :param tumor_path:
    :param save: 保存标准答案吗，保存到 inference_file_pred，并命名为 name_ground_truth.nii.gz
    :return:
    """
    hcc = nb.load(hcc_path)
    hcc_affine = hcc.affine
    hcc_data = hcc.get_data()

    tumor = nb.load(tumor_path)
    tumor_data = tumor.get_data()

    exam = hcc_data.copy() + tumor_data.copy()
    exam[exam > 1] = 1
    nb.Nifti1Image(exam, hcc_affine).to_filename(
        str(Path(inference_file_mid) / str('%s_mask.nii.gz' % Path(hcc_path).name[:-11])))
    if save:
        ground_truth = hcc_data.copy() + tumor_data.copy() * 2
        ground_truth[ground_truth > 2] = 2

        nb.Nifti1Image(ground_truth, hcc_affine).to_filename(
            str(Path(save_path) / str('%s_ground_truth.nii.gz' % Path(hcc_path).name[:-11])))

    return str(Path(inference_file_mid) / str('%s_mask.nii.gz' % Path(hcc_path).name[:-11]))
