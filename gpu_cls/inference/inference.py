import numpy as np

from gpu_cls.global_.global_path import *
from gpu_cls.global_.global_parameters import *
from gpu_cls.preprocessing.preprocessing import preprocessing
from gpu_cls.preprocessing.statistics import get_message
import nibabel as nb
import torch.utils.data


# 输入是原始数据的ct_npy 和占位的mask_npy，输出也是一个npy对mask_npy中的占位做标记，是HCC的全部赋值为1(需求)


def inference(in_img_path='', in_msk_path='', infer_model_path=best_model_path, out_dir_path=inference_file_pred):
    """
    文件/文件夹推理
    功能1：label_view=False 输入  没有真实标签，有img、mask   输出  染色预测
    :param in_img_path: 输入img文件
    :param in_lbl_path: 输入lbl文件
    :param (默认): 输入文件夹 inference_file_img
    :param (默认): 输出文件夹 inference_file_pred
    :param infer_model_path: 使用的模型绝对地址
    :return:
    """
    # model
    model = torch.load(infer_model_path)
    model = model.to(DEVICE)
    model.eval()
    # dataloader
    # 拿到一张图和一个mask，我需要输出一个相同大小mask，其中有三种标签，分别为0：非hcc非tumor 1：hcc 2：tumor
    # 首先拿到图和mask
    one_img = in_img_path
    one_msk = in_msk_path
    full_exist = Path(one_img).exists() and Path(one_msk).exists()
    if full_exist:
        _, affine, msg = get_message(one_msk, one_img)
        # print(msg)
        preprocess = preprocessing(msg, None)
        imgs, lbl = preprocess.pre(save=False)
        img_zero = torch.zeros(msg[0][6], dtype=torch.int)
        # imgs是处理好准备送进网络的tensor，_是单个占位的mask
        with torch.no_grad():
            for id in range(len(imgs)):
                data = imgs[id][0]
                data = torch.unsqueeze(data, 0).cuda()
                output = model(data)
                pred = -output.argmax(dim=1) + 2  # hcc=1,tumor=2
                pred = torch.squeeze(pred, 0)
                one_pred = pred.cpu() * torch.tensor(imgs[id][1])
                img_zero += one_pred
        img_zero = img_zero.long()
        nb.Nifti1Image(img_zero.numpy(), affine).to_filename(out_dir_path + '/%s' % str(Path(one_img).name))


"""
我想弄一个可视化给infer和test通用？？？？不不不test不用，infer给分个有label和没label的选项，可以
哪里在dataset做？读取数据，加载为一个一个预测，注意返回时返回的是img,label,one_placeholder_mask
训练结束后将pre与one_placeholder_mask相乘，取得一个空张量，以
"""
