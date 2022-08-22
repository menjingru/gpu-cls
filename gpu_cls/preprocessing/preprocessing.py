from gpu_cls.utils.utils import grey_use, GaussianNoise, reverse
# from gpu_cls.model.skvnet_def122 import SPPLayer
from gpu_cls.global_.global_path import *
from gpu_cls.global_.global_parameters import *
import pandas as pd
import torch
import torch.utils.data
import numpy as np
from pathlib import Path
import nibabel as nb
from scipy.ndimage.interpolation import zoom
import random
from torchvision import transforms
from time import time
from tqdm import tqdm


class preprocessing:
    def __init__(self, XLSX=XLSX, pre_npy=pre_npy):
        """
        根据XLSX的信息列表进行预处理
        :param XLSX: 信息列表 	[id, name, pixel, cut, label, spacing, shape, center(xyz)]
        :param pre_npy: 预处理后输出地址
        """
        # print("数据预处理中,请稍等:")
        self.XLSX = XLSX
        self.pre_npy = pre_npy
        self.annos_suc, self.annos_img, self.annos_cut, self.annos_lbl, self.annos_spc, self.annos_shp, self.annos_ctr, self.annos_dia, self.infer = self.get_XLSX(
                self.XLSX)
    def pre(self, save=True):
        """
        数据预处理并缓存
        ↓
        原图裁剪
        resize
        归一化
        img、label升维至[c,x,y,z]
        """
        imgs = []
        lbls = []

        for index in tqdm(range(len(self.annos_img)), desc="数据预处理中", colour='blue'):
            suc = self.annos_suc[index]
            img_ = self.annos_img[index]
            cut = self.annos_cut[index]
            lbl = self.annos_lbl[index]
            spc = self.annos_spc[index]
            shp = self.annos_shp[index]
            ctr = self.annos_ctr[index]
            dia = self.annos_dia[index]
            # img
            img = nb.load(img_)
            img_data = img.get_data()
            if self.infer:
                data = img_data
            else:
                liver_mask = nb.load(str(img_).replace('V.nii.gz', 'liver.nii.gz'))  # 载入图片
                liver_mask = liver_mask.get_data()

                data = img_data * liver_mask
                # data = img_data

            #  ↓
            time1 = time()
            # augmentation
            # if self.infer :
            #     pass
            # else:
            #     tr_transforms = []
            #     tr_transforms.append(GaussianNoise())
            #     tr_transforms.append(reverse())
            #     tr_transforms = transforms.Compose(tr_transforms)
            #     data = tr_transforms(data)
            ## 置灰
            time2 = time()
            data, _ = self.grey(data, suc, cut, shp)
            ## 重采样
            time3 = time()
            if self.infer:
                data_resampled = data
            else:
                data_resampled = self.resample_ct(data, spc, is_mask=False)
                # _resampled = self.resample_ct(_, spc, is_mask=True)

            ## 更新cut、ctr
            time4 = time()
            if self.infer:
                pass
            else:
                shp = [int(shp[0] * spc[0]), int(shp[1] * spc[1]), int(shp[2] * spc[2])]
                cut = [cut[0] * spc[0], cut[1] * spc[1], cut[2] * spc[2], cut[3] * spc[0], cut[4] * spc[1],
                       cut[5] * spc[2]]
                ctr = [ctr[0] * spc[0], ctr[1] * spc[1], ctr[2] * spc[2]]
            ## cut
            time5 = time()
            # one_ct_resampled = data_resampled * _resampled
            one_ct_resampled = data_resampled
            # data = data[cut[0]:cut[3], cut[1]:cut[4], cut[2]:cut[5]]
            # data = self.cutn(data_resampled, ctr, cut)
            # data32 = self.cutn(one_ct_resampled, ctr, cut, 32)
            data64 = self.cutn(one_ct_resampled, ctr, cut, 64)
            # data256 = self.cutn(one_ct_resampled, ctr, cut, 256)
            # norm
            time6 = time()
            # data = self.norm(data)
            # data32 = self.norm(data32)
            data64 = self.norm(data64)
            # data256 = self.norm(data256)
            # resize
            time7 = time()
            # data = np.expand_dims(np.resize(data, (128, 128, 128)), 0)
            # data32 = np.expand_dims(np.resize(data32, (128, 128, 128)), 0)
            # data64 = np.expand_dims(np.resize(data64, (128, 128, 128)), 0)
            # data256 = np.expand_dims(np.resize(data256, (128, 128, 128)), 0)
            # data = np.concatenate((data, data32, data64, data256), 0)
            # data = np.concatenate((data, data64), 0)
            data = data64

            # 调整维度
            time8 = time()
            data = np.ascontiguousarray(data)
            img = torch.tensor(data)
            img = img.type(torch.FloatTensor)
            # spp
            # img = torch.unsqueeze(img, 0)
            # img = torch.unsqueeze(img, 0)
            # img = img.to(DEVICE)
            # spp = SPPLayer().to(DEVICE)
            # # img = SPPLayer()(img)
            # print(img.shape)
            # img_list = spp(img)
            #
            # img8 = np.expand_dims(np.resize(torch.squeeze(torch.squeeze(img_list[0].cpu())).numpy(), (128, 128, 128)), 0)
            # img16 = np.expand_dims(np.resize(torch.squeeze(torch.squeeze(img_list[1].cpu())).numpy(), (128, 128, 128)), 0)
            # img32 = np.expand_dims(np.resize(torch.squeeze(torch.squeeze(img_list[2].cpu())).numpy(), (128, 128, 128)), 0)
            # img64 = np.expand_dims(np.resize(torch.squeeze(torch.squeeze(img_list[3].cpu())).numpy(), (128, 128, 128)), 0)
            # img128 = np.expand_dims(np.resize(torch.squeeze(torch.squeeze(img_list[4].cpu())).numpy(), (128, 128, 128)), 0)
            # img256 = np.expand_dims(data256, 0)
            # img = np.concatenate((img8, img16, img32, img64, img128, img256), 0)
            #  ↑

            img = np.expand_dims(img, 0)
            img = torch.tensor(img)
            img = img.type(torch.FloatTensor)
            # img = torch.squeeze(img,0)
            # img = torch.squeeze(img,0)

            # label
            label = np.array([lbl])
            label = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.Tensor(label), 0), 0), 0).long()
            torch.cuda.empty_cache()
            # print(img.shape)  # torch.Size([1, 4, 128, 128, 128])
            # print(label.shape)  # torch.Size([1, 1, 1, 1])
            time9 = time()
            if save:
                img = img.numpy()
                label = label.numpy()
                save_one = self.save(self.pre_npy, index, img_, img, label)
            else:
                save_one = ''
            imgs.append([img, _])
            lbls.append(label)
            time10 = time()
            # print("\n {} \n time12:{:.6f},\ntime23:{:.6f},\ntime34:{:.6f},\ntime45:{:.6f},\ntime56:{:.6f},\n"
            #       "time67:{:.6f},\ntime78:{:.6f},\ntime89:{:.6f},\ntime910:{:.6f},\n".format(save_one,
            #     time2 - time1, time3 - time2,
            #     time4 - time3, time5 - time4, time6 - time5, time7 - time6, time8 - time7, time9 - time8,
            #     time10 - time9))
        print('preprocessing down!')
        return imgs, lbls

    @staticmethod
    def get_XLSX(XLSX):
        """
        list 地址下所有图片的绝对地址
        :param XLSX: 信息列表  [id, name, pixel, cut, label, spacing, shape, center(xyz)]
        :return: suc占位绝对地址, img原图绝对地址, lbl标签, cut原图裁剪坐标, spc原图像素间隔, shp原图尺寸, ctr原图占位中心
        """
        if type(XLSX) == str:
            infer = False
            exc_data = pd.read_excel(XLSX, sheet_name=0, header=0, index_col=0)
            datas_list = [i for i in exc_data.values]
        elif type(XLSX) == list:
            infer = True
            datas_list = XLSX
        else:
            infer = False
            datas_list = []
            print('wrong type input, you can input str or message list!')

        random.shuffle(datas_list)
        suc = []
        img = []
        lbl = []
        cut = []
        spc = []
        shp = []
        ctr = []
        dia = []
        datas = datas_list
        for one_msg in datas:
            # print(one_msg)
            # augmentation
            if infer:
                times = 1
            else:
                if 'hcc.nii.gz' in str(one_msg[0]):
                    # times = 3
                    times = 1
                else:
                    # times = 2
                    times = 1
            # load message
            for k in range(times):
                suc.append(one_msg[0])
                img.append(one_msg[1])
                if infer:
                    cut.append(one_msg[3])
                else:
                    cut.append([int(p) for p in one_msg[3].lstrip('(').rstrip(')').split(',')])
                lbl.append(int(one_msg[4]))
                if infer:
                    spc.append(one_msg[5])
                    shp.append(one_msg[6])
                    ctr.append(one_msg[7])
                else:
                    spc.append([float(p) for p in one_msg[5].lstrip('[').rstrip(']').split(',')])
                    shp.append([int(p) for p in one_msg[6].lstrip('(').rstrip(')').split(',')])
                    ctr.append([float(p) for p in one_msg[7].lstrip('[').rstrip(']').split(',')])
                dia.append(int(one_msg[8]))
        return suc, img, cut, lbl, spc, shp, ctr, dia, infer

    @staticmethod
    def grey(data, suc, cut, shp):  # 数组，图片地址
        """
        置灰
        :param data: 原图
        :param suc: 原图绝对地址
        :param cut: 原图裁剪坐标（xyz min,xyz max）
        :param shp: 原图尺寸
        :return:data_greyed灰化后的图，one_placeholder_mask单占位mask
        """
        # hcc mask + tumor mask
        hcc = str(suc).replace('tumor.nii.gz', 'hcc.nii.gz')
        tumor = str(suc).replace('hcc.nii.gz', 'tumor.nii.gz')
        hcc_img = nb.load(hcc)
        hcc_data = hcc_img.get_data()
        if Path(tumor).exists():
            tumor_img = nb.load(tumor)
            tumor_data = tumor_img.get_data()
            placeholder_mask = hcc_data + tumor_data
        else:
            placeholder_mask = hcc_data
        # only one placeholder mask
        one_placeholder_mask = placeholder_mask[cut[3]:cut[0], cut[4]:cut[1], cut[5]:cut[2]]
        one_placeholder_mask = grey_use(one_placeholder_mask)
        zero_placeholder_mask = np.zeros(shp)
        zero_placeholder_mask[cut[3]:cut[0], cut[4]:cut[1], cut[5]:cut[2]] = one_placeholder_mask
        one_placeholder_mask = placeholder_mask
        # no for grey mask
        no_grey_mask = np.abs((placeholder_mask - one_placeholder_mask) - 1)
        data_greyed = data * no_grey_mask
        data_greyed[data_greyed == 0] = np.mean(data_greyed)
        return data_greyed, one_placeholder_mask

    @staticmethod
    def resample(imgs, spacing):
        """
        输入原图，输出重采样后的图, new_spacing=[1, 1, 1]
        :param imgs: 原图
        :param spacing: 原像素间距
        :return: 像素间距为[1, 1, 1]的新图
        """
        time34_1 = time()
        new_shape = []
        for i in range(3):
            # print("（zyx）像素间隔", i, ":", spacing[-i - 1])  ###   spacing（zyx）
            new_zyx = np.round(imgs.shape[i] * spacing[i])
            new_shape.append(new_zyx)
        # print(new_shape)
        time34_2 = time()
        resize_factor = []
        for i in range(3):
            resize_zyx = new_shape[i] / imgs.shape[i]
            resize_factor.append(resize_zyx)
        # print(resize_factor)
        time34_3 = time()
        imgs = zoom(imgs, resize_factor, )
        time34_4 = time()
        # print(imgs.shape)
        # print("time34_12:{:.6f}\ntime34_23:{:.6f}\ntime34_34:{:.6f}\n".format(time34_2 - time34_1, time34_3 - time34_2,
        #                                                                       time34_4 - time34_3))
        return imgs

    @staticmethod
    def cutn(data, ctr, cut, length=0):
        """
        裁剪体积 xyz, 默认正方体
        :param data: 待裁剪图
        :param ctr: 占位中心
        :param cut: 裁剪坐标
        :param length: 提供两种功能：length=0为裁剪自适应尺寸，length！=0为裁剪固定尺寸
        :return:
        """
        shp = data.shape
        if length == 0:
            length = max(cut[3] - cut[0], cut[4] - cut[1], cut[5] - cut[2]) * 3 // 2
            length = np.ceil(length / 32) * 32
        else:
            length = length
        cut_list = []
        for i in range(len(shp)):  # 0,1,2   →  x,y,z
            a = ctr[i] - length // 2
            b = ctr[i] + length // 2
            if a < 0:
                a = 0
                b = length
            elif b > shp[i]:
                a = shp[i] - length
                b = shp[i]
            else:
                pass
            cut_list.append(int(a))
            cut_list.append(int(b))
        data = data[cut_list[0]:cut_list[1], cut_list[2]:cut_list[3], cut_list[4]:cut_list[5]]
        return data

    @staticmethod
    def norm(boximg):
        """
        0-1归一化
        :param boximg: 原图
        :return:归一化后图
        """
        # 输入置灰后的固定尺寸256的图，输出归一化去均值的256的图
        boximg = boximg
        max_num = 250
        min_num = -150
        boximg = (boximg - min_num) / (max_num - min_num)
        boximg[boximg > 1] = 1.
        boximg[boximg < 0] = 0.
        return boximg

    @staticmethod
    def save(pre_npy, index, img_, img, label):
        np.save(str(Path(pre_npy) / 'img' / '%s_%s.npy') % (index, str(Path(img_).stem)), img)
        np.save(str(Path(pre_npy) / 'lbl' / '%s_%s.npy') % (index, str(Path(img_).stem)), label)
        return str(Path(pre_npy) / 'img' / '%s_%s.npy') % (index, str(Path(img_).stem))

    @staticmethod
    def resample_ct(img, zoom_factor, is_mask=True):
        """
        统一分辨率，resize等操作就用这个，没问题
        :param img:
        :param zoom_factor: 缩放系数: src_spacing / aim_spacing
        :param is_mask:
        :return:
        """
        if is_mask:
            img_resize = zoom(img, zoom_factor, order=0)
        else:
            img_resize = zoom(img, zoom_factor, order=3)

        return img_resize
