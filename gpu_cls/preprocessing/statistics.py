# -- coding: utf-8 --

from gpu_cls.global_.global_path import *
from pathlib import Path
import nibabel as nb
from skimage import measure
import pandas as pd
import time
import openpyxl

simple = True


def get_message(mask_path, img_path=''):
    """
    逐占位读取信息
    注意：推理时所有label为0
    :param mask_path: mask绝对地址
    :return:
    """
    # 读取信息
    ct_img_path = []  # img绝对地址
    placeholder_path = []  # mask绝对地址
    pix_num = []  # 体积像素
    bbox_cut = []  # 外接矩形
    one_label = []  # label类别
    pix_spacing = []  # 像素间隔
    one_shape = []  # 尺寸
    one_center = []  # 图像坐标中心点
    one_d = []  # 直径

    msk = nb.load(mask_path)
    msk_affine = msk.affine
    data = msk.get_data()

    head = msk.header
    pixdim = head['pixdim'][1:4]
    pixdim = [i for i in pixdim]
    shape = data.shape

    # data0是numpy，对numpy进行外接矩形提取
    labels = measure.label(data)  # 标记连通域

    # 取单个占位
    properties = measure.regionprops(labels)
    # print(len(properties))
    # print(mask_path)
    hcc_ = 'hcc' in Path(mask_path).stem
    if hcc_ and len(properties) > 10:
        pass
    else:
        for p in properties:
            bbox = p.bbox
            if 'hcc' in Path(mask_path).stem:
                label_one = 1
            else:
                label_one = 0
            placeholder_path.append(mask_path)
            ct_img_path.append(img_path)
            pix_num.append(p.area)
            bbox_cut.append(p.bbox)
            one_label.append(label_one)
            pix_spacing.append(pixdim)
            one_shape.append(shape)
            one_center.append([(bbox[3] + bbox[0]) // 2, (bbox[4] + bbox[1]) // 2, (bbox[5] + bbox[2]) // 2])
            one_cut = p.bbox
            one_spc = pixdim
            one_d.append(max((one_cut[3] - one_cut[0]) * one_spc[0], (one_cut[4] - one_cut[1]) * one_spc[1],
                             (one_cut[5] - one_cut[2]) * one_spc[2]))
    return [placeholder_path,
            ct_img_path,
            pix_num,
            bbox_cut,
            one_label,
            pix_spacing,
            one_shape,
            one_center,
            one_d], msk_affine, [[placeholder_path[i],
                                  ct_img_path[i],
                                  pix_num[i],
                                  bbox_cut[i],
                                  one_label[i],
                                  pix_spacing[i],
                                  one_shape[i],
                                  one_center[i],
                                  one_d[i]] for i in range(len(ct_img_path))]


def get_excel():
    """
    获取全部数据的信息表格并保存到excel_path
    :return:
    """
    print("生成信息表中,请稍等:")
    # 获取placeholder mask
    source_ = Path(source_path)
    list_V = [str(i) for i in source_.glob('*V.nii.gz')]
    list_ = [i.replace('V.nii.gz', 'hcc.nii.gz') for i in list_V if Path(i.replace('V.nii.gz', 'hcc.nii.gz')).exists()]
    list_ += [i.replace('V.nii.gz', 'tumor.nii.gz') for i in list_V if
              Path(i.replace('V.nii.gz', 'tumor.nii.gz')).exists()]
    # 统计信息
    placeholder_paths = []  # 占位绝对地址
    volume_paths = []  # ct平扫期绝对地址
    pix_nums = []  # 体积像素
    bbox_cuts = []  # 外接矩形
    one_labels = []  # label类别
    pix_spacings = []  # 像素间隔
    one_shapes = []  # 尺寸
    one_centers = []  # 图像坐标中心点
    diam = []  # 直径
    error_txt = ''
    for i in range(len(list_)):
        # 确认尺寸一致、每套数据完整包含vv、liver、hcc、tumor（可有可无）
        volume = list_[i].replace('hcc.nii.gz', 'V.nii.gz').replace('tumor.nii.gz', 'V.nii.gz')
        liver = list_[i].replace('hcc.nii.gz', 'liver.nii.gz').replace('tumor.nii.gz', 'liver.nii.gz')
        hcc = list_[i].replace('tumor.nii.gz', 'hcc.nii.gz')
        tumor = list_[i].replace('hcc.nii.gz', 'tumor.nii.gz')
        same_size = False
        if Path(volume).exists() and Path(liver).exists():
            volume_size = nb.load(volume).dataobj.shape
            liver_size = nb.load(liver).dataobj.shape

            if Path(tumor).exists() and Path(hcc).exists():
                tumor_size = nb.load(tumor).dataobj.shape
                hcc_size = nb.load(hcc).dataobj.shape
                if volume_size == liver_size == hcc_size == tumor_size:
                    same_size = True
                else:
                    error_txt = '%s 的体积V和掩膜/LABEL尺寸不一致, 已剔除\n' % list_[i]
            elif Path(hcc).exists():
                hcc_size = nb.load(hcc).dataobj.shape
                if volume_size == liver_size == hcc_size:
                    same_size = True
                else:
                    error_txt = '%s 的体积V和掩膜/LABEL尺寸不一致, 已剔除\n' % list_[i]
            elif Path(tumor).exists():
                tumor_size = nb.load(tumor).dataobj.shape
                if volume_size == liver_size == tumor_size:
                    same_size = True
                else:
                    error_txt = '%s 的体积V和掩膜/LABEL尺寸不一致, 已剔除\n' % list_[i]
        # recoder
        with open(error_txt_path, 'a+') as txt_file:
            txt_file.write(error_txt)
        # handle
        if same_size:
            [name, ct_name, pixel, cut, label, spacing, shape, center, d], affine, _ = get_message(list_[i])
            placeholder_paths.extend(name)
            volume_paths.extend(ct_name)
            pix_nums.extend(pixel)
            bbox_cuts.extend(cut)
            one_labels.extend(label)
            pix_spacings.extend(spacing)
            one_shapes.extend(shape)
            one_centers.extend(center)
            diam.extend(d)
            black = "■" * int(i / len(list_) * 25)
            white = "□" * (25 - int(i / len(list_) * 25))
            already = i / len(list_) * 100
            print("\r{}{}{:.2f}%".format(black, white, already), end="")
            time.sleep(0.1)
    dic = {"placeholder": placeholder_paths,
           "volume": volume_paths,
           "pixel": pix_nums,
           "cut": bbox_cuts,
           "label": one_labels,
           "spacing": pix_spacings,
           "shape": one_shapes,
           "center(xyz)": one_centers,
           "d": diam}
    dic_pd = pd.DataFrame(dic)
    dic_pd.to_excel(excel_path)
    print('\n信息表存于%s\n' % excel_path)
    print('get excel down!\n')


def get_txt(exc=excel_path):
    """
    对全部数据的信息表格进行条件筛选并保存到XLSX
    注意：此时演示的是  d>=20mm
    :param exc: 全部数据的信息表格
    :return:
    """
    # 统计数据集（种类、尺寸、数量），并输出文档
    # 筛选
    screened = []
    # 统计所有
    cls_all = 0
    cls_hcc = 0
    cls_fhcc = 0
    # 占位层厚>=3
    than_3h_all = 0
    than_3h_hcc = 0
    than_3h_fhcc = 0
    # 直径<=20mm（微小），20mm<=直径<=50mm（小），50mm<=直径<=128mm（中），直径>=128mm（大）
    little = 0
    small = 0
    mid = 0
    big = 0
    little_hcc = 0
    little_fhcc = 0
    small_hcc = 0
    small_fhcc = 0
    mid_hcc = 0
    mid_fhcc = 0
    big_hcc = 0
    big_fhcc = 0
    exc_data = pd.read_excel(exc, sheet_name=0, header=0, index_col=0)
    for one_msg in exc_data.values:
        cut_ = [int(p) for p in one_msg[2].lstrip('(').rstrip(')').split(',')]  # xmin,ymin,zmin,xmax,ymax,zmax
        # spc_ = [float(p) for p in one_msg[4].lstrip('[').rstrip(']').split(',')]
        # d = max((cut_[3] - cut_[0]) * spc_[0], (cut_[4] - cut_[1]) * spc_[1], (cut_[5] - cut_[2]) * spc_[2])
        d = int(one_msg[-1])
        ch = cut_[5] - cut_[2]
        # 统计所有
        cls_all += 1
        if 'hcc' in Path(one_msg[0]).stem:
            cls_hcc += 1
            # 筛选
            if d >= 20 and ch > 2:
                than_3h_hcc += 1
                than_3h_all += 1
            # 后加
            if 20 < d < 50:
                screened.append(one_msg)
        else:
            cls_fhcc += 1
            # 筛选
            if d >= 5 and ch > 2:
                than_3h_fhcc += 1
                than_3h_all += 1
            # 后加
            if 20 < d < 50:
                screened.append(one_msg)
        # 统计直径（筛选）
        hcc_select = 'hcc' in Path(one_msg[0]).stem and d >= 20
        tumor_select = 'tumor' in Path(one_msg[0]).stem and d >= 5
        # if True: # （不筛选）
        if hcc_select or tumor_select:
            if d < 20:
                little += 1
                if 'hcc' in Path(one_msg[0]).stem:
                    little_hcc += 1
                else:
                    little_fhcc += 1
            elif 20 <= d <= 50:
                small += 1
                if 'hcc' in Path(one_msg[0]).stem:
                    small_hcc += 1
                else:
                    small_fhcc += 1
            elif 50 <= d <= 128:
                mid += 1
                if 'hcc' in Path(one_msg[0]).stem:
                    mid_hcc += 1
                else:
                    mid_fhcc += 1
            elif 128 <= d:
                big += 1
                if 'hcc' in Path(one_msg[0]).stem:
                    big_hcc += 1
                else:
                    big_fhcc += 1
    text = '数据集源位置：%s，\n' \
           '--------------------------------\n' \
           '该数据集肝脏占位总数为%s，其中hcc有%s，非hcc有%s，\n' \
           '条件筛选（hcc直径>20mm，非hcc直径>5mm）的有%s，其中hcc有%s，非hcc有%s，\n' \
           '微小占位（直径d<=20mm）的有%s，其中hcc有%s，非hcc有%s\n' \
           '小占位（20mm<=直径d<=50mm）的有%s，其中hcc有%s，非hcc有%s\n' \
           '中占位（50mm<=直径d<=128mm）的有%s，其中hcc有%s，非hcc有%s\n' \
           '大占位（直径d>=128mm）的有%s，其中hcc有%s，非hcc有%s。\n' \
           % (source_path, cls_all, cls_hcc, cls_fhcc, than_3h_all, than_3h_hcc, than_3h_fhcc,
              little, little_hcc, little_fhcc, small, small_hcc, small_fhcc, mid, mid_hcc, mid_fhcc, big, big_hcc,
              big_fhcc)
    print(text)
    with open(text_path, 'w') as txt_file:
        txt_file.write(text)

    h3_pd = pd.DataFrame(screened)
    h3_pd.to_excel(XLSX)
    print('get txt down!')
