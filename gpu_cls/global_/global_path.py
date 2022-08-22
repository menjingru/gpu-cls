from pathlib import Path
import os
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 动

#       ↓
# source_path = '/home/deepliver2/Disksdb/JZ/train_data/CT/hcc_data'
# project = '/home/deepliver2/Disksdb/menjingru/dataset/hcc_0'
source_path = 'E:\HCC_data\HCC_data'
project = 'E:\dataset\hcc_0'
#       ↑


# 不动

excel_path = str(Path(project) / 'all_.xlsx')
XLSX = str(Path(project) / 'all_sx.xlsx')
# XLSX = project+'/all_test.xlsx'
text_path = str(Path(project) / 'all_message.txt')
error_txt_path = str(Path(project) / 'error_log.txt')
Path(project).mkdir(exist_ok=True, parents=True)

pre_npy = str(Path(project) / 'in')
out = str(Path(project) / 'out')
inference_file = str(Path(project) / 'infer')
indicators_path = str(Path(out) / 'indicators')
model_path = str(Path(out) / 'model')
Path(out).mkdir(exist_ok=True, parents=True)
Path(indicators_path).mkdir(exist_ok=True, parents=True)
Path(model_path).mkdir(exist_ok=True, parents=True)

pre_npy_img = str(Path(pre_npy) / 'img')
pre_npy_lbl = str(Path(pre_npy) / 'lbl')
Path(pre_npy_img).mkdir(exist_ok=True, parents=True)
Path(pre_npy_lbl).mkdir(exist_ok=True, parents=True)

inference_file_img = str(Path(inference_file) / 'inf_img')
inference_file_mid = str(Path(inference_file) / 'mid')
inference_file_pred = str(Path(inference_file) / 'predict')
Path(inference_file_img).mkdir(exist_ok=True, parents=True)
Path(inference_file_mid).mkdir(exist_ok=True, parents=True)
Path(inference_file_pred).mkdir(exist_ok=True, parents=True)

train_model_path = str(Path(model_path) / 'train_model.pth')
best_model_path = str(Path(model_path) / 'best_model.pth')
