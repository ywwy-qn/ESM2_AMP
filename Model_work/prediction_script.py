import re
import os
import sys
import torch
import json
import argparse
import torch.nn as nn
import numpy as np
import pandas as pd
project_path = os.path.dirname(os.getcwd())
sys.path.append(os.path.join(project_path, 'Model_work'))
from AMPmodel import download_model_weidht
from AMPmodel.dataset import CustomDataset
from AMPmodel.model import AMP_model
from AMPmodel.check import fix_state_dict, model_info
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--model_w', type=str, required=True)
args = parser.parse_args()


model_name = args.model
data_file = os.path.join(project_path, r'Dataset_work\dataset_feature\data_feature.h5')
save_path = os.path.join(project_path, r'Model_work\model_out')
config_file = os.path.join(project_path, r'Model_work\model_configs.json')

with open(config_file, "r") as f:
    model_dicts = json.load(f)
model_dict = model_dicts[model_name]


model_file = args.model_w
if model_file is None:
    for weight_path in model_dict["weight"]:
        if os.path.exists(weight_path):
            model_file = weight_path
            break
    # 如果没有找到存在的权重文件，则下载
    if model_file is None:
        download_model_weidht(model_dict["weight_url"], model_dict["weight_file"])
        model_file = model_dict["weight_file"]
        model_file = os.path.join(project_path, model_file)

### 数据预处理
os.makedirs(save_path, exist_ok=True)
print('开始进行数据预处理中...')
feature_all = pd.read_hdf(data_file, key='df')
feature_columns = []
if model_dict["feature"] == "seg":
    patterns_f = ['Protein1', 'Protein2', 'Pairs', r'1_ESM2_segment\d+', r'2_ESM2_segment\d+']
if model_dict["feature"] == "split":
    patterns_f = ['Protein1', 'Protein2', 'Pairs', r'1_ESM2_cls\d+', r'1_ESM2_segment\d+', r'1_ESM2_eos\d+',  
                  r'2_ESM2_cls\d+', r'2_ESM2_segment\d+', r'2_ESM2_eos\d+']
else:
    print(f'确保model_dict["feature"]必须为 "seg" 或者 "split".')

for pattern in patterns_f:
    for col in feature_all.columns:
        if re.match(pattern, col):
            feature_columns.append(col)
feature_all = feature_all[feature_columns]
PPI_df = feature_all[['Pairs', 'Protein1', 'Protein2']]
PPI_feature = torch.tensor(feature_all.iloc[:, 3:].values, dtype=torch.float32).view(model_dict["feature_shape"])
del feature_all
print('开始进行数据预处理完成...')

### 模型推理
model = AMP_model(input_dim=1280, hidden1_dim=model_dict['hidden1'], hidden2_dim=model_dict['hidden2'], output_dim=1, encoder_type=model_dict['mode']).float().to(device)
checkpoint = torch.load(model_file, map_location=device)
checkpoint = checkpoint["model_state_dict"]
if 'module' in list(checkpoint.keys())[0]:
    checkpoint = fix_state_dict(checkpoint)
model.load_state_dict(checkpoint)

batch_size = 32
predictions_df = model_info(model, PPI_feature, batch_size, device)
PPI_df = pd.concat([PPI_df, predictions_df], axis=1)
del PPI_feature, predictions_df


# 打印评估结果
PPI_df.to_csv(os.path.join(save_path, 'test_predictions_result.csv'), index=False)
print(f"工作完成，预测结果已保存至：{os.path.join(save_path, 'test_predictions_result.csv')}")