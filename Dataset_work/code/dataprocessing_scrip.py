#导入并初始化路径
import os
import re
import sys
import time
import torch
import argparse
import subprocess
import numpy as np
import pandas as pd
project_path = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(os.path.join(project_path, 'Dataset_work'))
from protloc_mex_X.ESM2_fr import Esm2LastHiddenFeatureExtractor
from transformers import AutoTokenizer, AutoModelForMaskedLM


parser = argparse.ArgumentParser()
parser.add_argument('--seqf', type=str, required=True)
parser.add_argument('--pairsf', type=str, required=True)
args = parser.parse_args()


#代码的路径
sequence_file = args.seqf
pairs_file = args.pairsf

chunk_folder = os.path.join(project_path, r'Dataset_work\dataset_feature')
protein_ppifile = os.path.join(project_path, r'Dataset_work\dataset_feature\data_feature.h5')


PPI_protein = pd.read_excel(sequence_file)
PPI_protein = PPI_protein[PPI_protein['Sequence'].apply(len)<=4000]
os.makedirs(chunk_folder, exist_ok=True)

### PPI_protein
###===========================================================================
# Initialize the tokenizer and model with the pretrained ESM2 model
tokenizer = AutoTokenizer.from_pretrained(r"E:\Working\Model\esm2_t33_650M_UR50D")
model = AutoModelForMaskedLM.from_pretrained(r"E:\Working\Model\esm2_t33_650M_UR50D", output_hidden_states=True)
feature_extractor = Esm2LastHiddenFeatureExtractor(tokenizer, model,
                                                   compute_cls=True, compute_eos=True, compute_mean=True,
                                                   compute_segments=True)

### 基于循环的蛋白质特征提取，以缓解内存问题
chunk_size=700
# 循环遍历DataFrame的所有分块
for chunk_idx, i in enumerate(range(0, len(PPI_protein), chunk_size)):
    # 获取当前分块的数据
    df_chunk = PPI_protein.iloc[i:i + chunk_size]

    # 对当前分块应用特征提取
    print(f"Processing rows {i} / {len(PPI_protein)} in all {len(PPI_protein)}")
    df_chunk_represent = feature_extractor.get_last_hidden_features_combine(df_chunk, sequence_name='Sequence', batch_size=1)
    
    # 保存当前分块的特征结果到文件夹中的h5文件
    chunk_file = os.path.join(chunk_folder, f'protein_feature_chunk{chunk_idx}.h5')
    df_chunk_represent.to_hdf(chunk_file, key='df', mode='w')

print("Feature extraction for all chunks completed.")


### 所需文件路径定义======================================================================
# 存储所有读取的数据
all_data = []
for file_name in os.listdir(chunk_folder):
    chunk_file = os.path.join(chunk_folder, file_name)
        
    # 检查文件是否存在
    if os.path.exists(chunk_file):
        # 读取HDF5文件
        df = pd.read_hdf(chunk_file, key='df')
        all_data.append(df)
final_df = pd.concat(all_data, ignore_index=True)

#获取需要的特征
pairs = pd.read_excel(pairs_file)
feature_columns = []
patterns_f = ['Entry', r'ESM2_cls\d+', r'ESM2_eos\d+',  r'ESM2_segment\d+']
for pattern in patterns_f:
    for col in final_df.columns:
        if re.match(pattern, col):
            feature_columns.append(col)
feature_all_reserve1 = final_df[feature_columns]

#将列重新排序
new_index = ['Entry']
for prefix in ['ESM2_cls', 'ESM2_segment0_mean', 'ESM2_segment1_mean',
               'ESM2_segment2_mean','ESM2_segment3_mean','ESM2_segment4_mean',
               'ESM2_segment5_mean','ESM2_segment6_mean','ESM2_segment7_mean','ESM2_segment8_mean',
               'ESM2_segment9_mean', 'ESM2_eos']:
    for i in range(1280):
        col_name = f"{prefix}{i}"
        if col_name in feature_all_reserve1.columns:
            new_index.append(col_name)
feature_all_reserve = feature_all_reserve1[new_index]
del feature_all_reserve1, final_df

##蛋白质对数据
test_data = pd.read_excel(pairs_file)
#重命名列名
test_data = pd.merge(test_data, feature_all_reserve, how='left', left_on='Protein1', right_on='Entry')
new_column_names1 = []
for name in test_data.columns:
    if re.match('ESM.', name):
        new_name = '1_' + name
    elif name == 'Entry':
        new_name = 'Entry1'
    else:
        new_name = name
    new_column_names1.append(new_name)
test_data.columns = new_column_names1
#重命名列名
test_data = pd.merge(test_data, feature_all_reserve, how='left', left_on='Protein2', right_on='Entry')
new_column_names2 = []
for name in test_data.columns:
    if re.match('ESM.', name):
        new_name2 = '2_' + name
    else:
        new_name2 = name
    new_column_names2.append(new_name2)
test_data.columns = new_column_names2


#第二个Entry没有设置为2
test_data = test_data.drop(columns=['Entry1', 'Entry'])
test_data.to_hdf(protein_ppifile, key='df', mode='w')
print(f"The datafeature was saved to {protein_ppifile}.")