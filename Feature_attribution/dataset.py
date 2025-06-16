import numpy as np
import pandas as pd
import re
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class CustomDataset(Dataset):
    def __init__(self, data_features, data_label, reshape_shape=None):
        if reshape_shape:
            self.features = torch.tensor(data_features.values.reshape(reshape_shape)).float().to(device)
        else:
            self.features = torch.tensor(data_features.values).float32().to(device)

        assert len(data_features) == len(data_label), "The sample size of the feature and label data does not match!"
        self.labels = torch.tensor(data_label.values).float().to(device)


    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_dataset_before_ae(mode='split',
                 feature_file='data/real_test_dataset_features.h5'):
    assert mode in ['split', 'segment'], "The mode parameter must be 'split' and 'segment'."

    protein_feature = pd.read_hdf(feature_file, key='df')
    # test_sample = pd.read_excel(sample_file)


    if mode == 'split':
        patterns = [r'ESM2_cls\d+', r'ESM2_eos\d+', r'ESM2_segment\d+']
        suffixes = ['_cls', '_segment0', '_segment1', '_segment2', '_segment3', '_segment4', 
                    '_segment5', '_segment6', '_segment7', '_segment8', '_segment9', '_eos']
        feature_prefix = 'ESM2_'
        start_feature = 'ESM2_cls0'
    elif mode == 'segment':
        patterns = [r'ESM2_segment\d+']
        suffixes = ['_segment0', '_segment1', '_segment2', '_segment3', '_segment4', 
                    '_segment5', '_segment6', '_segment7', '_segment8', '_segment9']
        feature_prefix = 'ESM2_'
        start_feature = 'ESM2_segment0_mean0'


    feature_columns = ['Entry']
    for pattern in patterns:
        feature_columns += [col for col in protein_feature.columns if re.match(pattern, col)]
    feature_all = protein_feature[feature_columns]
    
    
    data_protein = feature_all[['Entry']]
    # data_protein.to_excel(r'D:\Aywwy\wy\other_dataset\Seg_TreeSHAP\Seg_AE\seg_ae_pretrain\pretrain_result\seg_AE_infer\test1024_data_protein.xlsx', index=False)

    
    sample_names = protein_feature['Entry']


    feature_all = feature_all.set_index(['Entry'])


    features = feature_all.loc[:, start_feature:]
    flattened_data = features.values.reshape(-1, 1280)


    full_names = [f"{name}{suffix}" for name in sample_names for suffix in suffixes]

    full_names_df = pd.DataFrame(full_names, columns=['Name'])

    return data_protein, flattened_data, full_names_df


def load_dataset_after_ae(data_protein, full_names_df, z_npy):

    data_all = pd.concat([full_names_df, pd.DataFrame(z_npy)], axis=1)
    data_protein_list = data_protein['Entry'].to_list()
    combined_features_list = []

    for protein in tqdm(data_protein_list, desc="Processing Proteins", unit="protein"):
        related_rows = data_all[data_all['Name'].str.contains(protein)]

        if related_rows.empty:
            print(f"Warning: No rows found for protein {protein}.")
            continue

        suffixes = related_rows['Name'].str.split('_').str[-1]
        
        combined_features = {}

        for suffix in suffixes:
            matching_rows = related_rows[related_rows['Name'].str.endswith(suffix)]
            
            features = matching_rows.iloc[:, 1:].copy()
            num_features = features.shape[1]
            
            new_columns = [f"{suffix}_{i}" for i in range(num_features)]
            features.columns = new_columns
            
            for col in features.columns:
                if col not in combined_features:
                    combined_features[col] = features.iloc[0][col]
        
        combined_features_list.append(combined_features)


    result_df = pd.DataFrame(combined_features_list)
    result_df.insert(0, 'new_name', data_protein_list)



    return result_df
