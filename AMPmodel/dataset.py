import numpy as np
import pandas as pd
import re
import torch
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class CustomDataset(Dataset):
    def __init__(self, data_features, data_label, reshape_shape=None):
        if reshape_shape:
            self.features = torch.tensor(data_features.values.reshape(reshape_shape)).float().to(device)
        else:
            self.features = torch.tensor(data_features.values).float().to(device)

        assert len(data_features) == len(data_label), "The sample size of the feature and label data does not match！"
        self.labels = torch.tensor(data_label.values).float().to(device)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_dataset(mode='split',
                 feature_file='data/real_test_dataset_features.h5',
                 sample_file='data/real_test_dataset_samples.xlsx'):
    assert mode in ['split', 'segment', 'mean'],

    protein_feature = pd.read_hdf(feature_file, key='df')
    test_sample = pd.read_excel(sample_file)

    if mode == 'split':
        patterns = [r'ESM2_cls\d+', r'ESM2_eos\d+', r'ESM2_segment\d+']
        reshape_shape = (-1, 24, 1280)
        feature_prefix = 'ESM2_'
    elif mode == 'segment':
        patterns = [r'ESM2_segment\d+']
        reshape_shape = (-1, 20, 1280)
        feature_prefix = 'ESM2_'
    elif mode == 'mean':
        patterns = [r'ESM2_mean\d+']
        reshape_shape = None
        feature_prefix = 'ESM2_'

    feature_columns = ['Entry']
    for pattern in patterns:
        feature_columns += [col for col in protein_feature.columns if re.match(pattern, col)]
    feature_all = protein_feature[feature_columns]

    test_sample = pd.merge(test_sample, feature_all, how='left',
                           left_on='Protein1', right_on='Entry')
    test_sample = test_sample.rename(columns=lambda x: '1_' + x if x.startswith(feature_prefix) else x)
    test_sample = test_sample.rename(columns={'Entry': 'Entry1'})

    test_sample = pd.merge(test_sample, feature_all, how='left',
                           left_on='Protein2', right_on='Entry')
    test_sample = test_sample.rename(columns=lambda x: '2_' + x if x.startswith(feature_prefix) else x)
    test_sample = test_sample.rename(columns={'Entry': 'Entry2'})

    test_sample = test_sample.drop(columns=['Entry1', 'Entry2'])

    if test_sample.isnull().values.any():
        print("The features of test_sample contain null values. Dropping them...")
        test_sample = test_sample.dropna()
    else:
        print("No null values.")

    feature_columns = [col for col in test_sample.columns if col.startswith('1_' + feature_prefix) or col.startswith('2_' + feature_prefix)]
    data_features = test_sample[feature_columns]
    data_label = test_sample[['Label']]

    return CustomDataset(data_features, data_label, reshape_shape=reshape_shape)
