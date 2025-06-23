import torch
import numpy as np
import pandas as pd


def load_and_concatenate_pandata(datasetPath, 
                              positive_train_one_colums, positive_train_one_features, positive_train_one_infor,
                              positive_train_two_colums, positive_train_two_features, positive_train_two_infor,
                              positive_test_colums, positive_test_features, positive_test_infor,
                              negative_train_one_colums, negative_train_one_features, negative_train_one_infor,
                              negative_train_two_colums, negative_train_two_features, negative_train_two_infor,
                              negative_test_colums, negative_test_features, negative_test_infor,
                              sample_nums = None):

    # positive_train_one
    test_load11 = np.load(datasetPath + positive_train_one_features)
    feature_columns_load11 = pd.read_excel(datasetPath + positive_train_one_colums)
    feature_columns_save11 = feature_columns_load11.T
    del feature_columns_load11
    test_load11 = pd.DataFrame(test_load11, columns=feature_columns_save11.iloc[0,:])
    feature_infro_load11 = pd.read_excel(datasetPath + positive_train_one_infor)
    feature_positive1 = pd.concat([feature_infro_load11, test_load11], axis=1)
    del test_load11
    del feature_infro_load11
    
    # positive_train_two
    test_load12 = np.load(datasetPath + positive_train_two_features)
    feature_columns_load12 = pd.read_excel(datasetPath + positive_train_two_colums)
    feature_columns_save12 = feature_columns_load12.T
    del feature_columns_load12
    test_load12 = pd.DataFrame(test_load12, columns=feature_columns_save12.iloc[0,:])
    feature_infro_load12 = pd.read_excel(datasetPath + positive_train_two_infor)
    feature_positive2 = pd.concat([feature_infro_load12, test_load12], axis=1)
    del test_load12
    del feature_infro_load12
    
    # positive_test
    test_load13 = np.load(datasetPath + positive_test_features)
    feature_columns_load13 = pd.read_excel(datasetPath + positive_test_colums)
    feature_columns_save13 = feature_columns_load13.T
    del feature_columns_load13
    test_load13 = pd.DataFrame(test_load13, columns=feature_columns_save13.iloc[0,:])
    feature_infro_load13 = pd.read_excel(datasetPath + positive_test_infor)
    feature_positive3 = pd.concat([feature_infro_load13, test_load13], axis=1)
    del test_load13
    del feature_infro_load13
    
    # 整合正数据
    feature_positive = pd.concat([feature_positive1, feature_positive2], axis=0)
    feature_positive = pd.concat([feature_positive, feature_positive3], axis=0)
    del feature_positive1
    del feature_positive2
    del feature_positive3
    
    
    
    # negative_train_one
    test_load21 = np.load(datasetPath + negative_train_one_features)
    feature_columns_load21 = pd.read_excel(datasetPath + negative_train_one_colums)
    feature_columns_save21 = feature_columns_load21.T
    del feature_columns_load21
    test_load21 = pd.DataFrame(test_load21, columns=feature_columns_save21.iloc[0,:])
    feature_infro_load21 = pd.read_excel(datasetPath + negative_train_one_infor)
    feature_negative1 = pd.concat([feature_infro_load21, test_load21], axis=1)
    del test_load21
    del feature_infro_load21
    
    # negative_train_two
    test_load22 = np.load(datasetPath + negative_train_two_features)
    feature_columns_load22 = pd.read_excel(datasetPath + negative_train_two_colums)
    feature_columns_save22 = feature_columns_load22.T
    del feature_columns_load22
    test_load22 = pd.DataFrame(test_load22, columns=feature_columns_save22.iloc[0,:])
    feature_infro_load22 = pd.read_excel(datasetPath + negative_train_two_infor)
    feature_negative2 = pd.concat([feature_infro_load22, test_load22], axis=1)
    del test_load22
    del feature_infro_load22
    
    # negative_test
    test_load23 = np.load(datasetPath + negative_test_features)
    feature_columns_load23 = pd.read_excel(datasetPath + negative_test_colums)
    feature_columns_save23 = feature_columns_load23.T
    del feature_columns_load23
    test_load23 = pd.DataFrame(test_load23, columns=feature_columns_save23.iloc[0,:])
    feature_infro_load23 = pd.read_excel(datasetPath + negative_test_infor)
    feature_negative3 = pd.concat([feature_infro_load23, test_load23], axis=1)
    del test_load23
    del feature_infro_load23
    
    
    feature_negative = pd.concat([feature_negative1, feature_negative2], axis=0)
    feature_negative = pd.concat([feature_negative, feature_negative3], axis=0)
    del feature_negative1
    del feature_negative2
    del feature_negative3
    
    
    feature_all = pd.concat([feature_positive, feature_negative], axis=0)
    del feature_negative
    del feature_positive
    
    if sample_nums == None:
        feature_all = feature_all
    else:
        feature_all = feature_all.sample(n=sample_nums, random_state=42)
    
    
    return feature_all



class CustomDataset1(torch.utils.data.Dataset):
    def __init__(self, data_features, data_label, reshape_shape=(-1, 20, 1280)):
        
        self.features = torch.tensor(data_features.values.reshape(reshape_shape)).float()
        
        
        assert len(data_features) == len(data_label), "The sample size of the feature and label data does not match!"

        
        self.labels = torch.tensor(data_label.values).float()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
    
class CustomDataset2(torch.utils.data.Dataset):
    def __init__(self, data_features, data_label, reshape_shape=(-1, 24, 1280)):
        
        self.features = torch.tensor(data_features.values.reshape(reshape_shape)).float()
        
        
        assert len(data_features) == len(data_label), "The sample size of the feature and label data does not match!"

        
        self.labels = torch.tensor(data_label.values).float()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    

class CustomDataset3(torch.utils.data.Dataset):
    def __init__(self, data_features, data_label):
        
        self.features = torch.tensor(data_features.values).float()
        
        
        assert len(data_features) == len(data_label), "The sample size of the feature and label data does not match!"


        self.labels = torch.tensor(data_label).float()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]