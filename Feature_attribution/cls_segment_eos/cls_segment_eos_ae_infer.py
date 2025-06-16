import os
import torch
import pandas as pd
import re
import sys
from pathlib import Path


project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# import module
from Feature_attribution.dataset import load_dataset_before_ae, load_dataset_after_ae

from Feature_attribution.Autoencoder_model import Swish, ResBlock, AutoEncoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# Build the correct file path
feature_file = os.path.join(project_root, "model_pred/data", "real_test_dataset_features.h5")
sample_file = os.path.join(project_root, "model_pred/data", "real_test_dataset_samples.xlsx")

data_protein, flattened_data, full_names_df = load_dataset_before_ae(
    mode='split',
    feature_file=feature_file
)


data_feature_tensor = torch.from_numpy(flattened_data).float()

ae_model = AutoEncoder(input_dim=1280, latent_dim=150)

# loading model
weights_path = 'Feature_attribution/cls_segment_eos/train_model/cls_segment_eos_AutoEncoder_model.pth'
ae_model.load_state_dict(torch.load(weights_path))
del flattened_data

ae_model.eval()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ae_model.to(device)

data_feature_tensor = data_feature_tensor.to(device)



with torch.no_grad():
    z, _ = ae_model(data_feature_tensor)


z_npy = z.cpu().numpy()


result_df = load_dataset_after_ae(data_protein, full_names_df, z_npy)




test_sample = pd.read_excel(sample_file)

test_sample = pd.merge(test_sample, result_df, how='left',
                                  left_on='Protein1', right_on='new_name')


new_column_names1 = []

for name in test_sample.columns:
    if name == 'new_name':
        new_name = 'new_name1'
    elif name in ['Protein1', 'Protein2', 'Label', 'Pairs']:
        new_name = name
    else:
        new_name = '1_' + name

    new_column_names1.append(new_name)

test_sample.columns = new_column_names1


test_sample = pd.merge(test_sample, result_df, how='left',
                                  left_on='Protein2', right_on='new_name')


new_column_names2 = []

for name in test_sample.columns:
    if re.match('1_', name):
        new_name2 = name
    elif name in ['Protein1', 'Protein2', 'Label', 'Pairs', 'new_name1']:
        new_name2 = name
    elif name == 'new_name':
        new_name2 = 'new_name2'
    else:
        new_name2 = '2_' + name
        
    
    new_column_names2.append(new_name2)


test_sample.columns = new_column_names2


test_sample = test_sample.drop(columns=['new_name1', 'new_name2'])

save_path = 'Feature_attribution/cls_segment_eos/AE_output'
os.makedirs(save_path, exist_ok=True)
test_sample.to_csv(save_path + '/cls_segment_eos_ae_output_pairs_features.csv', index=False)


