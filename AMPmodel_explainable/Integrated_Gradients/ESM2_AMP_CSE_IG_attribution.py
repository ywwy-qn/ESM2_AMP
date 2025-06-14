import os
import numpy as np
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import sys
from pathlib import Path


project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))


# Import module
from AMPmodel.dataset import load_dataset
from AMPmodel.model import AMP_model
from AMPmodel.check import fix_state_dict

# 设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


feature_file = os.path.join(project_root, "model_pred/data", "real_test_dataset_features.h5")
sample_file = os.path.join(project_root, "model_pred/data", "real_test_dataset_samples.xlsx")

test_dataset = load_dataset(
    mode='split',
    feature_file=feature_file,
    sample_file=sample_file
)


test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# loading model
model = AMP_model(input_dim=1280, hidden1_dim=480, hidden2_dim=240, output_dim=1, encoder_type="transformer").float().to(device)
checkpoint = torch.load("model_pred/weights_file/ESM2_AMP_CSE.pth")

if 'module' in list(checkpoint['model_state_dict'].keys())[0]:
    checkpoint['model_state_dict'] = fix_state_dict(checkpoint['model_state_dict'])

model.load_state_dict(checkpoint['model_state_dict'])


# IG
class Dim1ImportanceAnalyzer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.ig = IntegratedGradients(model)
        
    def compute_baseline(self, dataloader, n_samples=200):
        """Two hundred samples were randomly selected to calculate the baseline"""
        all_features = []
        for features, _ in dataloader:
            all_features.append(features)
            if sum([f.shape[0] for f in all_features]) >= n_samples:
                break
                
        X_all = torch.cat(all_features, dim=0)
        np.random.seed(0)
        sample_idx = np.random.choice(X_all.shape[0], size=min(n_samples, X_all.shape[0]), replace=False)
        baseline = X_all[sample_idx].mean(dim=0, keepdim=True).to(self.device)
        return baseline
    
    def compute_dim1_importance(self, dataloader, baseline=None):
        """
        - dim1_imp: numpy array shape(24,)
        - all_attributions: he original attribution of all samples
        """
        if baseline is None:
            baseline = torch.zeros((1, 24, 1280)).to(self.device)
        
        all_attributions = []
        for features, _ in dataloader:
            features = features.float().to(self.device)
            
            # Calculation attribution (batch, 20, 1280)
            attributions = self.ig.attribute(
                inputs=features,
                baselines=baseline,
                n_steps=25,
                internal_batch_size=32
            )
            
            batch_imp = torch.mean(torch.abs(attributions), dim=2)
            all_attributions.append(attributions.cpu().detach().numpy())
            

        dim1_imp = np.mean(np.abs(np.concatenate(all_attributions, axis=0)), axis=(0, 2))
        return dim1_imp, np.concatenate(all_attributions, axis=0)


# Initialize the analyzer and calculate the baseline
analyzer = Dim1ImportanceAnalyzer(model, device)
baseline = analyzer.compute_baseline(test_dataloader, n_samples=200)
    
# Calculate the importance of features
dim1_imp, all_attributions = analyzer.compute_dim1_importance(test_dataloader, baseline)

save_path = os.path.join('AMPmodel_explainable/Integrated_Gradients/output')
os.makedirs(save_path, exist_ok=True)

# Save
dim1_importance_df = pd.DataFrame(dim1_imp, columns=['value'])
label_name = ['A_cls', 'A_segment0', 'A_segment1', 'A_segment2', 'A_segment3', 'A_segment4', 'A_segment5', 
              'A_segment6', 'A_segment7', 'A_segment8', 'A_segment9', 'A_eos', 'B_cls', 'B_segment0', 
              'B_segment1', 'B_segment2', 'B_segment3', 'B_segment4', 'B_segment5', 'B_segment6', 'B_segment7', 
              'B_segment8', 'B_segment9', 'B_eos']
dim1_importance_df['feature'] = label_name
dim1_importance_df.to_excel(save_path + '/ESM2_AMP_CSE_IG.xlsx', index=False)
np.save(os.path.join(save_path, "ESM2_AMP_CSE_attributions.npy"), all_attributions)

# Visualization

# Sort by value from largest to smallest
dim1_importance_df_sorted = dim1_importance_df.sort_values('value', ascending=False)

plt.figure(figsize=(6, 10))
sns.barplot(x='value', 
            y='feature', 
            data=dim1_importance_df_sorted,
            color='#20AEDD',
            order=dim1_importance_df_sorted['feature'])

plt.title('ESM2_AMP_CSE model features mean IG values', fontsize=16)
plt.xlabel('Feature Importance (IG value)', fontsize=14) 
plt.ylabel('Value', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()


plt.savefig(save_path + '/ESM2_AMP_CSE_model_features_mean_IG_values.pdf', bbox_inches='tight')
plt.show()






