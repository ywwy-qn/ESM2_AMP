import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
from joblib import load
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))


from Feature_attribution.segment_0_9.AE_DNN.DNN_model import DNN

class IGFeatureImportance:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.ig = IntegratedGradients(model)
    
    def init_ig_analysis(self, X_train_tensor, target_label=None):
        self.model.eval()
        if isinstance(X_train_tensor, pd.DataFrame):
            X_train_tensor = torch.tensor(X_train_tensor.values, dtype=torch.float32)
        # Two hundred samples were randomly selected as baselines
        np.random.seed(0)
        background_index = np.random.choice(X_train_tensor.shape[0], size=X_train_tensor.shape[0], replace=True)
        # background_index = np.random.choice(X_train_tensor.shape[0], size=200, replace=False)
        baseline_data = X_train_tensor[background_index]

        device = next(self.model.parameters()).device
        X_train_tensor = X_train_tensor.to(device)
        baseline_data = baseline_data.to(device)

        # Initialize IG
        self.ig = IntegratedGradients(self.model)

        attributions_all = self.ig.attribute(inputs=X_train_tensor,
                                        baselines=baseline_data,
                                        target=target_label,
                                        n_steps=50,
                                        internal_batch_size=64)
    
        return attributions_all
    
    @staticmethod
    def group_by_prefix(attributions, feature_names):
        """Group by feature prefixes and take the average absolute value"""
        if isinstance(attributions, torch.Tensor):
            attributions = attributions.cpu().numpy()
        prefixes = []
        for f in feature_names:
            parts = f.split('_')
            if parts[0] == '1':
                parts[0] = 'A'
            elif parts[0] == '2':
                parts[0] = 'B'
            prefixes.append("_".join(parts[:2]))

        unique_prefixes = sorted(set(prefixes))


        grouped_attributions = np.zeros((attributions.shape[0], len(unique_prefixes)))
        for j, prefix in enumerate(unique_prefixes):
            cols = [i for i, f in enumerate(feature_names)
                    if f.startswith(prefix.replace("A_", "1_").replace("B_", "2_"))]
            grouped_attributions[:, j] = np.mean(np.abs(attributions[:, cols]), axis=1)

        return grouped_attributions, unique_prefixes


    @staticmethod
    def plot_importance(mean_attributions, feature_names, save_path=None):
        """Draw a bar chart of feature importance"""
        # Calculate the global average importance
        global_importance = np.mean(mean_attributions, axis=0)
        
        # Sort
        sorted_idx = np.argsort(global_importance)[::-1]
        sorted_names = [feature_names[i] for i in sorted_idx]
        sorted_values = global_importance[sorted_idx]


        # Plot
        plt.figure(figsize=(8, 12))
        plt.barh(range(len(sorted_names)), sorted_values, align='center')
        plt.yticks(range(len(sorted_names)), sorted_names)
        plt.ylabel('Feature')
        plt.xlabel('IG Attribution')
        plt.title('Segment0-9 Integrated Gradients Feature Importance')
        plt.gca().invert_yaxis()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', format='pdf')
        plt.show()
        # plt.close()

def main():
    open_path = os.path.join('Feature_attribution/segment_0_9/AE_output')
    save_path = os.path.join('Feature_attribution/segment_0_9/AE_DNN_IG/output')
    os.makedirs(save_path, exist_ok= True)
    # Import data
    feature_all = pd.read_csv(os.path.join(open_path, 'segment_ae_output_pairs_features.csv'))
    columns_to_drop = ['Protein1', 'Protein2', 'Label', 'Pairs']
    X_test = feature_all.drop(columns_to_drop, axis=1)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    feature_names = X_test.columns.tolist()
    
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DNN(input_dim=3000, hidden1_dim=1796, hidden2_dim=382, output_dim=1).to(device)
    model.load_state_dict(torch.load('Feature_attribution/segment_0_9/train_model/Segment_ae_DNN_model.pth'))
    
    # Initialize the IG analyzer
    ig_analyzer = IGFeatureImportance(model, device)
    
    # Calculate the attribution value
    attributions = ig_analyzer.init_ig_analysis(X_test_tensor)

    grouped_attributions, group_names = ig_analyzer.group_by_prefix(attributions, feature_names)

    pd.DataFrame(grouped_attributions, columns=group_names).to_csv(
        os.path.join(save_path, 'IG_attributions.csv'), index=False)


    ig_analyzer.plot_importance(
        grouped_attributions,
        group_names,
        save_path=os.path.join(save_path, 'Segment0_9_Integrated_Gradients_Feature_Importance_plot.pdf')
    )
main()
