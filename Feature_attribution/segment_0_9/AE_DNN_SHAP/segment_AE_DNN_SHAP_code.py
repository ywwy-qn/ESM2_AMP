import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import shap
import matplotlib.pyplot as plt
from joblib import load
from pathlib import Path
import sys
from tqdm import tqdm
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))



from Feature_attribution.segment_0_9.AE_DNN.DNN_model import DNN

class SHAPFeatureImportance:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def compute_attributions(self, X_test, baseline_type='random', baseline_samples=200, batch_size=32):

        if baseline_type == 'zeros':
            baseline = np.zeros((1, X_test.shape[1]))
        elif baseline_type == 'mean':
            baseline = np.mean(X_test, axis=0).reshape(1, -1)
        else:
            baseline = X_test.sample(n=min(baseline_samples, len(X_test)), random_state=42)

        baseline_tensor = torch.tensor(baseline.values).float().to(self.device)

        # Initialize the SHAP interpreter
        explainer = shap.DeepExplainer(
            model=self.model,
            data=baseline_tensor
        )

        all_shap_values = []
        for i in tqdm(range(0, len(X_test), batch_size), desc="Computing SHAP values"):
            batch = X_test.iloc[i:i + batch_size]
            batch_tensor = torch.tensor(batch.values).float().to(self.device)
            batch_shap = explainer.shap_values(batch_tensor)

            if isinstance(batch_shap, list):
                batch_shap = batch_shap[0]
            if batch_shap.ndim == 3:
                batch_shap = batch_shap[:, :, 0]

            all_shap_values.append(batch_shap)

        attributions = np.concatenate(all_shap_values, axis=0)
        if attributions.ndim == 1:
            attributions = attributions.reshape(-1, 1)

        return attributions

    @staticmethod
    def group_by_prefix(attributions, feature_names):
        if attributions.ndim == 1:
            attributions = attributions.reshape(-1, 1)

        modified_names = []
        for name in feature_names:
            if name.startswith('1_'):
                modified_names.append('A_' + name[2:])
            elif name.startswith('2_'):
                modified_names.append('B_' + name[2:])
            else:
                modified_names.append(name)

        prefixes = list(set(["_".join(name.split("_")[:2]) for name in modified_names]))
        prefixes.sort()

        grouped_attributions = np.zeros((attributions.shape[0], len(prefixes)))
        for j, prefix in enumerate(prefixes):
            cols = [i for i, name in enumerate(modified_names) if name.startswith(prefix)]
            if cols:
                grouped_attributions[:, j] = np.mean(np.abs(attributions[:, cols]), axis=1)
            else:
                grouped_attributions[:, j] = 0

        return grouped_attributions, prefixes

    @staticmethod
    def plot_importance(mean_attributions, feature_names, save_path=None):

        if mean_attributions.ndim > 1:
            mean_attributions = np.mean(mean_attributions, axis=0)

        sorted_idx = np.argsort(mean_attributions)[::-1]
        sorted_names = [feature_names[i] for i in sorted_idx]
        sorted_values = mean_attributions[sorted_idx]

        plt.figure(figsize=(8, 12))
        plt.barh(range(len(sorted_names)), sorted_values, align='center')
        plt.yticks(range(len(sorted_names)), sorted_names)
        plt.xlabel('Average Absolute SHAP Value')
        plt.ylabel('Feature')
        plt.title('SHAP Feature Importance')
        plt.gca().invert_yaxis()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', format='pdf')
        plt.show()
        plt.close()


def main():
    open_path = os.path.join('Feature_attribution/segment_0_9/AE_output')
    save_path = os.path.join('Feature_attribution/segment_0_9/AE_DNN_SHAP/output')
    os.makedirs(save_path, exist_ok= True)
    # Import data
    feature_all = pd.read_csv(os.path.join(open_path, 'segment_ae_output_pairs_features.csv'))
    columns_to_drop = ['Protein1', 'Protein2', 'Label', 'Pairs']
    X_test = feature_all.drop(columns_to_drop, axis=1)
    feature_names = X_test.columns.tolist()
    
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DNN(input_dim=3000, hidden1_dim=1796, hidden2_dim=382, output_dim=1).to(device)
    model.load_state_dict(torch.load('Feature_attribution/segment_0_9/train_model/Segment_ae_DNN_model.pth'))

    # Initialize the SHAP analyzer
    shap_analyzer = SHAPFeatureImportance(model, device)

    # Calculate the SHAP value (using 200 random samples as the baseline)
    print("Computing SHAP values using 200 random samples as baseline...")
    attributions = shap_analyzer.compute_attributions(
        X_test,
        baseline_type='random',
        baseline_samples=200,
        batch_size=32
    )

    grouped_attributions, group_names = shap_analyzer.group_by_prefix(attributions, feature_names)

    pd.DataFrame(grouped_attributions, columns=group_names).to_csv(
        os.path.join(save_path, 'segment_AE_DNN_SHAP_values.csv'), index=False)

    shap_analyzer.plot_importance(
        grouped_attributions,
        group_names,
        save_path=os.path.join(save_path, 'segment_AE_DNN_SHAP_plot.pdf')
    )


if __name__ == "__main__":
    main()
