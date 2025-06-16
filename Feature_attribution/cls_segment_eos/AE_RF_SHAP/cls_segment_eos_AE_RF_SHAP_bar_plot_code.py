import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from joblib import load
from pathlib import Path
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


class SHAPFeatureImportance_RF:
    def __init__(self, model):
        self.model = model
        self.explainer = shap.TreeExplainer(model)

    def compute_attributions(self, X_test):
        shap_values = self.explainer.shap_values(X_test)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 0]

        return shap_values

    @staticmethod
    def group_by_prefix(attributions, feature_names):
        if attributions.ndim == 1:
            attributions = attributions.reshape(-1, 1)

        modified_names = []
        for name in feature_names:
            if name.startswith("1_"):
                modified_names.append("A_" + name[2:])
            elif name.startswith("2_"):
                modified_names.append("B_" + name[2:])
            else:
                modified_names.append(name)

        prefixes = sorted(set(["_".join(name.split("_")[:2]) for name in modified_names]))
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
        plt.xlabel('SHAP Value')
        plt.ylabel('Feature')
        plt.title('SHAP Feature Importance')
        plt.gca().invert_yaxis()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', format='pdf')
        plt.show()
        plt.close()

def main():
    # Load test data
    open_path = os.path.join('Feature_attribution/cls_segment_eos/AE_output')
    save_path = os.path.join('Feature_attribution/cls_segment_eos/AE_RF_SHAP/output')
    os.makedirs(save_path, exist_ok= True)
    # Import data
    feature_all = pd.read_csv(os.path.join(open_path, 'cls_segment_eos_ae_output_pairs_features.csv'))
    
    columns = ['Protein1', 'Protein2', 'Label', 'Pairs']
    X = feature_all.drop(columns, axis=1)
    y = feature_all[['Label']]

    # Use all data as test set
    X_test = X
    y_test = y
    feature_names = X_test.columns.tolist()
    # Load the model
    model = load('Feature_attribution/cls_segment_eos/train_model/cls_segment_eos_AE_RF_model.pkl')
    
    # Calculate SHAP
    shap_tool = SHAPFeatureImportance_RF(model)
    shap_values = shap_tool.compute_attributions(X_test)

    # Group
    grouped_vals, group_names = shap_tool.group_by_prefix(shap_values, feature_names)

    pd.DataFrame(grouped_vals, columns=group_names).to_csv(
        os.path.join(save_path, 'cls_segment_eos_AE_RF_SHAP_values.csv'), index=False)

    SHAPFeatureImportance_RF.plot_importance(
        grouped_vals,
        group_names,
        save_path=os.path.join(save_path, 'cls_segment_eos_AE_RF_SHAP_plot.pdf')
    )


# Execute main process
if __name__ == "__main__":
    main()