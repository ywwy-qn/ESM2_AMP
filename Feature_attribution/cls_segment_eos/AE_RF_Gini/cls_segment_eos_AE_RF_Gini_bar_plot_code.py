

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import optuna
import pandas as pd
from joblib import dump, load

open_path = os.path.join('Feature_attribution/cls_segment_eos/AE_output')
save_path = os.path.join('Feature_attribution/cls_segment_eos/AE_RF_Gini/output')
os.makedirs(save_path, exist_ok= True)
# Import data
feature_all = pd.read_csv(os.path.join(open_path, 'cls_segment_eos_ae_output_pairs_features.csv'))

columns = ['Protein1', 'Protein2', 'Label', 'Pairs']
X = feature_all.drop(columns, axis=1)
y = feature_all[['Label']]

# Use all data as test set
X_test = X
y_test = y

# Load the model
loaded_best_classifier = load('Feature_attribution/cls_segment_eos/train_model/cls_segment_eos_AE_RF_model.pkl')



feature_importances = loaded_best_classifier.feature_importances_
feature_names = X_test.columns

prefixes = ['_'.join(f.split('_')[:2]) for f in feature_names]

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Prefix': prefixes,
    'Importance': feature_importances
})

grouped_importance = importance_df.groupby('Prefix')['Importance'].mean().reset_index()
grouped_importance = grouped_importance.sort_values(by='Importance', ascending=False)
importance_df.to_csv(save_path + '/cls_segment_eos_AE_RF_Gini_value.csv', index=False)

grouped_importance['Prefix'] = grouped_importance['Prefix'].str.replace('1_', 'A_')
grouped_importance['Prefix'] = grouped_importance['Prefix'].str.replace('2_', 'B_')


# Visualization
plt.figure(figsize=(8, 12))

ax = grouped_importance.sort_values(by='Importance', ascending=True).plot(
    kind='barh',
    x='Prefix',
    y='Importance',
    title='Feature Group Importance',
    color='skyblue',
    width=0.8,
    figsize=(8, 12)
)

plt.xlabel('Total Gini Importance', fontsize=12)
plt.ylabel('')
plt.yticks(fontsize=10)
plt.xticks(fontsize=10)
plt.grid(axis='x', linestyle='--', alpha=0.6)


plt.tight_layout(pad=2)


plt.savefig(save_path + '/cls_segment_eos_AE_RF_Gini_plot.pdf', bbox_inches='tight')
plt.show()