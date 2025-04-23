import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef


class DNN(nn.Module):
    def __init__(self, input_dim=3000, hidden1_dim=1500, hidden2_dim=500, output_dim=1):
        super(MLP, self).__init__()
        self.mlp_layers = nn.Sequential(
            nn.Linear(input_dim, hidden1_dim),
            nn.ReLU(),
            nn.Linear(hidden1_dim, hidden2_dim),
            nn.ReLU(),
            nn.Linear(hidden2_dim, output_dim))
        
    def forward(self, X):
        return self.mlp_layers(X)

class CustomDataset(Dataset):
    def __init__(self, data_features, data_label=None):
        self.features = torch.tensor(data_features.values).float()
        self.has_labels = data_label is not None
        if self.has_labels:
            self.labels = torch.tensor(data_label).float()
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.has_labels:
            return self.features[idx], self.labels[idx]
        return self.features[idx]

def load_test_data():

    test_feature_all = pd.read_csv('ae_output_pairs_features.csv')
    
    X_test = test_feature_all.drop(columns=['protein1', 'protein2', 'Label', 'Pairs'])
    y_test = test_feature_all[['Label']] if 'Label' in test_feature_all.columns else None
    
    return X_test, y_test

def predict(model, test_loader, device):

    model.eval()
    y_pred_list = []
    y_prob_list = []
    y_true_list = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)
            y_true_list.extend(labels.cpu().numpy())
                
            outputs = model(features)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            y_prob_list.extend(probs)
            y_pred_list.extend(preds)
    

    metrics = {
        'accuracy': accuracy_score(y_true_list, y_pred_list),
        'recall': recall_score(y_true_list, y_pred_list, pos_label=1),
        'f1': f1_score(y_true_list, y_pred_list, pos_label=1),
        'auc': roc_auc_score(y_true_list, y_prob_list),
        'mcc': matthews_corrcoef(y_true_list, y_pred_list)
    }
    

    print("\nEvaluation Metrics:")
    print("=" * 40)
    for name, value in metrics.items():
        print(f"{name:>10}: {value:.4f}")
    print("=" * 40 + "\n")
    
    return {
        'predictions': y_pred_list,
        'probabilities': y_prob_list,
        'true_labels': y_true_list,
        'metrics': metrics
    }
