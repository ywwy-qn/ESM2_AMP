

import torch
import torch.nn as nn
from torch.utils.data import Dataset

class DNN(nn.Module):
    def __init__(self, input_dim=3600, hidden1_dim=1500, hidden2_dim=500, output_dim=1):
        super(DNN, self).__init__()
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
