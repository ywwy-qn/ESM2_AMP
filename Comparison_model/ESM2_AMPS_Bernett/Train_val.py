
import numpy as np
import pandas as pd
import re
import os
import sys
import torch
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from AMPmodel.dataset import CustomDataset
from AMPmodel.model import AMP_model
from AMPmodel.check import SaveMetricsAndBestCheckpoints



data_path = "Comparison_model/data/bernett_seg_train_val"


train_data = pd.read_hdf(data_path + "/train_data.h5")
train_data = train_data.sample(n=len(train_data)//10*3, random_state=42)
X_train, y_train = train_data.loc[:, '1_ESM2_segment0_mean0':], train_data[['Label']]
train_dataset = CustomDataset(X_train, y_train, reshape_shape=(-1, 20, 1280))
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
del train_data, X_train, y_train, train_dataset



val_data = pd.read_hdf(data_path + "/val_data.h5")
val_data = val_data.sample(n=len(val_data)//10*3, random_state=42)
X_val, y_val = val_data.loc[:, '1_ESM2_segment0_mean0':], val_data[['Label']]
val_dataset = CustomDataset(X_val, y_val, reshape_shape=(-1, 20, 1280))
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False)
del val_data, X_val, y_val, val_dataset


import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import AdamW
import optuna
import torch.nn.init as init
from torch.nn.utils import clip_grad_norm


def train_model(trial):

    save_trail_path = f"Comparison_model/ESM2_AMPS_Bernett/weights_file/trial_{trial.number}_bestcheckpoint"
    os.makedirs(save_trail_path)

    # Define the hyperparameters required for model training
    lr = trial.suggest_float('lr', low=1e-5, high=1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', low=1e-5, high=1e-4, log=True)
    hidden1_dim = trial.suggest_int('hidden1_dim', low=480, high=640)
    hidden2_dim = trial.suggest_int('hidden2_dim', low=80, high=320)
    print(f"Starting tail work {trial.number}")
    
    eval_metrics = "MCC"
    num_epochs = 50
        
    model = AMP_model(input_dim=1280, hidden1_dim=hidden1_dim, hidden2_dim=hidden2_dim, output_dim=1, encoder_type='transformer').float()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, verbose=True, min_lr=1e-6)
    saver = SaveMetricsAndBestCheckpoints(file_name="val_metrics.xlsx", checkpoint_dir=save_trail_path, top_k=3, metrics_name=eval_metrics)
    
    step = 0
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (features, labels) in enumerate(train_dataloader):
            features, labels = features.to(device), labels.float().to(device)
            optimizer.zero_grad()
            outputs = model(features)
            train_loss = criterion(outputs, labels)
            step += 1
            step_loss = train_loss.item()
            epoch_loss += train_loss.item()
            train_loss.backward()
            optimizer.step()

        avg_train_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch}: Average Train Loss = {avg_train_loss:.4f}")
        
        saver.save_checkpoint(model, optimizer, epoch, val_dataloader, criterion, device)
        # saver.save_checkpoint(model, optimizer, epoch, train_dataloader, criterion, device)


    metrics_df = pd.read_excel(save_trail_path + "/val_metrics.xlsx")

    sorted_metrics_df = metrics_df.sort_values(by=eval_metrics, ascending=False)

    best_mcc = sorted_metrics_df.iloc[0][eval_metrics]        
        
    return best_mcc



study = optuna.create_study(study_name='train model', direction='maximize')
study.optimize(lambda trial: train_model(trial), n_trials=1, n_jobs=1)

# Obtain the best test
best_trial = study.best_trial

# Output the optimal MCC value and the corresponding hyperparameters
print("Best trial:", best_trial.number)
print("Best val MCC:", best_trial.value)
print("Best hyperparameters:")
for key, value in best_trial.params.items():
    print(f"  {key}: {value}")

