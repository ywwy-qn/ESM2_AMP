
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
# train_data = train_data.sample(n=len(train_data)//10*1, random_state=42)
X_train, y_train = train_data.loc[:, '1_ESM2_segment0_mean0':], train_data[['Label']]
train_dataset = CustomDataset(X_train, y_train, reshape_shape=(-1, 20, 1280))
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
del train_data, X_train, y_train, train_dataset


val_data = pd.read_hdf(data_path + "/val_data.h5")
# val_data = val_data.sample(n=len(val_data)//10*1, random_state=42)
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

save_trail_path = "Comparison_model/ESM2_GRU_Bernett/weights_file/bestcheckpoint"


lr = 2.692891894619611e-05
weight_decay = 3.8509404299962344e-05
hidden1_dim = 614
hidden2_dim = 299

eval_metrics = "MCC"
num_epochs = 80

model = AMP_model(input_dim=1280, hidden1_dim=hidden1_dim, hidden2_dim=hidden2_dim, output_dim=1, encoder_type='gru').float()


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
        print(f"Epoch {epoch} Step {step}: Average Train Loss = {step_loss:.4f}")
        epoch_loss += train_loss.item()
        train_loss.backward()
        optimizer.step()

    avg_train_loss = epoch_loss / len(train_dataloader)
    print(f"Epoch {epoch}: Average Train Loss = {avg_train_loss:.4f}")

    saver.save_checkpoint(model, optimizer, epoch, val_dataloader, criterion, device)
#     saver.save_checkpoint(model, optimizer, epoch, train_dataloader, criterion, device)

del train_dataloader, val_dataloader


metrics_df = pd.read_excel(save_trail_path + "/val_metrics.xlsx")

sorted_metrics_df = metrics_df.sort_values(by=eval_metrics, ascending=False)

best_mcc = sorted_metrics_df.iloc[0]["MCC"]      
best_epoch = sorted_metrics_df.iloc[0]["Epoch"]

print(f"Best model is epoch：{best_epoch}, MCC: {best_mcc}")




data_path = "Comparison_model/data/bernett_seg_test"


test_data = pd.read_hdf(data_path + "/test_data.h5")
# test_data = test_data.sample(n=len(test_data)//10*1, random_state=42)
X_test, y_test = test_data.loc[:, '1_ESM2_segment0_mean0':], test_data[['Label']]
test_dataset = CustomDataset(X_test, y_test, reshape_shape=(-1, 20, 1280))
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=True)
del test_data, X_test, y_test, test_dataset



model = AMP_model(input_dim=1280, hidden1_dim=hidden1_dim, hidden2_dim=hidden2_dim, output_dim=1, encoder_type='gru').float()
model_path = save_trail_path + f"/checkpoint_epoch_{best_epoch:.0f}.pth"



checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs")
    model = nn.DataParallel(model)




epoch = 999    
model.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, verbose=True, min_lr=1e-6)
save_path = 'Comparison_model/ESM2_GRU_Bernett'
saver = SaveMetricsAndBestCheckpoints(file_name="gru_test_metrics.xlsx", checkpoint_dir=save_path, top_k=3, metrics_name=eval_metrics)
saver.save_checkpoint(model, optimizer, epoch, test_dataloader, criterion, device)

