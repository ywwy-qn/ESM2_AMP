### Initialization Process ===========================================================
print("Initialization")
import re
import os
import sys
import torch
import json
import argparse
import torch.nn as nn
import numpy as np
import pandas as pd

# Configure project paths
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import custom modules
from AMPmodel.dataset import load_and_concatenate_pandata, CustomDataset3
from AMPmodel.model import DNN
from AMPmodel.check import setup_logger, evaluate_model, ensure_2d_labels, calculate_cv_loss, save_fold_step_loss_as_csv

# Set device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_path = "Model_work/data"
outputPath = "Model_work/ESM2_DPM/output"

# Example data loading for 5-fold cross-validation using pandataset
feature_all = load_and_concatenate_pandata(datasetPath = dataset_path, 
                                        positive_train_one_colums = "/train_pairs_mean_feature/train_pairs_mean_feature/SuppA_positive/one_part/Homo_sapiens_pan_train_onepart_positive_pairs_columns_mean.xlsx", 
                                        positive_train_one_features = "/train_pairs_mean_feature/train_pairs_mean_feature/SuppA_positive/one_part/Homo_sapiens_pan_train_onepart_positive_pairs_features_mean.npy", 
                                        positive_train_one_infor = "/train_pairs_mean_feature/train_pairs_mean_feature/SuppA_positive/one_part/Homo_sapiens_pan_train_onepart_positive_pairs_infor_mean.xlsx",
                                        positive_train_two_colums = "/train_pairs_mean_feature/train_pairs_mean_feature/SuppA_positive/two_part/Homo_sapiens_pan_train_twopart_positive_pairs_columns_mean.xlsx", 
                                        positive_train_two_features = "/train_pairs_mean_feature/train_pairs_mean_feature/SuppA_positive/two_part/Homo_sapiens_pan_train_twopart_positive_pairs_features_mean.npy", 
                                        positive_train_two_infor = "/train_pairs_mean_feature/train_pairs_mean_feature/SuppA_positive/two_part/Homo_sapiens_pan_train_twopart_positive_pairs_infor_mean.xlsx",
                                        positive_test_colums = "/test_pairs_mean_feature/test_pairs_mean_feature/positive_test_mean/Homo_sapiens_pan_test_positive_pairs_columns_mean.xlsx", 
                                        positive_test_features = "/test_pairs_mean_feature/test_pairs_mean_feature/positive_test_mean/Homo_sapiens_pan_test_positive_pairs_features_mean.npy", 
                                        positive_test_infor = "/test_pairs_mean_feature/test_pairs_mean_feature/positive_test_mean/Homo_sapiens_pan_test_positive_pairs_infor_mean.xlsx",
                                        negative_train_one_colums = "/train_pairs_mean_feature/train_pairs_mean_feature/SuppB_negative/one_part/Homo_sapiens_pan_train_onepart_negative_pairs_columns_mean.xlsx", 
                                        negative_train_one_features = "/train_pairs_mean_feature/train_pairs_mean_feature/SuppB_negative/one_part/Homo_sapiens_pan_train_onepart_negative_pairs_features_mean.npy", 
                                        negative_train_one_infor = "/train_pairs_mean_feature/train_pairs_mean_feature/SuppB_negative/one_part/Homo_sapiens_pan_train_onepart_negative_pairs_infor_mean.xlsx",
                                        negative_train_two_colums = "/train_pairs_mean_feature/train_pairs_mean_feature/SuppB_negative/two_part/Homo_sapiens_pan_train_twopart_negative_pairs_columns_mean.xlsx", 
                                        negative_train_two_features = "/train_pairs_mean_feature/train_pairs_mean_feature/SuppB_negative/two_part/Homo_sapiens_pan_train_twopart_negative_pairs_features_mean.npy", 
                                        negative_train_two_infor = "/train_pairs_mean_feature/train_pairs_mean_feature/SuppB_negative/two_part/Homo_sapiens_pan_train_twopart_negative_pairs_infor_mean.xlsx",
                                        negative_test_colums = "/test_pairs_mean_feature/test_pairs_mean_feature/negative_test_mean/Homo_sapiens_pan_test_negative_pairs_columns_mean.xlsx", 
                                        negative_test_features = "/test_pairs_mean_feature/test_pairs_mean_feature/negative_test_mean/Homo_sapiens_pan_test_negative_pairs_features_mean.npy", 
                                        negative_test_infor = "/test_pairs_mean_feature/test_pairs_mean_feature/negative_test_mean/Homo_sapiens_pan_test_negative_pairs_infor_mean.xlsx",
                                        sample_nums = None)



train_data_all = feature_all.set_index(['Pairs'])
train_data_features = train_data_all.loc[:, '1_ESM2_mean0':]
train_data_label = train_data_all[['Label']]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


del feature_all
del train_data_all


print('Data processing completed')


### Optuna Logic Preset ===========================================================
print('Setting up model framework')
import time
import json
import torch
import optuna
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
import logging

# Initialize logger
logger = setup_logger('AMP_model_Training', outputPath + "/training.log")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_model(trial):
    all_5_val_loss_list = []
    all_5_mcc_list = []
    all_5_recall_list = []
    all_5_f1_list = []
    all_5_accuracy_list = []
    all_5_auc_list = []
    
    
    # Get hyperparameters
    lr = trial.suggest_loguniform('lr', 1e-6, 1e-5)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-3, 1e-1)
    hidden1_dim = trial.suggest_int('hidden1_dim', low=960, high=1280, step=320)
    hidden2_dim = trial.suggest_int('hidden2_dim', low=320, high=640, step=160)
    hidden3_dim = trial.suggest_int('hidden3_dim', low=40, high=160, step=60)
    

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_count = 1  # Initialize fold counter
    step_loss_records = {'fold_1_step': [], 'fold_1_step_loss': [],
                         'fold_2_step': [], 'fold_2_step_loss': [],
                         'fold_3_step': [], 'fold_3_step_loss': [],
                         'fold_4_step': [], 'fold_4_step_loss': [],
                         'fold_5_step': [], 'fold_5_step_loss': [],}

    for train_index, val_index in skf.split(train_data_features, train_data_label['Label']):
        logging.info(f"Starting Fold {fold_count}/{skf.n_splits}")  # Log current fold
        avg_val_loss_list = []
        avg_mcc_list = []
        accuracy_list = []
        recall_list = []
        f1_list = []
        auc_list = []

        model = DNN(input_dim=2560, hidden1_dim=hidden1_dim, hidden2_dim=hidden2_dim, hidden3_dim=hidden3_dim, output_dim=1).float().to(device)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs")
            model = nn.DataParallel(model)
        model.to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True, min_lr=1e-8)
        
        logging.info(f"Starting Fold {fold_count}/{skf.n_splits}")

        X_train, X_val = train_data_features.iloc[train_index, :], train_data_features.iloc[val_index, :]
        y_train, y_val = train_data_label['Label'].iloc[train_index], train_data_label['Label'].iloc[val_index]
        
        y_train = y_train.values.reshape(-1, 1)
        y_val = y_val.values.reshape(-1, 1)
        
        train_dataset = CustomDataset3(X_train, y_train)
        val_dataset = CustomDataset3(X_val, y_val)

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

        num_epochs = 35  # Total epochs
        step = 0
        step_loss = 0
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0

            for batch_idx, (features, labels) in enumerate(train_dataloader):
                features, labels = features.to(device), labels.float().to(device)
                optimizer.zero_grad()
                outputs = model(features)
                train_loss = criterion(outputs, ensure_2d_labels(labels))
                step += 1
                step_loss = train_loss.item()
                step_loss_records[f'fold_{fold_count}_step'].append(step)
                step_loss_records[f'fold_{fold_count}_step_loss'].append(step_loss)
                
                epoch_loss += train_loss.item()
                train_loss.backward()
                optimizer.step()

            avg_train_loss = epoch_loss / len(train_dataloader)
            logging.info(f"Epoch {epoch + 1}: Average Train Loss = {avg_train_loss:.4f}")

            
            avg_val_loss, accuracy, recall, f1, auc, mcc, precision, predictions_df = evaluate_model(model, val_dataloader, criterion, device)
            # Store performance metrics for current epoch
            avg_val_loss_list.append(avg_val_loss)
            avg_mcc_list.append(mcc)
            accuracy_list.append(accuracy)
            recall_list.append(recall)
            f1_list.append(f1)
            auc_list.append(auc)
            scheduler.step(avg_val_loss)
            logging.info(f"Epoch {epoch + 1} completed: Validation average Loss = {avg_val_loss:.4f}, MCC = {mcc:.4f}, Accuracy = {accuracy:.4f}, Recall = {recall:.4f}, F1 Score = {f1:.4f}, AUC = {auc:.4f}")


        # Aggregate performance metrics across 5 folds
        all_5_val_loss_list.append(avg_val_loss_list)
        all_5_mcc_list.append(avg_mcc_list)
        all_5_accuracy_list.append(accuracy_list)
        all_5_recall_list.append(recall_list)
        all_5_f1_list.append(f1_list)
        all_5_auc_list.append(auc_list)
        
        logging.info(f"Fold {fold_count} completed with validation MCC: {calculate_cv_loss(avg_mcc_list):.4f}, Accuracy = {calculate_cv_loss(accuracy_list):.4f}, Recall = {calculate_cv_loss(recall_list):.4f}, F1 Score = {calculate_cv_loss(f1_list):.4f}, AUC = {calculate_cv_loss(auc_list):.4f}")
        trial.set_user_attr(f'fold_{fold_count}_step_loss_records', step_loss_records)
        fold_count += 1  # Increment fold counter

    trial.set_user_attr('best_model_state_dict', model.state_dict())
    logging.info("Training completed.")

    avg_5_val_loss = calculate_cv_loss(all_5_val_loss_list)
    avg_5_mcc = calculate_cv_loss(all_5_mcc_list)
    
    # Store mcc_list in trial's user attributes
    trial.set_user_attr('mcc_list', all_5_mcc_list)
    trial.set_user_attr('accuracy_list', all_5_accuracy_list)
    trial.set_user_attr('recall_list', all_5_recall_list)
    trial.set_user_attr('f1_list', all_5_f1_list)
    trial.set_user_attr('auc_list', all_5_auc_list)
    
    return avg_5_mcc

# Other necessary imports and settings
logging.basicConfig(level=logging.INFO)


### Hyperparameter Tuning and Saving ===========================================================
st = time.time()
study = optuna.create_study(study_name='test_DNN', direction='minimize')
study.optimize(lambda trial: train_model(trial), n_trials= 6 )
# Get optimal hyperparameters
best_trial = study.best_trial

# Get all metric lists from best trial
best_mcc_list = best_trial.user_attrs.get('mcc_list', None)
best_accuracy_list = best_trial.user_attrs.get('accuracy_list', None)
best_recall_list = best_trial.user_attrs.get('recall_list', None)
best_f1_list = best_trial.user_attrs.get('f1_list', None)
best_auc_list = best_trial.user_attrs.get('auc_list', None)


file_name = 'model'
# Create dictionary to store best hyperparameters
best_params_dict = {
    'lr': best_trial.params['lr'],
    'weight_decay': best_trial.params['weight_decay'],
    'hidden1_dim': best_trial.params['hidden1_dim'],
    'hidden2_dim': best_trial.params['hidden2_dim'],
    'hidden3_dim': best_trial.params['hidden3_dim']
}

# Save best hyperparameters as JSON file
with open(outputPath + '/' + file_name + '_best_params.json', 'w') as f:
    json.dump(best_params_dict, f)

print("Best hyperparameters found: ", study.best_params)
print("Minimum validation loss: ", study.best_value)
print("Total optimization time: ", time.time() - st)
print("Best Loss:", study.best_value)

# Save best model evaluation results
best_mcc_list_df = pd.DataFrame(best_mcc_list)
best_mcc_list_df.to_csv(outputPath + '/' + file_name + 'best_mcc_list.csv', index=False)

best_accuracy_list_df = pd.DataFrame(best_accuracy_list)
best_accuracy_list_df.to_csv(outputPath + '/' + file_name + 'best_accuracy_list.csv', index=False)

best_recall_list_df = pd.DataFrame(best_recall_list)
best_recall_list_df.to_csv(outputPath + '/' + file_name + 'best_recall_list.csv', index=False)

best_f1_list_df = pd.DataFrame(best_f1_list)
best_f1_list_df.to_csv(outputPath + '/' + file_name + 'best_f1_list.csv', index=False)

best_auc_list_df = pd.DataFrame(best_auc_list)
best_auc_list_df.to_csv(outputPath + '/' + file_name + 'best_auc_list.csv', index=False)

# Save best model weights
torch.save(best_trial.user_attrs['best_model_state_dict'], outputPath + '/' + file_name + '_model_weights.pth')
save_fold_step_loss_as_csv(best_trial = best_trial, outputPath = outputPath, file_name = 'best_trail')

print("Job completed successfully!")

