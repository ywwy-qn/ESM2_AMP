import os
import torch
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Any
from collections import OrderedDict
from sklearn.metrics import (
    roc_auc_score, matthews_corrcoef, accuracy_score, recall_score, 
    f1_score, precision_score, confusion_matrix, precision_recall_curve, auc
)


# Configure the loger
def setup_logger(name, log_file, level=logging.INFO):
    """Function setup as many loggers as you want"""

    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(stream_handler)

    return logger


def signal_save_checkpoint(model, optimizer, scheduler, filename):
    """
    Save the state dictionary of the model, the state dictionary of the optimizer and the state dictionary of the learning rate scheduler.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    torch.save(checkpoint, filename)


def ensure_2d_labels(labels: torch.Tensor) -> torch.Tensor:
    """
    Convert the input label to a two-dimensional tensor of shape [N, 1]
    
    parameters:
    labels (torch.Tensor): Input the label tensor, with the shape of [N] or [N, 1]
    
    return:
    torch.Tensor: A two-dimensional tensor of the shape [N, 1]
    
    Exception:
    ValueError: If the dimension of the input tensor is not 1 or 2, or the second dimension of the two-dimensional tensor is not 1
    """
    if not isinstance(labels, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(labels)}")
    
    if labels.dim() == 1:
        return labels.unsqueeze(1)
    elif labels.dim() == 2:
        if labels.shape[1] == 1:
            return labels
        else:
            raise ValueError(f"Expected 2D labels with shape (N, 1), got {labels.shape}")
    else:
        raise ValueError(f"Labels must be 1D or 2D tensor, got {labels.dim()}D tensor") 


        
def fix_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove 'module.' of dataparallel
        new_state_dict[name] = v
    return new_state_dict
        

    
def model_info(model, dataset, batch_size, device):
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    y_pred_list = []
    all_outputs = []
    probabilities_list = []
    with torch.no_grad():
        for features in tqdm(dataloader, desc="Running model inference"):
            features = features.to(device)
            outputs = model(features)
            all_outputs.append(outputs)
            
            probs = outputs.sigmoid().cpu().numpy().flatten()
            predictions = (probs > 0.5).astype(float)
            
            probabilities_list.extend(probs.tolist())
            y_pred_list.extend(predictions.tolist())

    predictions_df = pd.DataFrame({
        "Prediction": y_pred_list,
        "Probability": probabilities_list
    })

    return predictions_df    
    
    
        
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    y_pred_list = []
    y_true_list = []
    all_outputs = []
    all_labels = []
    total_loss = 0.0
    probabilities_list = []

    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.float().to(device)
            outputs = model(features)
            all_outputs.append(outputs)
            all_labels.append(ensure_2d_labels(labels))
            loss = criterion(outputs, ensure_2d_labels(labels))
            total_loss += loss.item()
            
            probs = outputs.sigmoid().cpu().numpy().flatten()
            predictions = (probs > 0.5).astype(float)
            
            probabilities_list.extend(probs.tolist())
            y_pred_list.extend(predictions.tolist()) 
            y_true_batch = labels.cpu().numpy().flatten()
            y_true_list.extend(y_true_batch.tolist())     

    all_outputs_tensor = torch.cat(all_outputs, dim=0)
    all_labels_tensor = torch.cat(all_labels, dim=0)
    auc = roc_auc_score(all_labels_tensor.cpu().numpy(), all_outputs_tensor.cpu().numpy())

    mcc = matthews_corrcoef(y_true_list, y_pred_list)
    accuracy = accuracy_score(y_true_list, y_pred_list)
    recall = recall_score(y_true_list, y_pred_list, pos_label=1)
    f1 = f1_score(y_true_list, y_pred_list, pos_label=1)
    precision = precision_score(y_true_list, y_pred_list, pos_label=1)
    avg_loss = total_loss / len(dataloader)

    predictions_df = pd.DataFrame({
        "Label": y_true_list,
        "Prediction": y_pred_list,
        "Probability": probabilities_list
    })


    return avg_loss, accuracy, recall, f1, auc, mcc, precision, predictions_df
        


class SaveMetricsAndBestCheckpoints:
    def __init__(self, file_name="val_metrics.xlsx", checkpoint_dir="best_checkpoints", top_k=10, metrics_name="MCC"):
        self.file_path = os.path.join(checkpoint_dir, file_name)
        self.checkpoint_dir = checkpoint_dir
        self.top_k = top_k
        self.metrics_name = metrics_name
        self.metrics_data = []
        self.best_checkpoints = []
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, model, optimizer, epoch_cord, val_dataloader, criterion, device):
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            y_pred_list = []
            y_true_list = []
            all_outputs = []
            all_labels = []

            for features, labels in val_dataloader:
                features, labels = features.to(device), labels.float().to(device)
                outputs = model(features)
                all_outputs.append(outputs)
                all_labels.append(ensure_2d_labels(labels))
                loss = criterion(outputs, ensure_2d_labels(labels))
                val_loss += loss.item()
                predictions = outputs.sigmoid().cpu().numpy() > 0.5
                y_pred_list.extend(predictions)
                y_true_list.extend(labels.cpu().numpy())

        all_outputs2 = torch.cat(all_outputs, dim=0)
        all_labels2 = torch.cat(all_labels, dim=0)
        try:
            auc_score = roc_auc_score(all_labels2.cpu().numpy(), all_outputs2.cpu().numpy())
        except ValueError:
            auc_score = 0.5

        precision, recall, _ = precision_recall_curve(all_labels2.cpu().numpy(), all_outputs2.cpu().numpy())
        pr_auc = auc(recall, precision)

        mcc = matthews_corrcoef(y_true_list, y_pred_list)
        accuracy = accuracy_score(y_true_list, y_pred_list)
        recall = recall_score(y_true_list, y_pred_list, pos_label=1)
        f1 = f1_score(y_true_list, y_pred_list, pos_label=1)
        precision_score_val = precision_score(y_true_list, y_pred_list, pos_label=1)

        tn, fp, fn, tp = confusion_matrix(y_true_list, y_pred_list).ravel()
        specificity = tn / (tn + fp)

        balanced_accuracy = (recall + specificity) / 2

        avg_val_loss = val_loss / len(val_dataloader)

        metrics = {
            "Epoch": epoch_cord,
            "Val Loss": avg_val_loss,
            "AUC": auc_score,
            "PR AUC": pr_auc,
            "MCC": mcc,
            "Accuracy": accuracy,
            "Recall": recall,
            "F1": f1,
            "Precision": precision_score_val,
            "Specificity": specificity,
            "Balanced Accuracy": balanced_accuracy
        }
        self.metrics_data.append(metrics)

        df = pd.DataFrame(self.metrics_data)
        df.to_excel(self.file_path, index=False)

        current_mcc = metrics[self.metrics_name]

        self.best_checkpoints.append((current_mcc, epoch_cord))
        self.best_checkpoints.sort(reverse=True, key=lambda x: (x[0], x[1]))

        if len(self.best_checkpoints) > self.top_k:
            removed_mcc, removed_epoch = self.best_checkpoints.pop()
            removed_checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{removed_epoch:.0f}.pth")
            if os.path.exists(removed_checkpoint_path):
                print(f"Removing checkpoint from epoch {removed_epoch} with MCC {removed_mcc:.4f}")
                os.remove(removed_checkpoint_path)
            else:
                print(f"Checkpoint file not found: {removed_checkpoint_path}")

        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch_cord:.0f}.pth")
        checkpoint = {
            'epoch': epoch_cord,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': avg_val_loss,
            'auc': auc_score,
            'pr_auc': pr_auc,
            'mcc': mcc,
            'accuracy': accuracy,
            'recall': recall,
            'f1': f1,
            'precision': precision_score_val,
            'specificity': specificity,
            'balanced_accuracy': balanced_accuracy
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"{metrics}")

# saver = SaveMetricsAndBestCheckpoints(file_name="val_metrics.xlsx", checkpoint_dir="best_checkpoints", top_k=5, metrics_name="MCC")
# saver.save_checkpoint(model, optimizer, epoch, val_dataloader, criterion, device)



def calculate_cv_loss(all_5_val_loss_list, mode="last"):
    """
    Calculate the average validation loss of the best epoch or the last epoch in the five-fold cross-validation.

    parameters:
    all_5_val_loss_list (list of list of float): 
        The verification loss data of five-fold cross-validation, with each sublist containing one fold of the loss values for all epochs.
    mode (str): 
        Calculation mode, optional value:
            - "best": Calculate the average loss of the best epoch per fold
            - "last": Calculate the average loss of the last epoch of each fold

    return:
    float: The average verification loss in the specified mode

    Exception:
    ValueError: If the mode parameter is not "best" or "last"
    """
    if mode not in ["best", "last"]:
        raise ValueError("The mode parameter must be 'best' or 'last'.")
    
    if not all(isinstance(fold, list) for fold in all_5_val_loss_list):
        raise TypeError("all_5_val_loss_list must be a nested list")
    
    if mode == "best":
        selected_losses = [min(fold) for fold in all_5_val_loss_list]
    else:  # mode == "last"
        selected_losses = [fold[-1] for fold in all_5_val_loss_list]
    
    return np.mean(selected_losses)


def save_fold_step_loss_as_csv(best_trial, outputPath, file_name):
    step_loss_records = {}
    for key in best_trial.user_attrs.keys():
        if 'fold_' in key and '_step_loss_records' in key:
            fold_number = int(key.split('_')[1])
            step_loss_records[fold_number] = best_trial.user_attrs[key]

    all_fold_step_loss = pd.DataFrame()

    for fold_number, record in sorted(step_loss_records.items()):
        fold_df = pd.DataFrame({
            f'fold_{fold_number}_step': record[f'fold_{fold_number}_step'],
            f'fold_{fold_number}_step_loss': record[f'fold_{fold_number}_step_loss']
        })
        if all_fold_step_loss.empty:
            all_fold_step_loss = fold_df
        else:
            all_fold_step_loss = pd.concat([all_fold_step_loss, fold_df], axis=1)

    output_file_path = outputPath + '/' + file_name + '_all_folds_model_train_step_loss.csv'
    all_fold_step_loss.to_csv(output_file_path, index=False)

    print(f'Step loss records for all folds saved to {output_file_path}')



def save_evaluation_metrics(outputPath, file_name, **metrics):
    
    import pandas as pd
    import os

    os.makedirs(outputPath, exist_ok=True)
    
    for metric, data in metrics.items():
        df = pd.DataFrame(data)
        filename = f"{file_name}_best_{metric}_list.csv"
        filepath = os.path.join(outputPath, filename)
        df.to_csv(filepath, index=False)
    
    

def convert_step_loss_to_dataframe(step_loss_records: Dict[str, List[float]]) -> pd.DataFrame:
    """
   Convert a dictionary containing multiple folding training steps and loss values into a structured DataFrame.

    Args:
        step_loss_records (Dict[str, List[float]]): 
            Input dictionary. Format example:
                {
                    'fold_1_step': [0, 1, 2],
                    'fold_1_step_loss': [0.45, 0.32, 0.28],
                    'fold_2_step': [...],
                    ...
                }

    Return:
        pd.DataFrame: A DataFrame containing three columns:
        -fold: Folding name (such as 'fold_1')
        -step: Training step number
        -loss: The corresponding loss value

    output example：
        |     fold | step |  loss |
        |----------|------|-------|
        |  fold_1  |  0   | 0.45  |
        |  fold_1  |  1   | 0.32  |
        |  ...     | ...  | ...   |
    """
    
    folds = set()
    for key in step_loss_records.keys():
        fold_prefix = '_'.join(key.split('_')[:2])
        folds.add(fold_prefix)
    folds = sorted(folds) 

    data = []
    for fold in folds:
        steps = step_loss_records.get(f"{fold}_step", [])
        losses = step_loss_records.get(f"{fold}_step_loss", [])
        
        if len(steps) != len(losses):
            raise ValueError(f"The steps of folding {fold} do not match the length of the loss value")
        
        for step, loss in zip(steps, losses):
            data.append({"fold": fold, "step": step, "loss": loss})

    df = pd.DataFrame(data)
    return df
