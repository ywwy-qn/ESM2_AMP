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


# 配置日志器
def setup_logger(name, log_file, level=logging.INFO):
    """Function setup as many loggers as you want"""

    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')

    handler = logging.FileHandler(log_file)  # 将日志写入文件
    handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()  # 同时将日志输出到控制台
    stream_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(stream_handler)

    return logger


def signal_save_checkpoint(model, optimizer, scheduler, filename):
    """
    保存模型的状态字典、优化器的状态字典和学习率调度器的状态字典。
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    torch.save(checkpoint, filename)


def ensure_2d_labels(labels: torch.Tensor) -> torch.Tensor:
    """
    将输入标签转换为形状为 [N, 1] 的二维张量
    
    参数:
    labels (torch.Tensor): 输入标签张量，形状为 [N] 或 [N, 1]
    
    返回:
    torch.Tensor: 形状为 [N, 1] 的二维张量
    
    异常:
    ValueError: 如果输入张量维度不为1或2，或二维张量第二维度不为1
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
            
            # 计算概率和预测结果
            probs = outputs.sigmoid().cpu().numpy().flatten()  # 展平为1D数组
            predictions = (probs > 0.5).astype(float)
            
            # 收集结果
            probabilities_list.extend(probs.tolist())     # 概率值
            y_pred_list.extend(predictions.tolist())      # 预测标签

    # 生成并保存预测结果DataFrame
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
    probabilities_list = []  # 新增：用于保存概率值

    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.float().to(device)
            outputs = model(features)
            all_outputs.append(outputs)
            all_labels.append(ensure_2d_labels(labels))
            loss = criterion(outputs, ensure_2d_labels(labels))
            total_loss += loss.item()
            
            # 计算概率和预测结果
            probs = outputs.sigmoid().cpu().numpy().flatten()  # 展平为1D数组
            predictions = (probs > 0.5).astype(float)
            
            # 收集结果
            probabilities_list.extend(probs.tolist())     # 概率值
            y_pred_list.extend(predictions.tolist())      # 预测标签
            y_true_batch = labels.cpu().numpy().flatten() # 真实标签展平
            y_true_list.extend(y_true_batch.tolist())     

    # 计算AUC
    all_outputs_tensor = torch.cat(all_outputs, dim=0)
    all_labels_tensor = torch.cat(all_labels, dim=0)
    auc = roc_auc_score(all_labels_tensor.cpu().numpy(), all_outputs_tensor.cpu().numpy())

    # 计算其他指标
    mcc = matthews_corrcoef(y_true_list, y_pred_list)
    accuracy = accuracy_score(y_true_list, y_pred_list)
    recall = recall_score(y_true_list, y_pred_list, pos_label=1)
    f1 = f1_score(y_true_list, y_pred_list, pos_label=1)
    precision = precision_score(y_true_list, y_pred_list, pos_label=1)
    avg_loss = total_loss / len(dataloader)

    # 生成并保存预测结果DataFrame
    predictions_df = pd.DataFrame({
        "Label": y_true_list,
        "Prediction": y_pred_list,
        "Probability": probabilities_list
    })


    return avg_loss, accuracy, recall, f1, auc, mcc, precision, predictions_df
        


class SaveMetricsAndBestCheckpoints:
    def __init__(self, file_name="val_metrics.xlsx", checkpoint_dir="best_checkpoints", top_k=10, metrics_name="MCC"):
        self.file_path = os.path.join(checkpoint_dir, file_name)  # Excel 文件路径
        self.checkpoint_dir = checkpoint_dir  # 检查点保存目录
        self.top_k = top_k  # 保留的检查点数量
        self.metrics_name = metrics_name  # 用于比较的指标名称
        self.metrics_data = []  # 用于存储每个 epoch 的指标数据
        self.best_checkpoints = []  # 用于存储表现最好的检查点信息
        os.makedirs(self.checkpoint_dir, exist_ok=True)  # 创建检查点保存目录

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

        # 计算 AUC
        all_outputs2 = torch.cat(all_outputs, dim=0)  # 拼接所有批次的输出
        all_labels2 = torch.cat(all_labels, dim=0)  # 拼接所有批次的标签
        try:
            auc_score = roc_auc_score(all_labels2.cpu().numpy(), all_outputs2.cpu().numpy())
        except ValueError:
            auc_score = 0.5  # 如果只有一个类存在，AUC无法计算，设为0.5

        # 计算 PR AUC
        precision, recall, _ = precision_recall_curve(all_labels2.cpu().numpy(), all_outputs2.cpu().numpy())
        pr_auc = auc(recall, precision)

        # 计算性能指标
        mcc = matthews_corrcoef(y_true_list, y_pred_list)
        accuracy = accuracy_score(y_true_list, y_pred_list)
        recall = recall_score(y_true_list, y_pred_list, pos_label=1)
        f1 = f1_score(y_true_list, y_pred_list, pos_label=1)
        precision_score_val = precision_score(y_true_list, y_pred_list, pos_label=1)

        # 计算特异性 (Specificity)
        tn, fp, fn, tp = confusion_matrix(y_true_list, y_pred_list).ravel()
        specificity = tn / (tn + fp)

        # 计算平衡准确率 (Balanced Accuracy)
        balanced_accuracy = (recall + specificity) / 2

        avg_val_loss = val_loss / len(val_dataloader)

        # 将指标数据添加到列表中
        metrics = {
            "Epoch": epoch_cord,
            "Val Loss": avg_val_loss,
            "AUC": auc_score,
            "PR AUC": pr_auc,  # 新增 PR AUC
            "MCC": mcc,
            "Accuracy": accuracy,
            "Recall": recall,
            "F1": f1,
            "Precision": precision_score_val,
            "Specificity": specificity,
            "Balanced Accuracy": balanced_accuracy
        }
        self.metrics_data.append(metrics)

        # 将数据保存到 Excel 文件
        df = pd.DataFrame(self.metrics_data)
        df.to_excel(self.file_path, index=False)

        # 获取当前 MCC 值
        current_mcc = metrics[self.metrics_name]

        # 更新表现最好的检查点列表
        self.best_checkpoints.append((current_mcc, epoch_cord))
        self.best_checkpoints.sort(reverse=True, key=lambda x: (x[0], x[1]))  # 按 MCC 降序，epoch 升序

        # 如果列表长度超过 top_k，移除最差的检查点
        if len(self.best_checkpoints) > self.top_k:
            removed_mcc, removed_epoch = self.best_checkpoints.pop()
            removed_checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{removed_epoch:.0f}.pth")
            if os.path.exists(removed_checkpoint_path):
                print(f"Removing checkpoint from epoch {removed_epoch} with MCC {removed_mcc:.4f}")
                os.remove(removed_checkpoint_path)
            else:
                print(f"Checkpoint file not found: {removed_checkpoint_path}")

        # 保存当前检查点
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch_cord:.0f}.pth")
        checkpoint = {
            'epoch': epoch_cord,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': avg_val_loss,
            'auc': auc_score,
            'pr_auc': pr_auc,  # 新增 PR AUC
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

# 使用示例
# saver = SaveMetricsAndBestCheckpoints(file_name="val_metrics.xlsx", checkpoint_dir="best_checkpoints", top_k=5, metrics_name="MCC")
# saver.save_checkpoint(model, optimizer, epoch, val_dataloader, criterion, device)



def calculate_cv_loss(all_5_val_loss_list, mode="last"):
    """
    计算五折交叉验证中最佳epoch或最后一个epoch的平均验证损失。

    参数:
    all_5_val_loss_list (list of list of float): 
        五折交叉验证的验证损失数据，每个子列表包含一折所有epoch的损失值。
    mode (str): 
        计算模式，可选值:
            - "best": 计算每折最佳epoch的平均损失
            - "last": 计算每折最后一个epoch的平均损失

    返回:
    float: 指定模式下的平均验证损失

    异常:
    ValueError: 如果 mode 参数不是 "best" 或 "last"
    """
    # 输入验证
    if mode not in ["best", "last"]:
        raise ValueError("mode 参数必须为 'best' 或 'last'")
    
    if not all(isinstance(fold, list) for fold in all_5_val_loss_list):
        raise TypeError("all_5_val_loss_list 必须是一个嵌套列表")
    
    # 根据模式提取损失值
    if mode == "best":
        selected_losses = [min(fold) for fold in all_5_val_loss_list]
    else:  # mode == "last"
        selected_losses = [fold[-1] for fold in all_5_val_loss_list]
    
    # 计算平均损失
    return np.mean(selected_losses)


def save_fold_step_loss_as_csv(best_trial, outputPath, file_name):
    # 从最佳试验的用户属性中获取所有折叠的训练步骤损失记录
    step_loss_records = {}
    for key in best_trial.user_attrs.keys():
        if 'fold_' in key and '_step_loss_records' in key:
            fold_number = int(key.split('_')[1])
            step_loss_records[fold_number] = best_trial.user_attrs[key]

    # 创建一个空的 DataFrame 来存储所有折叠的步进损失记录
    all_fold_step_loss = pd.DataFrame()

    # 遍历所有折叠的记录并将它们添加到 DataFrame 中
    for fold_number, record in sorted(step_loss_records.items()):
        fold_df = pd.DataFrame({
            f'fold_{fold_number}_step': record[f'fold_{fold_number}_step'],
            f'fold_{fold_number}_step_loss': record[f'fold_{fold_number}_step_loss']
        })
        if all_fold_step_loss.empty:
            all_fold_step_loss = fold_df
        else:
            all_fold_step_loss = pd.concat([all_fold_step_loss, fold_df], axis=1)

    # 保存为 CSV 文件
    output_file_path = outputPath + '/' + file_name + '_all_folds_model_train_step_loss.csv'
    all_fold_step_loss.to_csv(output_file_path, index=False)

    print(f'Step loss records for all folds saved to {output_file_path}')



def save_evaluation_metrics(outputPath, file_name, **metrics):
    """
    保存模型评估指标到CSV文件

    参数:
    outputPath (str): 输出目录路径
    file_name (str): 文件名前缀
    **metrics: 可变关键字参数，键为指标名称，值为对应的指标列表
    """
    import pandas as pd
    import os

    # 确保输出目录存在
    os.makedirs(outputPath, exist_ok=True)
    
    # 遍历所有传入的指标
    for metric, data in metrics.items():
        # 创建DataFrame
        df = pd.DataFrame(data)
        # 生成文件路径
        filename = f"{file_name}_best_{metric}_list.csv"
        filepath = os.path.join(outputPath, filename)
        # 保存为CSV
        df.to_csv(filepath, index=False)
    
    

def convert_step_loss_to_dataframe(step_loss_records: Dict[str, List[float]]) -> pd.DataFrame:
    """
    将包含多个折叠训练步骤和损失值的字典转换为结构化 DataFrame。

    Args:
        step_loss_records (Dict[str, List[float]]): 
            输入字典，格式示例：
                {
                    'fold_1_step': [0, 1, 2],
                    'fold_1_step_loss': [0.45, 0.32, 0.28],
                    'fold_2_step': [...],
                    ...
                }

    Returns:
        pd.DataFrame: 包含三列的 DataFrame：
            - fold: 折叠名称（如 'fold_1'）
            - step: 训练步骤编号
            - loss: 对应的损失值

    示例输出：
        |     fold | step |  loss |
        |----------|------|-------|
        |  fold_1  |  0   | 0.45  |
        |  fold_1  |  1   | 0.32  |
        |  ...     | ...  | ...   |
    """
    
    # 提取所有唯一的折叠名称（如 fold_1, fold_2 等）
    folds = set()
    for key in step_loss_records.keys():
        # 通过分割键名获取折叠前缀（例如 'fold_1_step' -> 'fold_1'）
        fold_prefix = '_'.join(key.split('_')[:2])
        folds.add(fold_prefix)
    folds = sorted(folds)  # 按折叠顺序排序

    # 构建结构化数据列表
    data = []
    for fold in folds:
        # 获取当前折叠的步骤和损失值列表
        steps = step_loss_records.get(f"{fold}_step", [])
        losses = step_loss_records.get(f"{fold}_step_loss", [])
        
        # 确保步骤和损失值长度一致
        if len(steps) != len(losses):
            raise ValueError(f"折叠 {fold} 的步骤和损失值长度不匹配")
        
        # 将数据添加到列表
        for step, loss in zip(steps, losses):
            data.append({"fold": fold, "step": step, "loss": loss})

    # 转换为 DataFrame
    df = pd.DataFrame(data)
    return df