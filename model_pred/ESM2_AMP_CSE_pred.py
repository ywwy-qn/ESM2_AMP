import os

import torch
import torch.nn as nn

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# import module
from AMPmodel.dataset import load_dataset
from AMPmodel.model import AMP_model
from AMPmodel.check import fix_state_dict, evaluate_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

feature_file = os.path.join(project_root, "model_pred/data", "real_test_dataset_features.h5")
sample_file = os.path.join(project_root, "model_pred/data", "real_test_dataset_samples.xlsx")

test_dataset = load_dataset(
    mode='split',
    feature_file=feature_file,
    sample_file=sample_file
)


test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# loading model
model = AMP_model(input_dim=1280, hidden1_dim=480, hidden2_dim=240, output_dim=1, encoder_type="transformer").float().to(device)
checkpoint = torch.load("model_pred/weights_file/ESM2_AMP_CSE.pth")

if 'module' in list(checkpoint['model_state_dict'].keys())[0]:
    checkpoint['model_state_dict'] = fix_state_dict(checkpoint['model_state_dict'])

model.load_state_dict(checkpoint['model_state_dict'])

# evaluation model
criterion = nn.BCEWithLogitsLoss()
avg_loss, accuracy, recall, f1, auc, mcc, precision, predictions_df = evaluate_model(model, test_dataloader, criterion, device)

print(f"accuracy: {accuracy:.4f}")
print(f"recall: {recall:.4f}")
print(f"f1 score: {f1:.4f}")
print(f"AUC: {auc:.4f}")
print(f"MCC: {mcc:.4f}")
print(f"Precision: {precision:.4f}")

predictions_df.to_csv(os.path.join('test_predictions_ESM2_AMP_CSE.csv'), index=False)
