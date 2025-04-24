import os
import sys
import torch
import torch.nn as nn
project_path = os.path.dirname(os.getcwd())
sys.path.append(os.path.join(project_path, 'Model_work'))
from AMPmodel.dataset import CustomDataset
from AMPmodel.model import AMP_model
from AMPmodel.check import fix_state_dict, evaluate_model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


X_test = test_data_features
y_test = test_data_label['Label'].values.reshape(-1, 1)
del test_data_features
test_dataset = CustomDataset(X_test, y_test, reshape_shape=(-1, 20, 1280))
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
del X_test, y_test, test_dataset


# Load the trained model weights
model = AMP_model(input_dim=1280, hidden1_dim=640, hidden2_dim=240, output_dim=1, encoder_type="transformer").float().to(device)
checkpoint = torch.load("ESM2_AMPS_checkpoint.pth")
if 'module' in list(checkpoint['model_state_dict'].keys())[0]:
    checkpoint['model_state_dict'] = fix_state_dict(checkpoint['model_state_dict'])
model.load_state_dict(checkpoint['model_state_dict'])



# Evaluate the model on the test dataset
criterion = nn.BCEWithLogitsLoss()
avg_loss, accuracy, recall, f1, auc, mcc, precision, predictions_df = evaluate_model(model, test_dataloader, criterion, device)
print(f"avg_loss: {avg_loss:.4f}")
print(f"accuracy: {accuracy:.4f}")
print(f"recall: {recall:.4f}")
print(f"f1 score: {f1:.4f}")
print(f"AUC: {auc:.4f}")
print(f"MCC: {mcc:.4f}")
print(f"Precision: {precision:.4f}")
predictions_df.to_csv(os.path.join('test_predictions_ESM2_AMPS.csv'), index=False)