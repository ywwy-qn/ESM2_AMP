
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.init import kaiming_normal_
from torch.optim import AdamW
import optuna
import time
from sklearn.metrics import roc_auc_score, matthews_corrcoef, accuracy_score, recall_score, f1_score, precision_score
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
import torch.nn.init as init
from torch.nn.utils import clip_grad_norm


# Construct CustomDataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_features, data_label):

        self.features = torch.tensor(data_features.values).float()

        assert len(data_features) == len(data_label), "The sample number of features and label data does not match!"

        self.labels = torch.tensor(data_label).float()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# Construct DNN class
class DNN(nn.Module):
    def __init__(self, input_dim=2560, hidden1_dim=1280, hidden2_dim=480, hidden3_dim=100, output_dim=1):
        super(DNN, self).__init__()

        self.dnn_layers = nn.Sequential(
            nn.Linear(input_dim, hidden1_dim),
            nn.LayerNorm(hidden1_dim),
            nn.ReLU(),
            nn.Linear(hidden1_dim, hidden2_dim),
            nn.LayerNorm(hidden2_dim),
            nn.ReLU(),
            nn.Linear(hidden2_dim, hidden3_dim),
            nn.LayerNorm(hidden3_dim),
            nn.ReLU(),
            nn.Linear(hidden3_dim, output_dim)
        )

        for m in self.dnn_layers.modules():
            if isinstance(m, nn.Linear):
                kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')

    def forward(self, X):
        dnn_out = self.dnn_layers(X)
        return dnn_out


# Define functions that evaluate the performance of the model on given dataset
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    y_pred_list = []
    y_true_list = []
    all_outputs = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.float().to(device)
            outputs = model(features)
            all_outputs.append(outputs)
            all_labels.append(labels)
            loss = criterion(outputs, labels)
            total_loss += loss.item()            
            predictions = outputs.sigmoid().cpu().numpy() > 0.5
            y_pred_list.extend(predictions)
            y_true_list.extend(labels.cpu().numpy())
            


    all_outputs2 = torch.cat(all_outputs, dim=0)
    all_labels2 = torch.cat(all_labels, dim=0)
    auc = roc_auc_score(all_labels2.cpu().numpy(), all_outputs2.cpu().numpy())

    mcc = matthews_corrcoef(y_true_list, y_pred_list)
    accuracy = accuracy_score(y_true_list, y_pred_list)
    recall = recall_score(y_true_list, y_pred_list, pos_label=1)
    f1 = f1_score(y_true_list, y_pred_list, pos_label=1)
    precision = precision_score(y_true_list, y_pred_list, pos_label=1)
    avg_loss = total_loss / len(dataloader)

    return avg_loss, accuracy, recall, f1, auc, mcc, precision




model = DNN().float().to(device)


# Load the trained model weights
checkpoint = torch.load("ESM2_DPM_checkpoint.pth")
model.load_state_dict(checkpoint['model_state_dict'])


X_test = test_data_features
y_test = test_data_label['Label']
        
y_test = y_test.values.reshape(-1, 1)

        
test_dataset = CustomDataset(X_test, y_test)


test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)



# Evaluate the model on the test dataset
criterion = nn.BCEWithLogitsLoss()
avg_loss, accuracy, recall, f1, auc, mcc, precision = evaluate_model(model, test_dataloader, criterion, device)

print(f"avg_loss: {avg_loss:.4f}")
print(f"accuracy: {accuracy:.4f}")
print(f"recall: {recall:.4f}")
print(f"f1 score: {f1:.4f}")
print(f"AUC: {auc:.4f}")
print(f"MCC: {mcc:.4f}")
print(f"Precision: {precision:.4f}")

# Generate predictions on test dataset
model.eval()
predictions = []
probabilities = []

with torch.no_grad():
    for features, _ in test_dataloader:
        features = features.to(device)
        outputs = model(features)
        probs = outputs.sigmoid().cpu().numpy()
        preds = (probs > 0.5).astype(float)
        predictions.extend(preds)
        probabilities.extend(probs)



predictions_df = pd.DataFrame({
    "Label": test_data_label["Label"],
    "Prediction": [p[0] for p in predictions],
    "Probability": [p[0] for p in probabilities]
})
predictions_df.to_csv(os.path.join(save_path, 'test_predictions_ESM2_DPM.csv'), index=False)




