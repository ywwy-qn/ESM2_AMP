# from torch.nn import init
import numpy as np
import pandas as pd
import re
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import spectral_norm
from sklearn.model_selection import KFold
from torch.optim import AdamW
import torch.nn.init as init
import optuna
import json
from torch.utils.data import TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging

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

log_filename = 'AutoEncoder_pretaining.log'
full_log_path = os.path.join(save_path, log_filename)
logger = setup_logger('AutoEncoder_Training', full_log_path)
logger.info('beginning')
feature_all = np.load(open_path + '/pan_data_protein_reshaped/pan_data_protein_reshaped/pan_data_protein_cls_segment0-9_eos_reshaped.npy')

flattened_data = feature_all.reshape(-1, 1280)
flattened_data_df = pd.DataFrame(flattened_data)
flattened_data = torch.tensor(flattened_data_df.values, dtype=torch.float32)

class Swish(nn.Module):
    def __init__(self, trainable_beta=False, initial_beta=1.0):
        super(Swish, self).__init__()
        if trainable_beta:
            self.beta = nn.Parameter(torch.tensor(initial_beta))
        else:
            self.beta = initial_beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


swish = Swish(trainable_beta=True)
class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResBlock, self).__init__()
        self.fc = spectral_norm(nn.Linear(in_dim, out_dim), n_power_iterations=3)
        self.bn = nn.BatchNorm1d(out_dim)
        self.activation = swish
        self.downsample = None
        if in_dim != out_dim:
            self.downsample = spectral_norm(nn.Linear(in_dim, out_dim), n_power_iterations=3)
        init.kaiming_normal_(self.fc.weight, nonlinearity='leaky_relu')
        if self.downsample is not None:
            init.kaiming_normal_(self.downsample.weight, nonlinearity='leaky_relu')

    def forward(self, x):
        residual = x
        out = self.activation(self.bn(self.fc(x)))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return out


class AutoEncoder(nn.Module):
    def __init__(self, input_dim=1280, latent_dim=150):
        super(AutoEncoder, self).__init__()

        encoder_layer_dims = [input_dim, 512, 256]
        self.fc_latent = spectral_norm(nn.Linear(encoder_layer_dims[-1], latent_dim), n_power_iterations=5)
        
        self.encoder = nn.ModuleList()
        for i in range(len(encoder_layer_dims) - 1):
            self.encoder.append(ResBlock(encoder_layer_dims[i], encoder_layer_dims[i + 1]))

        decoder_layer_dims = [latent_dim, 256, 512]
        
        self.decoder = nn.ModuleList()
        for i in range(len(decoder_layer_dims) - 1):
            self.decoder.append(ResBlock(decoder_layer_dims[i], decoder_layer_dims[i + 1]))

        self.fc3 = spectral_norm(nn.Linear(decoder_layer_dims[-1], input_dim), n_power_iterations=5)

    def encode(self, x):
        for block in self.encoder:
            x = block(x)
        latent_representation = self.fc_latent(x)
        return latent_representation

    def decode(self, z):
        for block in self.decoder:
            z = block(z)
        return self.fc3(z)  # Reconstructed input

    def forward(self, x):
        latent_representation = self.encode(x.view(-1, x.shape[1]))
        reconstructed = self.decode(latent_representation)
        return latent_representation, reconstructed
    

def loss_function(recon_x, x):
    recon_loss = nn.MSELoss(reduction='mean')(recon_x, x)
    return recon_loss


def train_AutoEncoder(trial, flattened_data, n_splits=5):
    latent_dim = 150
    input_dim = 1280
    logger.info(f"Starting trial {trial.number}")
    

    lr = trial.suggest_loguniform('lr', 1e-6, 1e-3)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flattened_tensor = torch.tensor(flattened_data, dtype=torch.float32)
    dataset = TensorDataset(flattened_tensor)
    
    total_loss = 0
    best_model_weights = None
    
    min_valid_loss = float('inf')
    for fold, (train_idx, valid_idx) in enumerate(kf.split(flattened_data)):
        train_data = DataLoader(dataset, batch_size=128, sampler=torch.utils.data.SubsetRandomSampler(train_idx))
        valid_data = DataLoader(dataset, batch_size=128, sampler=torch.utils.data.SubsetRandomSampler(valid_idx))

        model = AutoEncoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True, min_lr=1e-8)

        for epoch in range(100):
            model.train()
            running_loss = 0.0
            for batch in train_data:
                x = batch[0].to(device)
                optimizer.zero_grad()
                latent_representation, reconstructed = model(x)
                recon_loss = loss_function(reconstructed, x)
                # recon_loss, kl_loss, loss = loss_function(reconstructed, x, mu, log_var)
                recon_loss.backward()
                optimizer.step()
                running_loss += recon_loss.item()

            model.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for batch in valid_data:
                    x = batch[0].to(device)
                    latent_representation, reconstructed = model(x)
                    loss = loss_function(reconstructed, x)
                    # _, _, loss = loss_function(reconstructed, x, mu, log_var)
                    valid_loss += loss.item()

            logger.info(f"Fold {fold+1}/{n_splits}, Epoch {epoch+1}, Train Loss: {running_loss/len(train_data)}, Valid Loss: {valid_loss/len(valid_data)}")

            if valid_loss < min_valid_loss:
                min_valid_loss = valid_loss
                best_model_weights = model.state_dict()

        total_loss += min_valid_loss / len(valid_data)
    
    trial.set_user_attr('model_weights', best_model_weights)

    return total_loss / n_splits


print('Optuna')
def objective(trial):
    return train_AutoEncoder(trial, flattened_data)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)

print('saving')
best_model_weights = study.best_trial.user_attrs['model_weights']
torch.save(best_model_weights, save_path + '/AutoEncoder_model.pth')
logger.info("save to AutoEncoder_model.pth")