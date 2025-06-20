import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_

# encoder_layer instantiation
encoder_layer = nn.TransformerEncoderLayer(d_model=1280, nhead=8,norm_first=True, batch_first=True, dropout=0.3)
# MLP class
class AMP_model(nn.Module):
    def __init__(self, input_dim=1280, hidden1_dim=512, hidden2_dim=128, output_dim=1, encoder_type='transformer'):
        super(AMP_model, self).__init__()
        
        self.encoder_type = encoder_type.lower()
        
        if self.encoder_type == 'transformer':
            # Transformer encoder
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        elif self.encoder_type == 'gru':
            # GRU encoder
            self.gru = nn.GRU(
                input_size=input_dim, 
                hidden_size=input_dim, 
                num_layers=6, 
                batch_first=True, 
                bidirectional=False
            )
        elif self.encoder_type != 'mean':
            raise ValueError("Unsupported encoder type. Choose from 'transformer', 'gru', or 'mean'.")
        
        # MLP layers
        self.mlp_layers = nn.Sequential(
            nn.Linear(input_dim, hidden1_dim),
            nn.ReLU(),
            nn.Linear(hidden1_dim, hidden2_dim),
            nn.ReLU(),
            nn.Linear(hidden2_dim, output_dim)
        )
        
        # Apply Kaiming to initialize all linear layers
        for m in self.mlp_layers.modules():
            if isinstance(m, nn.Linear):
                kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
        
        # Initialize the GRU layer
        if self.encoder_type == 'gru':
            for name, param in self.gru.named_parameters():
                if 'weight_ih' in name:
                    kaiming_normal_(param.data, a=0, mode='fan_in', nonlinearity='relu')
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    nn.init.constant_(param.data, 0)
    
    def forward(self, X):
        if self.encoder_type == 'mean':
            # Direct global average pooling
            pooled = torch.mean(X, dim=1)
        else:
            if self.encoder_type == 'transformer':
                encoded = self.transformer_encoder(X)  # (batch, seq, d_model)
            elif self.encoder_type == 'gru':
                encoded, _ = self.gru(X)  # (batch, seq, d_model)
                
            pooled = torch.mean(encoded, dim=1)  # (batch, d_model)
            
        return self.mlp_layers(pooled)




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
