import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_

#encoder_layer实例化
encoder_layer = nn.TransformerEncoderLayer(d_model=1280, nhead=8,norm_first=True, batch_first=True, dropout=0.3)
# 定义MLP类
class AMP_model(nn.Module):
    def __init__(self, input_dim=1280, hidden1_dim=512, hidden2_dim=128, output_dim=1, encoder_type='transformer'):
        super(AMP_model, self).__init__()
        
        self.encoder_type = encoder_type.lower()
        
        if self.encoder_type == 'transformer':
            # Transformer编码器
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        elif self.encoder_type == 'gru':
            # GRU编码器
            self.gru = nn.GRU(
                input_size=input_dim, 
                hidden_size=input_dim, 
                num_layers=6, 
                batch_first=True, 
                bidirectional=False
            )
        elif self.encoder_type != 'mean':
            raise ValueError("Unsupported encoder type. Choose from 'transformer', 'gru', or 'mean'.")
        
        # 定义MLP层
        self.mlp_layers = nn.Sequential(
            nn.Linear(input_dim, hidden1_dim),
            nn.ReLU(),
            nn.Linear(hidden1_dim, hidden2_dim),
            nn.ReLU(),
            nn.Linear(hidden2_dim, output_dim)
        )
        
        # 应用Kaiming初始化到所有线性层
        for m in self.mlp_layers.modules():
            if isinstance(m, nn.Linear):
                kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
        
        # 对GRU层进行初始化（可选）
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
            # 直接全局平均池化
            pooled = torch.mean(X, dim=1)
        else:
            # 通过编码器
            if self.encoder_type == 'transformer':
                encoded = self.transformer_encoder(X)  # (batch, seq, d_model)
            elif self.encoder_type == 'gru':
                encoded, _ = self.gru(X)  # (batch, seq, d_model)
            # 序列池化
            pooled = torch.mean(encoded, dim=1)  # (batch, d_model)
            
        # MLP前向传播
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
