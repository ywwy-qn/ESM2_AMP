
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.init as init

class Swish(nn.Module):
    def __init__(self, trainable_beta=False, initial_beta=1.0):
        super(Swish, self).__init__()
        if trainable_beta:
            self.beta = nn.Parameter(torch.tensor(initial_beta))  # 可训练的β
        else:
            self.beta = initial_beta  # 固定的β

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
        return self.fc3(z)

    def forward(self, x):
        latent_representation = self.encode(x.view(-1, x.shape[1]))
        reconstructed = self.decode(latent_representation)
        return latent_representation, reconstructed
    

def loss_function(recon_x, x):
    recon_loss = nn.MSELoss(reduction='mean')(recon_x, x)
    return recon_loss