import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt

from .config_torchwavenet import ModelConfig, SAEConfig
from .torchwavenet import Wave


class WaveQPSK(Wave):
    sigmoid = True
    hard_dec = False
    encoder_scaling = True
    decoder_scaling = True
    def __init__(self, cfg: ModelConfig, sae: SAEConfig):
        super().__init__(cfg)
        self.sigmoid = sae.sigmoid
        self.hard_dec = sae.hard_dec
        self.encoder_scaling = sae.encoder_scaling
        self.decoder_scaling = sae.decoder_scaling
        fsz = 5
        rc = self.cfg.residual_channels
        self.encoder = nn.Sequential(
            nn.Conv1d(rc, rc, fsz, stride=2, padding=fsz//2),
            nn.Conv1d(rc, rc, fsz, stride=2, padding=fsz//2),
            nn.Conv1d(rc, rc, fsz, stride=2, padding=fsz//2),
            nn.Conv1d(rc, rc, fsz, stride=2, padding=fsz//2),
            nn.Conv1d(rc, 2, 1)
        )
        self.scale = nn.Conv1d(1, 1, 1, bias=True)
        self.decoder = nn.Sequential(
            nn.Conv1d(1, 2, 2, stride=2, bias=True),
            nn.ConvTranspose1d(2, 2, 129, stride=2**4, bias=False)
        )

    def forward(self, input, return_bits=True):
        x = input
        x = self.input_projection(x)
        x = F.relu(x)

        skip = None
        for layer in self.residual_layers:
            x, skip_connection = layer(x)
            skip = skip_connection if skip is None else skip_connection + skip

        x = skip / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        y = self.output_projection(x)
        # autoencoder
        pw = y.square().sum(dim=1, keepdim=True).mean(dim=2, keepdim=True).sqrt() #TODO pw = torch.sqrt(pw+1e-8)
        b = self.encoder(x)
        b = torch.flatten(b.transpose(1,2), start_dim=1).unsqueeze(1)
        if self.encoder_scaling:
            b = b / pw
        b = self.scale(b)
        if self.hard_dec:
            x = (b >= 0).to(torch.float32) if self.sigmoid else (b >= 0.5).to(torch.float32)
        else:
            x = torch.sigmoid(b) if self.sigmoid else b
        x = self.decoder(x)
        if self.decoder_scaling: 
            x = x * pw
        start = (x.shape[-1] - input.shape[-1])//2
        x = x[..., start:start+input.shape[-1]]

        if return_bits:
            return x, b.squeeze(1)
        else:
            return x
        

class WaveOFDM(Wave):
    sigmoid = True
    hard_dec = False
    encoder_scaling = True
    decoder_scaling = True
    def __init__(self, cfg: ModelConfig, sae: SAEConfig):
        super().__init__(cfg)
        self.sigmoid = sae.sigmoid
        self.hard_dec = sae.hard_dec
        self.encoder_scaling = sae.encoder_scaling
        self.decoder_scaling = sae.decoder_scaling
        sym_in = 56
        sym_out = 80
        self.encoder = nn.Sequential(
            nn.Conv1d(cfg.residual_channels, sym_in*2, sym_out, stride=sym_out, padding=0, bias=False),
            nn.Conv1d(sym_in*2, sym_in*2, 1),
        )
        self.scale = nn.Conv1d(1, 1, 1, bias=True)
        self.decoder = nn.Sequential(
            nn.Conv1d(1, 2, 2, stride=2, bias=True),
            nn.Conv1d(2, sym_out*2, sym_in, stride=sym_in, bias=False)
        )

    def forward(self, input, return_bits=True):
        x = input
        x = self.input_projection(x)
        x = F.relu(x)

        skip = None
        for layer in self.residual_layers:
            x, skip_connection = layer(x)
            skip = skip_connection if skip is None else skip_connection + skip

        x = skip / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        y = self.output_projection(x)
        # autoencoder
        pw = y.square().sum(dim=1, keepdim=True).mean(dim=2, keepdim=True).sqrt()
        b = self.encoder(x)
        b = torch.flatten(b.transpose(1,2), start_dim=1).unsqueeze(1)
        if self.encoder_scaling:
            b = b / pw
        b = self.scale(b)
        if self.hard_dec:
            x = (b >= 0).to(torch.float32) if self.sigmoid else (b >= 0.5).to(torch.float32)
        else:
            x = torch.sigmoid(b) if self.sigmoid else b
        x = self.decoder(x)
        if self.decoder_scaling: 
            x = x * pw
        x = x.transpose(1,2).reshape((input.shape[0], -1, 2)).transpose(1,2)
        start = (x.shape[-1] - input.shape[-1])//2
        x = x[..., start:start+input.shape[-1]]

        if return_bits:
            return x, b.squeeze(1)
        else:
            return x