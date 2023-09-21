import torch.nn as nn
from torch import Tensor


class MyModel(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, cfg.hid_dim, 3, 1, 1), 
            nn.ReLU(), 
            nn.Conv2d(cfg.hid_dim, cfg.hid_dim, 3, 1, 1), 
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(cfg.hid_dim, cfg.hid_dim, 3, 1, 1), 
            nn.ReLU(),     
            nn.Conv2d(cfg.hid_dim, 3, 3, 1, 1), 
        )
    
    def forward(self, x: Tensor) -> Tensor:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat