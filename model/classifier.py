import torch.nn as nn
from torch import Tensor
from torchvision.models import resnet18, resnet34, resnet50


res_dict = {
    'res18': resnet18,
    'res34': resnet34,
    'res50': resnet50,
}


class ResModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = res_dict[cfg.model_type](pretrained=True)
        self.encoder.fc = nn.Linear(512, cfg.num_classes)

    def forward(self, x: Tensor) -> Tensor:
        r'''
        Input:
            x: [B, 3, H, W]
        Output:
            logits: [B, num_classes]
        '''
        return self.encoder(x)
    

class MyModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        D = cfg.hid_dim
        blocks = [nn.Sequential(
            nn.Conv2d(3, D, 3, 1, 1),
            nn.BatchNorm2d(D),
            nn.ReLU(),
        )]
        for i in range(cfg.num_blocks):
            blocks.append(nn.Sequential(
                nn.Conv2d(D, D*2, 3, 2, 1),
                nn.BatchNorm2d(D*2),
                nn.ReLU(),
            ))
            D *= 2
        blocks.append(nn.AdaptiveAvgPool2d(1))
        blocks.append(nn.Flatten())
        blocks.append(nn.Linear(D, cfg.num_classes))
        self.encoder = nn.Sequential(*blocks)
    
    def forward(self, x: Tensor) -> Tensor:
        r'''
        Input:
            x: [B, 3, H, W]
        Output:
            logits: [B, num_classes]
        '''
        return self.encoder(x)

       