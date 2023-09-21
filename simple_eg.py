import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from torch import optim, nn, utils, Tensor
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import pytorch_lightning as pl


# define the model
class MyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(32 * 32 * 3, 64), 
            nn.ReLU(), 
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
            )
        
        self.decoder = nn.Sequential(
            nn.Linear(64, 64), 
            nn.ReLU(),     
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32 * 32 * 3))
    
    def forward(self, x: Tensor) -> Tensor:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


class LightingModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = MyModel()

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        x_hat = self.model(x)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

lighting_model = LightingModel()
train_loader = utils.data.DataLoader(CIFAR10('data/', download=True, transform=ToTensor()), batch_size=320)
trainer = pl.Trainer(
    accelerator='gpu', 
    devices=1, 
    strategy='ddp', 
    max_steps=100, 
    )
trainer.fit(model=lighting_model, train_dataloaders=train_loader)