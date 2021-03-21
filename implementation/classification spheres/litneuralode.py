import torch.nn as nn
import pytorch_lightning as pl
from torchdyn.models import *

class LitNeuralODE(pl.LightningModule):
    def __init__(self, f:nn.Module, sensitivity="adjoint", solver="dopri5"):
        super().__init__()
        self.model = NeuralDE(f, sensitivity=sensitivity, solver=solver)
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch   
        self.model.nfe=0
        xS = self(x)
        loss = nn.CrossEntropyLoss()(xS, y)
        logs = {'loss', loss, 'nfe', self.model.nfe}
        self.log('nfe', self.model.nfe, prog_bar=True)
        return {'loss': loss}   
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-6)

    def train_dataloader(self):
        return trainloader