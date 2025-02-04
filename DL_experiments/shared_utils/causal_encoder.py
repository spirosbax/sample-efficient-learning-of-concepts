import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
import numpy as np
from collections import OrderedDict, defaultdict
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../..'))


class MLP(pl.LightningModule):
    """ Very simple MLP that will be trained to predict 1 of the causal factors from the learned encodings"""

    def __init__(self, c_in, c_hid, c_out, lr):

        super().__init__()
        self.save_hyperparameters()

        # Base Network
        self.encoder = nn.Sequential(
            nn.Linear(self.hparams.c_in, self.hparams.c_hid),
            nn.Tanh(),
            nn.Linear(self.hparams.c_hid, self.hparams.c_hid),
            nn.Tanh(), 
            nn.Linear(self.hparams.c_hid, self.hparams.c_out)
        )

    def forward(self, z):
        return self.encoder(z)

    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        return loss
