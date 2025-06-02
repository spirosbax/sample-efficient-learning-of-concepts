"""
Based on the Concept-Bottleneck models [Koh et al., 2020]

We will use the same encoder from the VAE that is trained in CITRIS. The decoder will just be an single
linear layer
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
import os

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from shared_utils.concept_models.shared import Encoder, TabularEncoder

class SimpleCBM(pl.LightningModule):
    """Concept Bottleneck Model"""

    def __init__(
        self,
        c_hid, 
        num_concepts, 
        num_output, 
        regression=True,
        conv=True, 
        img_width=64, 
        c_in=3,
        lr=0.001, 
        scheduler_step=1000

    ):
        super().__init__()
        self.save_hyperparameters()
        if regression:
            self.concept_encoder = nn.Sequential(
                Encoder(
                    num_latents=num_concepts, 
                    c_hid=c_hid,
                    c_in=c_in,
                    width=img_width,
                    variational=False
                ),
                nn.Sigmoid()
            )
        else:
            if conv:
                self.concept_encoder = nn.Sequential(
                    Encoder(
                        num_latents=num_concepts, 
                        c_hid=c_hid,
                        c_in=c_in,
                        width=img_width,
                        variational=False
                    ),
                    nn.Sigmoid()
                )
            else:
                self.concept_encoder = nn.Sequential(
                    TabularEncoder(
                        ni=c_in,
                        no=num_concepts,
                        nhidden=512, 
                        nlayers=6
                    ), 
                    nn.Sigmoid()
                )

        self.label_predictor = nn.Sequential(
            torch.nn.Linear(num_concepts, num_output)
        )
        self.regression = regression

    def forward(self, x):
        c_hat = self.concept_encoder(x)
        label_pred = self.label_predictor(c_hat)
        return label_pred

    def configure_optimizers(self):
        optimizer = optim.Adam([{"params": self.parameters(), "lr": self.hparams.lr}])
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=self.hparams.scheduler_step, 
            gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": lr_scheduler, "interval": "epoch", "frequency":1}}

    def lr_scheduler_step(self, scheduler, optimizer_idx=None, metric=None):
        scheduler.step()
    
    def _get_concept_loss(self, x, c, mode='train'):
        c_hat = self.concept_encoder(x)
        if self.regression:
            loss = F.mse_loss(c_hat, c)
        else:
            loss = F.binary_cross_entropy(c_hat, c)
        return loss, c_hat

    def _get_label_loss(self, c_hat, y, mode='train'):
        y_hat = self.label_predictor(c_hat)
        if self.regression:
            loss = F.mse_loss(y_hat, y)
        else:
            loss = F.binary_cross_entropy_with_logits(y_hat, y)
        return loss

    def training_step(self, batch, batch_idx):
        x, c, y = batch
        concept_loss, c_hat = self._get_concept_loss(x, c)
        label_loss = self._get_label_loss(c_hat, y)
        return concept_loss + 0.5 * label_loss

    def test_step(self, batch, batch_idx):
        x, c, y = batch

        concept_loss, c_hat = self._get_concept_loss(x, c)
        label_loss = self._get_label_loss(c_hat, y)
        return concept_loss + 0.5 * label_loss

    def validation_step(self, batch, batch_idx):
        x, c, y = batch
        concept_loss, c_hat = self._get_concept_loss(x, c)
        label_loss = self._get_label_loss(c_hat, y)
        return concept_loss + 0.5 * label_loss
    
