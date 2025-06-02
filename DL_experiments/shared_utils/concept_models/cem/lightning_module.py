"""
Based on the Concept-Embedding models [Zarlenga et al., 2022]

We will use the same encoder from the VAE that is trained in CITRIS. The decoder will just be an single
linear layer
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
import torch_explain as te
import numpy as np
import os

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from shared_utils.concept_models.shared import TabularEncoder

class CEM(pl.LightningModule):
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

        if conv:
            self.concept_encoder = Encoder(
                num_latents=num_concepts, 
                c_hid=c_hid,
                c_in=c_in,
                width=img_width,
                variational=False
            )
        else:
            self.concept_encoder = nn.Sequential(
                TabularEncoder(
                    ni=c_in,
                    no=4*c_hid,
                    nhidden=512, 
                    nlayers=6
                ),
                te.nn.ConceptEmbedding(
                    4*c_hid, 
                    num_concepts, 
                    8
                )
            )


        self.label_predictor = nn.Sequential(
            torch.nn.Linear(num_concepts*8, num_output)
        )
        self.regression = regression

    def forward(self, x):
        c_emb, c_hat = self.concept_encoder(x)
        label_pred = self.label_predictor(c_emb.reshape(len(c_emb), -1))
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
        c_emb, c_hat = self.concept_encoder(x)
        if self.regression:
            loss = F.mse_loss(c_hat, c)
        else:
            loss = F.binary_cross_entropy(c_hat, c)
        return loss, c_emb

    def _get_label_loss(self, c_emb, y, mode='train'):
        y_hat = self.label_predictor(c_emb.reshape(len(c_emb), -1))
        if self.regression:
            loss = F.mse_loss(y_hat, y)
        else:
            loss = F.binary_cross_entropy_with_logits(y_hat, y)
        return loss

    def training_step(self, batch, batch_idx):
        x, c, y = batch
        concept_loss, c_emb = self._get_concept_loss(x, c)
        label_loss = self._get_label_loss(c_emb, y)

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
    


class Encoder(nn.Module):
    """ 
    Convolution encoder network 
    We use a stack of convolutions with strides in every second convolution to reduce
    dimensionality. For the datasets in question, the network showed to be sufficient.
    """

    def __init__(self, c_hid, num_latents,
                 c_in=3,
                 width=32,
                 act_fn=lambda: nn.SiLU(),
                 use_batch_norm=True,
                 variational=True):
        super().__init__()
        num_layers = int(np.log2(width) - 2)
        NormLayer = nn.BatchNorm2d if use_batch_norm else nn.InstanceNorm2d
        self.scale_factor = nn.Parameter(torch.zeros(num_latents,))
        self.variational = variational
        self.net = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(c_in if l_idx == 0 else c_hid, 
                              c_hid,
                              kernel_size=3,
                              padding=1,
                              stride=2,
                              bias=False),
                    PositionLayer(c_hid) if l_idx == 0 else nn.Identity(),
                    NormLayer(c_hid),
                    act_fn(),
                    nn.Conv2d(c_hid, c_hid, kernel_size=3, stride=1, padding=1, bias=False),
                    NormLayer(c_hid),
                    act_fn()
                ) for l_idx in range(num_layers)
            ],
            nn.Flatten(),
            nn.Linear(4*4*c_hid, 4*c_hid),
            nn.LayerNorm(4*c_hid),
            act_fn(),
            te.nn.ConceptEmbedding(4*c_hid, num_latents, 8)
        )

    def forward(self, img):
        feats = self.net(img)
        if self.variational:
            mean, log_std = feats.chunk(2, dim=-1)
            s = F.softplus(self.scale_factor)
            log_std = torch.tanh(log_std / s) * s  # Stabilizing the prediction
            return mean, log_std
        else:
            return feats


class PositionLayer(nn.Module):
    """ Module for adding position features to images """

    def __init__(self, hidden_dim):
        super().__init__()
        self.pos_embed = nn.Linear(2, hidden_dim)

    def forward(self, x):
        pos = create_pos_grid(x.shape[2:], x.device)
        pos = self.pos_embed(pos)
        pos = pos.permute(2, 0, 1)[None]
        x = x + pos
        return x

def create_pos_grid(shape, device, stack_dim=-1):
    pos_x, pos_y = torch.meshgrid(torch.linspace(-1, 1, shape[0], device=device),
                                  torch.linspace(-1, 1, shape[1], device=device),
                                  indexing='ij')
    pos = torch.stack([pos_x, pos_y], dim=stack_dim)
    return pos