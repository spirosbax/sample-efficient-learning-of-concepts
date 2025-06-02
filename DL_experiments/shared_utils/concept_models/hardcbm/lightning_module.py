"""
Based on the Addressing leakage in Concept Bottleneck Modelss [Havasi et al., 2022]

PyTorch version adapted from https://github.com/mvandenhi/SCBM/blob/main/models/models.py

We will use the same encoder from the VAE that is trained in CITRIS. 
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import RelaxedBernoulli, MultivariateNormal

import pytorch_lightning as pl
import os

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from shared_utils.concept_models.shared import FeatureEncoder, TabularEncoder

class HardCBM(nn.Module):
    """HardCBM with side-channel and autoregressive options"""

    def __init__(
        self,
        c_hid, 
        num_concepts, 
        num_output, 
        num_latent_concepts=50,
        type="hard",
        conv=True,
        autoregressive=True,
        num_monte_carlo=100,
        img_width=64, 
        c_in=3,

    ):
        super().__init__()

        self.num_monte_carlo = num_monte_carlo
        self.num_concepts = num_concepts
        self.curr_temp = 1.0

        # Ouput is 4 x 4 x c_hid
        if conv:
            self.encoder = FeatureEncoder(
                    num_latents=num_concepts, 
                    c_hid=c_hid,
                    c_in=c_in,
                    width=img_width,
                    variational=False
                )
            n_features = 4 * 4 * c_hid
        else:
            self.encoder = TabularEncoder(
                    ni=c_in,
                    no=num_concepts,
                    nhidden=512, 
                    nlayers=6
                )
            n_features = num_concepts

        if autoregressive:
            self.concept_predictor = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(n_features + i, num_latent_concepts, bias=True),
                        nn.LeakyReLU(),
                        nn.Linear(num_latent_concepts, 1, bias=True),
                    )
                    for i in range(num_concepts)
                ]
            )

        else:
            self.concept_predictor = nn.Linear(
                n_features, num_concepts, bias=True
            )

        self.act_c = nn.Sigmoid()

        # Label predictor 
        self.pred_dim = num_output
        
        self.head = nn.Sequential(
            torch.nn.Linear(num_concepts, num_output)
        )

    def forward(self, x, epoch, c_true=None, validation=False, concepts_train_ar=False):
        intermediate = self.encoder(x)

        if validation:
            c_prob, c_hard = [], []
            for predictor in self.concept_predictor:
                if c_prob:
                    concept = []
                    for i in range(
                        self.num_monte_carlo
                    ):  # MCMC samples for evaluation and interventions, but not for training
                        concept_input_i = torch.cat(
                            [intermediate, torch.cat(c_hard, dim=1)[..., i]], dim=1
                        )
                        concept.append(self.act_c(predictor(concept_input_i)))
                    concept = torch.cat(concept, dim=-1)
                    c_relaxed = torch.bernoulli(concept)[:, None, :]
                    concept = concept[:, None, :]
                    concept_hard = c_relaxed

                else:
                    concept_input = intermediate
                    concept = self.act_c(predictor(concept_input))
                    concept = concept.unsqueeze(-1).expand(
                        -1, -1, self.num_monte_carlo
                    )
                    c_relaxed = torch.bernoulli(concept)
                    concept_hard = c_relaxed
                c_prob.append(concept)
                c_hard.append(concept_hard)
            c_prob = torch.cat([c_prob[i] for i in range(self.num_concepts)], dim=1)
            c = torch.cat([c_hard[i] for i in range(self.num_concepts)], dim=1)
        else:
             # Training
            if c_true is None and concepts_train_ar is not False:
                c_prob, c_hard = [], []
                for c_idx, predictor in enumerate(self.concept_predictor):
                    if c_hard:
                        concept_input = torch.cat(
                            [intermediate, concepts_train_ar[:, :c_idx]], dim=1
                        )
                    else:
                        concept_input = intermediate
                    concept = self.act_c(predictor(concept_input))

                    # No Gumbel softmax because backprop can happen through the input connection
                    c_relaxed = torch.bernoulli(concept)
                    concept_hard = c_relaxed

                    # NOTE that the following train-time variables are overly good because they are taking ground truth as input. At validation time, we sample
                    c_prob.append(concept)
                    c_hard.append(concept_hard)
                c_prob = torch.cat(
                    [c_prob[i] for i in range(self.num_concepts)], dim=1
                )
                c = torch.cat([c_hard[i] for i in range(self.num_concepts)], dim=1)

            else:  # Training the head with the GT concepts as input
                c_prob = c_true.float()
                c = c_true.float()

        # get predicted targets
        if validation:
            # Hard CBM or validation of AR. Takes MCMC samples.
            # MCMC loop for predicting label
            y_pred_probs_i = 0
            for i in range(self.num_monte_carlo):
                c_i = c[:, :, i]
                y_pred_logits_i = self.head(c_i)
                if self.pred_dim == 1:
                    y_pred_probs_i += torch.sigmoid(y_pred_logits_i)
                else:
                    y_pred_probs_i += torch.softmax(y_pred_logits_i, dim=1)
            y_pred_probs = y_pred_probs_i / self.num_monte_carlo

            if self.pred_dim == 1:
                y_pred_logits = torch.logit(y_pred_probs, eps=1e-6)
            else:
                y_pred_logits = torch.log(y_pred_probs + 1e-6)
        else:
            # CEM or training of AR. Takes ground truth concepts.
            # If CEM: c are predicte embeddings, if AR: c are ground truth concepts
            y_pred_logits = self.head(c)

        return c_prob, y_pred_logits, c

    def compute_temperature(self, epoch, device):
        final_temp = torch.tensor([0.5], device=device)
        init_temp = torch.tensor([1.0], device=device)
        rate = (torch.log(final_temp) - torch.log(init_temp)) / float(self.num_epochs)
        curr_temp = max(init_temp * torch.exp(rate * epoch), final_temp)
        self.curr_temp = curr_temp
        return curr_temp

    def freeze_c(self):
        self.head.apply(freeze_module)

    def freeze_t(self):
        self.head.apply(unfreeze_module)
        self.encoder.apply(freeze_module)
        self.concept_predictor.apply(freeze_module)


def freeze_module(m):
    m.eval()
    for param in m.parameters():
        param.requires_grad = False


def unfreeze_module(m):
    m.train()
    for param in m.parameters():
        param.requires_grad = True