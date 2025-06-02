"""
Utility methods for constructing loss functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


def create_loss(num_classes, alpha=1):
    """
    Create and return a loss function based on the configuration.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        nn.Module: The loss function.
    """
    return CBLoss(
        num_classes=num_classes,
        reduction="mean"
    )

class CBLoss(nn.Module):
    """
    Loss function for the Concept Bottleneck Model (CBM).
    """

    def __init__(
        self,
        num_classes: Optional[int] = 2,
        reduction: str = "mean"
    ) -> None:
        """
        Initialize the CBLoss.

        Args:
            num_classes (int, optional): Number of target classes.
            reduction (str, optional): Reduction method for the loss.
            alpha (float, optional): Weight in joint training.
            config (dict, optional): Configuration dictionary.
        """
        super(CBLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = 1.0
        self.reduction = reduction

    def forward(
        self,
        concepts_pred_probs: Tensor,
        concepts_true: Tensor,
        target_pred_logits: Tensor,
        target_true: Tensor,
    ) -> Tensor:
        """
        Compute the loss.

        Args:
            concepts_pred_probs (Tensor): Predicted concept probabilities.
            concepts_true (Tensor): Ground-truth concept values.
            target_pred_logits (Tensor): Predicted target logits.
            target_true (Tensor): Ground-truth target values.

        Returns:
            Tensor: Target loss, concept loss, and total loss.
        """

        concepts_loss = 0

        assert torch.all((concepts_true == 0) | (concepts_true == 1))

        for concept_idx in range(concepts_true.shape[1]):
            c_loss = F.binary_cross_entropy(
                concepts_pred_probs[:, concept_idx],
                concepts_true[:, concept_idx].float(),
                reduction=self.reduction,
            )
            concepts_loss += c_loss
        concepts_loss = self.alpha * concepts_loss

        if self.num_classes == 2:
            # Logits to probs
            target_pred_probs = nn.Sigmoid()(target_pred_logits.squeeze(0))
            target_loss = F.binary_cross_entropy(
                target_pred_probs, target_true.float(), reduction=self.reduction
            )
        else:
            target_loss = F.cross_entropy(
                target_pred_logits, target_true.long(), reduction=self.reduction
            )

        total_loss = target_loss + concepts_loss

        return target_loss, concepts_loss, total_loss

