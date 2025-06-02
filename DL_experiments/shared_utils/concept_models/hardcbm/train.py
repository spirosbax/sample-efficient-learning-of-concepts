"""
Adapted from https://github.com/mvandenhi/SCBM/blob/main/utils/training.py
"""

import os
from os.path import join
from pathlib import Path
import time
import uuid

import torch
import torch.optim as optim

from shared_utils.concept_models.hardcbm.loss import create_loss


def train(
    model,
    train_loader, 
    val_loader, 
    test_loader,
    epochs=100,
    validate_per_epoch=100
):
    # Setting device on GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Additional info when using cuda
    if device.type == "cuda":
        print("Using", torch.cuda.get_device_name(0))
    else:
        print("No GPU available")

    # Numbers of training epochs
    t_epochs = epochs
    c_epochs = epochs

    model.to(device)
    loss_fn = create_loss(num_classes=2)
    # ---------------------------------
    #            Training
    # ---------------------------------
    print("TRAINING HardCBM with AR")

    # Pretraining autoregressive concept structure for AR baseline
    # if (
    #     config.model.get("pretrain_concepts")
    #     and config.model.concept_learning == "autoregressive"
    # ):
    #     print("\nStarting concepts pre-training!\n")
    #     mode = "c"

    #     # Freeze the target prediction part
    #     model.freeze_c()
    #     model.encoder.apply(freeze_module)  # Freezing the encoder

    #     c_optimizer = create_optimizer(config.model, model)
    #     lr_scheduler = optim.lr_scheduler.StepLR(
    #         c_optimizer,
    #         step_size=config.model.decrease_every,
    #         gamma=1 / config.model.lr_divisor,
    #     )
    #     for epoch in range(p_epochs):
    #         # Validate the model periodically
    #         if epoch % config.model.validate_per_epoch == 0:
    #             print("\nEVALUATION ON THE VALIDATION SET:\n")
    #             validate_one_epoch(
    #                 val_loader, model, metrics, epoch, config, loss_fn, device
    #             )
    #         train_one_epoch(
    #             train_loader,
    #             model,
    #             c_optimizer,
    #             mode,
    #             metrics,
    #             epoch,
    #             config,
    #             loss_fn,
    #             device,
    #         )
    #         lr_scheduler.step()

    #     model.encoder.apply(unfreeze_module)  # Unfreezing the encoder

    # For sequential & independent training: first stage is training of concept encoder
    print("\nStarting concepts training!\n")
    mode = "c"

    # Freeze the target prediction part
    model.freeze_c()

    c_optimizer = optim.Adam([{
        "params": filter(lambda p: p.requires_grad, model.parameters()),
        # "params": model.parameters(), 
        "lr": 0.001}
    ])
    lr_scheduler = optim.lr_scheduler.StepLR(
        c_optimizer,
        step_size=1000,
        gamma=0.1,
    )
    for epoch in range(c_epochs):
        # Validate the model periodically
        if epoch % validate_per_epoch == 0:
            print("\nEVALUATION ON THE VALIDATION SET:\n")
            validate_one_epoch(
                val_loader, model, epoch, loss_fn, device
            )
        train_one_epoch(
            train_loader,
            model,
            c_optimizer,
            mode,
            epoch,
            loss_fn,
            device,
        )
        lr_scheduler.step()

    # Prepare parameters for target training by unfreezing the target prediction part and freezing the concept encoder
    model.freeze_t()

    print("\nStarting target training!\n")
    mode = "t"

    optimizer = optim.Adam([{
        "params": filter(lambda p: p.requires_grad, model.parameters()),
        # "params": model.parameters(), 
        "lr": 0.001}
    ])
    lr_scheduler = optim.lr_scheduler.StepLR(
        c_optimizer,
        step_size=1000,
        gamma=0.1,
    )

    # Second stage is training of target predictor
    for epoch in range(0, t_epochs):
        if epoch % validate_per_epoch == 0:
            print("\nEVALUATION ON THE VALIDATION SET:\n")
            validate_one_epoch(
                val_loader, model, epoch, loss_fn, device
            )
        train_one_epoch(
            train_loader,
            model,
            optimizer,
            mode,
            epoch,
            loss_fn,
            device,
        )
        lr_scheduler.step()

    model.apply(freeze_module)

    print("\nTRAINING FINISHED", flush=True)

    return model


def train_one_epoch(
    train_loader, model, optimizer, mode, epoch, loss_fn, device
):
    """
    Train a baseline method for one epoch.

    This function trains the CEM/AR/CBM for one epoch using the provided training data loader, model, optimizer, and loss function.
    It supports different training modes and updates the model parameters accordingly. The function also computes and logs
    various metrics during the training process.

    Args:
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        model (torch.nn.Module): The SCBM model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer for training the model.
        mode (str): The training mode. Supported modes are:
                    - "j": Joint training of the model.
                    - "c": Training the concept head only.
                    - "t": Training the classifier head only.
        metrics (object): An object to track and compute metrics during training.
        epoch (int): The current epoch number.
        config (dict): Configuration dictionary containing model and training settings.
        loss_fn (callable): The loss function used to compute losses.
        device (torch.device): The device to run the computations on.

    Returns:
        None

    Notes:
        - Depending on the training mode, certain parts of the model are set to evaluation mode.
        - The function iterates over the training data, performs forward and backward passes, and updates the model parameters.
        - Metrics are computed and logged at the end of each epoch.
    """

    model.train()

    if mode == "c":
        model.head.eval()
    elif mode == "t":
        model.encoder.eval()

    for k, batch in enumerate(train_loader):
        batch_img, batch_c, batch_y = batch
        batch_img = batch_img.to(device)
        batch_c = batch_c.to(device)
        batch_y = batch_y.to(device)

        # batch_features, target_true = batch["features"].to(device), batch[
        #     "labels"
        # ].to(device)
        # concepts_true = batch["concepts"].to(device)
        # Forward pass
        if mode == "c":
            concepts_pred_probs, target_pred_logits, concepts_hard = model(
                batch_img, epoch, concepts_train_ar=batch_c
            )
        else:
            concepts_pred_probs, target_pred_logits, concepts_hard = model(
                batch_img, epoch, batch_c 
            )
        # Backward pass depends on the training mode of the model
        optimizer.zero_grad()
        # Compute the loss
        target_loss, concepts_loss, total_loss = loss_fn(
            concepts_pred_probs, batch_c, target_pred_logits, batch_y
        )

        if mode == "j":
            total_loss.backward()
        elif mode == "c":
            concepts_loss.backward()
        else:
            target_loss.backward()
        optimizer.step()  # perform an update

    return


def validate_one_epoch(
    loader,
    model,
    epoch,
    loss_fn,
    device,
    test=False,
    concept_names_graph=None,
):
    """
    Validate a baseline method for one epoch.

    This function evaluates the CEM/AR/CBM for one epoch using the provided data loader, model, and loss function.
    It computes and logs various metrics during the validation process.

    Args:
        loader (torch.utils.data.DataLoader): DataLoader for the validation or test data.
        model (torch.nn.Module): The model to be validated.
        metrics (object): An object to track and compute metrics during validation.
        epoch (int): The current epoch number.
        config (dict): Configuration dictionary containing model and validation settings.
        loss_fn (callable): The loss function used to compute losses.
        device (torch.device): The device to run the computations on.
        test (bool, optional): Flag indicating whether this is the final evaluation on the test set. Default is False.

    Returns:
        None

    Notes:
        - The function sets the model to evaluation mode and disables gradient computation.
        - It iterates over the validation data, performs forward passes, and computes the losses.
        - Metrics are computed and logged at the end of the validation epoch.
    """
    model.eval()

    with torch.no_grad():
        for k, batch in enumerate(loader):
            batch_img, batch_c, batch_y = batch
            batch_img = batch_img.to(device)
            batch_c = batch_c.to(device)
            batch_y = batch_y.to(device)
            
            # batch_features, target_true = batch["features"].to(device), batch[
            #     "labels"
            # ].to(device)
            # concepts_true = batch["concepts"].to(device)

            concepts_pred_probs, target_pred_logits, concepts_hard = model(
                batch_img, epoch, validation=True
            )
            concepts_pred_probs = torch.mean(
                concepts_pred_probs, dim=-1
            )  # Calculating the metrics on the average probabilities from MCMC

            target_loss, concepts_loss, total_loss = loss_fn(
                concepts_pred_probs, batch_c, target_pred_logits, batch_y
            )

    print(target_loss, concepts_loss, total_loss)

    return


def freeze_module(m):
    m.eval()
    for param in m.parameters():
        param.requires_grad = False