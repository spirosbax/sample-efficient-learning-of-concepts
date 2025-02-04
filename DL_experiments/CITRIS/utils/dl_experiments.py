from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import time
from scipy.stats import spearmanr
from typing import Union
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
import torch.utils.data as data

from training.datasets import Causal3DDataset
from models.citris.shared import CausalEncoder

from utils.utils import get_device


def get_groups(model, all_encs, all_latents):
    # For CITRIS we have the assignment targets
    # The causal3d ident is coarse, so we have 7 causal variables instead of 10. that means the target assignment 
    # consists of 8 = (7 + 1) assignments, the last one shows which encoding variables are not assigned to any causal variable
    # The latents that are collapsed are:
    #   - (pos.x, pos.y, pos.z) -> pos
    #   - (rot.alpha, rot.beta) -> rot
    # The causal variables are in order, so we have to take one of the first 3 columns, and one of the 4-5th column. 
    # We add 1 extra dimension onto which we try to map the unassigned encoding dimensions.
    #  
    # For iVAE:
    # The assignment is learned by mapping all the (32) latent factors to all causal variables
    # And then picking the ones with the highest R^2 correlation factor. There are no unassigned variables in this case
    # We can mimic this by using only the lasso and picking the heighest weights 
    # (Maybe doing a OT matching )


    # Create new tensor dataset for training (50%) and testing (50%)
    full_dataset = data.TensorDataset(all_encs, all_latents)
    train_size = int(0.5 * all_encs.shape[0])
    test_size = all_encs.shape[0] - train_size
    train_dataset, test_dataset = data.random_split(
        full_dataset, 
        lengths=[train_size, test_size], 
        generator=torch.Generator().manual_seed(42)
    )

    if hasattr(model, 'prior_t1'):
        # If a target assignmente exist we will use it to create the groups
        target_assignment = model.prior_t1.get_target_assignment(hard=True)
        groups = target_assignment.argmax(dim=1).cpu().numpy()
        print("prior_t1 groups are:", groups)
    elif hasattr(model, 'target_assignment') and model.target_assignment is not None:
        target_assignment = model.target_assignment.clone()
        groups = target_assignment.argmax(dim=1).cpu().numpy()
        print(groups)
    else:
        # If no target assignment exist we will train a network and use the R^2 scores to 
        # get a target assignment, like in the CITRIS paper
        target_assignment = torch.eye(all_encs.shape[-1])
        # Train network to predict causal factors from latent variables
        encoder = train_test_network(
            model, 
            train_dataset, 
            target_assignment,
            num_train_epochs=100
        )
        encoder.eval()
        
        # Record predictions of model on test and calculate distances
        test_inps = all_encs[test_dataset.indices]
        test_labels = all_latents[test_dataset.indices]
        test_exp_inps, test_exp_labels = _prepare_input(
            test_inps, 
            target_assignment.cpu(), 
            test_labels, 
            flatten_inp=False
        )
        
        pred_dict = encoder.forward(test_exp_inps.to(model.device))
        for key in pred_dict:
            pred_dict[key] = pred_dict[key].cpu()
        _, dists, norm_dists = encoder.calculate_loss_distance(pred_dict, test_exp_labels)
        
        # Calculate statistics (R^2, pearson, etc.)
        avg_norm_dists, r2_matrix = calc_r2_matrix(
            encoder, 
            test_labels, 
            norm_dists
        )
        max_r2 = torch.from_numpy(r2_matrix).argmax(axis=-1)
        ta = F.one_hot(max_r2, num_classes=r2_matrix.shape[-1]).float()
        ta_1 = ta[:,:3].sum(dim=-1, keepdims=True)
        ta_2 = ta[:,3:5].sum(dim=-1, keepdims=True)
        ta_3 = ta[:,5:]
        ta = torch.cat([ta_1, ta_2, ta_3], dim=-1)
        
        groups = ta.argmax(dim=1).cpu().numpy()
        print("trained groups are:", groups)
    return groups

def find_best_model_and_latents(
        model_dir, 
        model_class
    ):
    # Find all subdirectories that match the version pattern
    version_dirs = [d for d in os.listdir(model_dir) if re.match(r'^version_\d+$', d)]
    
    # If no version directories found, return None
    if not version_dirs:
        return None
    
    latest_version = max(version_dirs, key=lambda d: int(d.split('_')[1]))
    print(f"-> version: {latest_version}")
    ckpt_dir = os.path.join(model_dir, latest_version)

    # true_latents are sources
    device = get_device()
    model_dir = os.path.join(ckpt_dir, 'checkpoints/last.ckpt')
    model = model_class.load_from_checkpoint(model_dir).to(device)

    encodings_latents_fname = os.path.join(ckpt_dir, 'encodings_latents.npz')

    if os.path.isfile(encodings_latents_fname):
        arrs = np.load(encodings_latents_fname)
        all_encs = torch.from_numpy(arrs["encs"])
        all_latents = torch.from_numpy(arrs["latents"])
    else:
        # if not we have to calculate the latents
        print("Computing latents and encodings")

        data_dir = 'data/causal3d_time_dep_all7_conthue_01_coarse'
        test_loader = load_datasets(data_dir=data_dir)

        all_encs = []
        all_latents = []
        for batch in test_loader:
            imgs, latents = batch
            encs = model.encode(imgs.to(model.device)).detach().cpu()
            all_encs.append(encs)
            all_latents.append(latents)
        all_encs = torch.cat(all_encs, dim=0)
        all_latents = torch.cat(all_latents, dim=0)
        # 25000 x 32, 25000 x 10

        all_encs = all_encs.numpy()
        all_latents = all_latents.numpy()

        np.savez(
            os.path.join(ckpt_dir, "encodings_latents.npz"), 
            encs=all_encs, 
            latents=all_latents
        )


    # Return the directory with the largest version number
    return ckpt_dir, model, all_encs, all_latents

def load_datasets(data_dir):
    dataset_args = {
        'coarse_vars': True, 
        'exclude_vars': None, 
        'exclude_objects': None
    }
    test_args = lambda train_set: {'causal_vars': train_set.full_target_names}

    train_dataset = Causal3DDataset(
        data_folder=data_dir, 
        split='train', 
        single_image=False, 
        triplet=False, 
        seq_len=2,
        **dataset_args
    )
    test_dataset = Causal3DDataset(
        data_folder=data_dir,
        split='test_indep', 
        single_image=True, 
        triplet=False, 
        return_latents=True, 
        **dataset_args, 
        **test_args(train_dataset)
    )

    test_loader = data.DataLoader(
        test_dataset, 
        batch_size=256,
        shuffle=False, 
        drop_last=False,
        num_workers=0
    )
    return test_loader

    

def train_test_network(
        model, 
        train_dataset, 
        target_assignment,
        num_train_epochs: int=100,
        cluster: bool=False
) -> CausalEncoder:
        device = model.device
        if hasattr(model, 'causal_encoder'):
            causal_var_info = model.causal_encoder.hparams.causal_var_info
        else:
            causal_var_info = model.hparams.causal_var_info
        # We use one, sufficiently large network that predicts for any input all causal variables
        # To iterate over the different sets, we use a mask which is an extra input to the model
        # This is more efficient than using N networks and showed same results with large hidden size
        encoder = CausalEncoder(
            c_hid=128,
            lr=4e-3,
            causal_var_info=causal_var_info,
            single_linear=True,
            c_in=model.hparams.num_latents*2,
            warmup=0
        )
        optimizer, _ = encoder.configure_optimizers()
        if isinstance(optimizer, (list, tuple)):
            optimizer = optimizer[0]
        print("dataset", len(train_dataset))
        train_loader = data.DataLoader( 
            train_dataset, 
            shuffle=True, 
            drop_last=False, 
            batch_size=512
        )
        target_assignment = target_assignment.to(device)
        encoder.to(device)
        encoder.train()
        with torch.enable_grad():
            range_iter = range(num_train_epochs)
            if not cluster:
                range_iter = tqdm(
                    range_iter, 
                    leave=False, 
                    desc=f'Training correlation encoder'
                )
            for epoch_idx in range_iter:
                avg_loss = 0.0
                for inps, latents in train_loader:
                    inps = inps.to(device)
                    latents = latents.to(device)
                    inps, latents = _prepare_input(inps, target_assignment, latents)
                    loss = encoder._get_loss([inps, latents], mode=None)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    avg_loss += loss.item()
        return encoder

def _prepare_input(inps, target_assignment, latents, flatten_inp=True):
    ta = target_assignment.detach()[None,:,:].expand(inps.shape[0], -1, -1)
    inps = torch.cat([inps[:,:,None] * ta, ta], dim=-2).permute(0, 2, 1)
    latents = latents[:,None].expand(-1, inps.shape[1], -1)
    if flatten_inp:
        inps = inps.flatten(0, 1)
        latents = latents.flatten(0, 1)

    return inps, latents


def calc_r2_matrix(
    encoder, 
    test_labels, 
    norm_dists, 
):
    avg_pred_dict = OrderedDict()
    for i, var_key in enumerate(encoder.hparams.causal_var_info):
        var_info = encoder.hparams.causal_var_info[var_key]
        gt_vals = test_labels[...,i]
        if var_info.startswith('continuous'):
            avg_pred_dict[var_key] = gt_vals.mean(dim=0, keepdim=True).expand(gt_vals.shape[0],)
        elif var_info.startswith('angle'):
            avg_angle = torch.atan2(torch.sin(gt_vals).mean(dim=0, keepdim=True), 
                                    torch.cos(gt_vals).mean(dim=0, keepdim=True)).expand(gt_vals.shape[0],)
            avg_angle = torch.where(avg_angle < 0.0, avg_angle + 2*np.pi, avg_angle)
            avg_pred_dict[var_key] = torch.stack([torch.sin(avg_angle), torch.cos(avg_angle)], dim=-1)
        elif var_info.startswith('categ'):
            gt_vals = gt_vals.long()
            mode = torch.mode(gt_vals, dim=0, keepdim=True).values
            avg_pred_dict[var_key] = F.one_hot(mode, int(var_info.split('_')[-1])).float().expand(gt_vals.shape[0], -1)
        else:
            assert False, f'Do not know how to handle key \"{var_key}\" in R2 statistics.'

    _, _, avg_norm_dists = encoder.calculate_loss_distance(
        avg_pred_dict, 
        test_labels, 
        keep_sign=True
    )

    r2_matrix = []
    for var_key in encoder.hparams.causal_var_info:
        ss_res = (norm_dists[var_key] ** 2).mean(dim=0)
        ss_tot = (avg_norm_dists[var_key] ** 2).mean(dim=0, keepdim=True)
        r2 = 1 - ss_res / ss_tot
        r2_matrix.append(r2)
    r2_matrix = torch.stack(r2_matrix, dim=-1).detach().cpu().numpy()
    return avg_norm_dists, r2_matrix


def calc_spearman_matrix(
    encoder, 
    pred_dict, 
    test_labels
):
    spearman_matrix = []
    for i, var_key in enumerate(encoder.hparams.causal_var_info):
        var_info = encoder.hparams.causal_var_info[var_key]
        gt_vals = test_labels[...,i]
        pred_val = pred_dict[var_key]
        if var_info.startswith('continuous'):
            spearman_preds = pred_val.squeeze(dim=-1)  # Nothing needs to be adjusted
        elif var_info.startswith('angle'):
            spearman_preds = F.normalize(pred_val, p=2.0, dim=-1)
            gt_vals = torch.stack([torch.sin(gt_vals), torch.cos(gt_vals)], dim=-1)
        elif var_info.startswith('categ'):
            spearman_preds = pred_val.argmax(dim=-1).float()
        else:
            assert False, f'Do not know how to handle key \"{var_key}\" in Spearman statistics.'

        gt_vals = gt_vals.cpu().numpy()
        spearman_preds = spearman_preds.detach().cpu().numpy()
        results = torch.zeros(spearman_preds.shape[1],)
        for j in range(spearman_preds.shape[1]):
            if len(spearman_preds.shape) == 2:
                if np.unique(spearman_preds[:,j]).shape[0] == 1:
                    results[j] = 0.0
                else:
                    results[j] = spearmanr(spearman_preds[:,j], gt_vals).correlation
            elif len(spearman_preds.shape) == 3:
                num_dims = spearman_preds.shape[-1]
                for k in range(num_dims):
                    if np.unique(spearman_preds[:,j,k]).shape[0] == 1:
                        results[j] = 0.0
                    else:
                        results[j] += spearmanr(spearman_preds[:,j,k], gt_vals[...,k]).correlation
                results[j] /= num_dims
            
        spearman_matrix.append(results)
    
    spearman_matrix = torch.stack(spearman_matrix, dim=-1).detach().cpu().numpy()
    return spearman_matrix


