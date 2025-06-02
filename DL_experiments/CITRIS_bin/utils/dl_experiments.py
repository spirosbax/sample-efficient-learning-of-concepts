from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], '../..'))

import re
import time
from scipy.stats import spearmanr
from typing import Union
from tqdm.auto import tqdm

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.preprocessing import SplineTransformer
from sklearn.kernel_approximation import RBFSampler

import torch
import torch.nn.functional as F
import torch.utils.data as data

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger

from training.datasets import Causal3DDataset, ConceptDataset
from models.citris.shared import CausalEncoder
from models.cbm.lightning_module import SimpleCBM
from models.cem.lightning_module import CEM 

from utils.utils import get_device

from shared_utils.experiment import calc_perm_errors, set_seed
from permutation_estimator.estimator import FeaturePermutationEstimator, KernelizedPermutationEstimator


def do_cbm_test(
    all_latents,
    y_values, 
    Ns, 
    seed, 
    model_type,
    base_dir, 
    cluster
):
    latent_dataset = data.TensorDataset(torch.from_numpy(all_latents))   
    label_dataset = data.TensorDataset(torch.from_numpy(y_values[:, np.newaxis]))

    data_dir = 'data/causal3d_time_dep_all7_conthue_01_coarse'
    img_dataset = Causal3DDataset(
        data_folder=data_dir, 
        split='test_indep', 
        single_image=True, 
        triplet=False, 
        coarse_vars=True,
        exclude_vars=None,
        exclude_objects=None
    )
    dataset = ConceptDataset(img_dataset, latent_dataset, label_dataset)

    ckpt_dir = os.path.join(base_dir, "CBM")

    mse_label_arr = np.zeros(len(Ns))
    r2_label_arr = np.zeros(len(Ns))

    mse_concept_arr = np.zeros(len(Ns))
    r2_concept_arr = np.zeros(len(Ns))

    times_arr = np.zeros(len(Ns))

    num_workers = 0 if cluster else 0

    for i, N in enumerate(Ns):
        print(f"working on {N} datapoints")
        logger = CSVLogger(ckpt_dir, name=f"N_{N}")

        train_dataset, test_dataset, _ = data.random_split(
            dataset, 
            [N, N, len(dataset) - 2*N], 
            torch.Generator().manual_seed(seed)
        )

        train_loader = data.DataLoader(
            train_dataset, 
            batch_size=256,
            shuffle=True,
            drop_last=False, 
            num_workers=num_workers
        )
        test_loader = data.DataLoader(
            test_dataset, 
            batch_size=256,
            shuffle=False,
            drop_last=False, 
            num_workers=num_workers
        )

        if model_type == "CBM":
            cbm = SimpleCBM(
                c_hid=32, 
                num_concepts=all_latents.shape[1], 
                num_output=1
            )
        elif model_type == "CEM":
            cbm = CEM(
                c_hid=32, 
                num_concepts=all_latents.shape[1],
                num_output=1
            )

        if cluster:
            trainer = Trainer(max_epochs=100, logger=logger, enable_progress_bar=False)
        else:
            trainer = Trainer(max_epochs=100, logger=logger)

        start = time.time()
        trainer.fit(cbm, train_loader, test_loader)
        end = time.time()

        cbm.eval()

        all_c = []
        all_c_hat = []

        all_y = []
        all_y_hat = []
        for batch in test_loader:
            x, c, y = batch
            if model_type == "CBM":
                c_hat = cbm.concept_encoder(x.to(cbm.device))
                y_hat = cbm.label_predictor(c_hat).detach().cpu()
            else:
                c_emb, c_hat = cbm.concept_encoder(x.to(cbm.device))
                y_hat = cbm.label_predictor(c_emb.reshape(len(c_emb), -1)).detach().cpu()
            all_c_hat.append(c_hat.detach().cpu())
            all_c.append(c)

            all_y_hat.append(y_hat)
            all_y.append(y)

        all_c = torch.cat(all_c, dim=0)
        all_c_hat = torch.cat(all_c_hat, dim=0)

        all_y = torch.cat(all_y, dim=0)
        all_y_hat = torch.cat(all_y_hat, dim=0)

        mse_concept_dims = mean_squared_error(all_c, all_c_hat, multioutput="raw_values")
        r2_concept_dims = r2_score(all_c, all_c_hat, multioutput="raw_values")

        mse_label_dims = mean_squared_error(all_y, all_y_hat, multioutput="raw_values")
        r2_label_dims = r2_score(all_y, all_y_hat, multioutput="raw_values")

        mse_concept_arr[i] = mse_concept_dims.mean()
        r2_concept_arr[i] = r2_concept_dims.mean()

        mse_label_arr[i] = mse_label_dims.mean()
        r2_label_arr[i] = r2_label_dims.mean()

        times_arr[i] = end - start

    result = {
        "mse_label": mse_label_arr, 
        "r2_label": r2_label_arr, 
        "mse_concept": mse_concept_arr, 
        "r2_concept": r2_concept_arr, 
        "times": times_arr
    }
    return result

def run_permutation_estimation(
    regularizer: str, 
    optim_kwargs: dict,
    N: int,
    feature_transform: list,
    n_features: int,
    encs: np.array, 
    latents: np.array,
    y_values: np.array,
    groups=None,
) -> np.array:
    n = encs.shape[0]
    nr_latents = latents.shape[1]
    z_dim = encs.shape[1]

    # TODO: change this somewhere else
    rand_perm = np.arange(nr_latents)

    # We use as many test data points as train data points
    indices = np.random.choice(size=n, a=np.arange(n),replace=False)
    train_idx, test_idx = indices[:N], indices[-N:]

    train_encs = encs[train_idx, :]
    train_latents = latents[train_idx, :]
    test_encs = encs[test_idx, :]
    test_latents = latents[test_idx, :]

    train_y = y_values[train_idx,]
    test_y = y_values[test_idx]

    if isinstance(feature_transform, str):
        estimator = KernelizedPermutationEstimator(
            regularizer=regularizer, 
            optim_kwargs=optim_kwargs, 
            kernel=feature_transform,
            parameter=1,
            n_features=n_features, 
            d_variables=z_dim,
            groups=groups
        )
    else:
        two_stage = "ridge" if regularizer == "lasso" else None
        estimator = FeaturePermutationEstimator(
            regularizer=regularizer, 
            optim_kwargs=optim_kwargs, 
            feature_transform=feature_transform, 
            n_features=n_features,
            d_variables=z_dim,
            groups=groups, 
            two_stage=two_stage
        )

    # To have it the same as the CBM model
    predictor = SGDRegressor()
    # predictor = Ridge(alpha=0.05, fit_intercept=True)

    if groups is None:
        res = estimator.fit(
            train_encs.T, train_latents.T
        )
    else:
        res = estimator.fit(
            train_encs.T, train_latents.T,
            recover_weights=False
        )

    train_latent_hat = estimator.predict_match(train_encs.T)
    test_latents_hat = estimator.predict_match(test_encs.T)

    mse_concept_dims = mean_squared_error(test_latents, test_latents_hat.T, multioutput="raw_values")
    r2_concept_dims = r2_score(test_latents, test_latents_hat.T, multioutput="raw_values")

    mse_concept = mse_concept_dims.mean()
    r2_concept = r2_concept_dims.mean()

    predictor.fit(train_latent_hat.T, train_y)
    test_y_hat = predictor.predict(test_latents_hat.T)

    mse_label_dims = mean_squared_error(test_y, test_y_hat, multioutput="raw_values")
    r2_label_dims = r2_score(test_y, test_y_hat, multioutput="raw_values")

    mse_label = mse_label_dims.mean()
    r2_label = r2_label_dims.mean()

    errors_match = calc_perm_errors(rand_perm, res["perm_hat_match"])

    times_match = res["time_match"]

    results = {
        "error_match": errors_match,
        "mse_concept": mse_concept,
        "r2_concept": r2_concept, 
        "mse_label": mse_label,
        "r2_label": r2_label,
        "times_match": times_match,
    }
    print(results)

    return results

def do_vae_test(
    all_latents,
    all_encs,
    y_values,
    alphas,
    Ns, 
    groups
) -> np.array:
    z_dim = all_encs.shape[1]

    n_knots = 6
    degree = 3
    n_features = n_knots + degree - 1

    splines_all_dims = [SplineTransformer(
        n_knots=n_knots,
        degree=degree) for _ in range(z_dim)]
    rff_all_dims = [RBFSampler(n_components=n_features) for _ in range(z_dim)]

    mse_concept = np.zeros((5, len(alphas), len(Ns)))
    r2_concept = np.zeros((5, len(alphas), len(Ns)))

    mse_label = np.zeros((5, len(alphas), len(Ns)))
    r2_label = np.zeros((5, len(alphas), len(Ns)))

    times = np.zeros((5, len(alphas), len(Ns)))

    overlapping_settings = {
        "latents": all_latents, 
        "encs": all_encs,
        "y_values": y_values
    }

    for i, alpha in enumerate(alphas):
        for j, N in enumerate(Ns):
            print(f"Working on experiments with alpha: {alpha} and N: {N}")
            overlapping_settings["N"] = N 

            results_linear = run_permutation_estimation(
                regularizer="group", 
                optim_kwargs={"alpha": alpha, "scale_reg": "group_size", "l1_reg": 0},
                feature_transform=None,
                n_features=1,
                groups=groups,
                **overlapping_settings
            )
            results_rff = run_permutation_estimation(
                regularizer="group", 
                optim_kwargs={"alpha": alpha, "scale_reg": "group_size", "l1_reg": 0},
                feature_transform=rff_all_dims,
                n_features=n_features,
                groups=groups,
                **overlapping_settings
            )
            results_spline = run_permutation_estimation(
                regularizer="group", 
                optim_kwargs={"alpha": alpha, "scale_reg": "group_size", "l1_reg": 0},
                feature_transform=splines_all_dims,
                n_features=n_features,
                groups=groups,
                **overlapping_settings
            )
            n_kernel = min(N, 20)
            results_laplacian = run_permutation_estimation(
                regularizer="group", 
                optim_kwargs={"alpha": alpha, "scale_reg": "group_size", "l1_reg": 0},
                feature_transform="laplacian",
                n_features=n_kernel,
                groups=groups,
                **overlapping_settings
            )
            results_two_stage = run_permutation_estimation(
                regularizer="lasso", 
                optim_kwargs={"alpha": alpha},
                feature_transform=splines_all_dims,
                n_features=n_features,
                groups=groups,
                **overlapping_settings
            )

            mse_concept[0, i, j] = results_linear["mse_concept"]
            mse_concept[1, i, j] = results_spline["mse_concept"]
            mse_concept[2, i, j] = results_rff["mse_concept"]
            mse_concept[3, i, j] = results_laplacian["mse_concept"]
            mse_concept[4, i, j] = results_two_stage["mse_concept"]

            r2_concept[0, i, j] = results_linear["r2_concept"]
            r2_concept[1, i, j] = results_spline["r2_concept"]
            r2_concept[2, i, j] = results_rff["r2_concept"]
            r2_concept[3, i, j] = results_laplacian["r2_concept"]
            r2_concept[4, i, j] = results_two_stage["r2_concept"]

            mse_label[0, i, j] = results_linear["mse_label"]
            mse_label[1, i, j] = results_spline["mse_label"]
            mse_label[2, i, j] = results_rff["mse_label"]
            mse_label[3, i, j] = results_laplacian["mse_label"]
            mse_label[4, i, j] = results_two_stage["mse_label"]

            r2_label[0, i, j] = results_linear["r2_label"]
            r2_label[1, i, j] = results_spline["r2_label"]
            r2_label[2, i, j] = results_rff["r2_label"]
            r2_label[3, i, j] = results_laplacian["r2_label"]
            r2_label[4, i, j] = results_two_stage["r2_label"]

            times[0, i, j] = results_linear["times_match"]
            times[1, i, j] = results_spline["times_match"]
            times[2, i, j] = results_rff["times_match"]
            times[3, i, j] = results_laplacian["times_match"]
            times[4, i, j] = results_two_stage["times_match"]


    results = {
        "mse_label": mse_label,
        "r2_label": r2_label,
        "mse_concept": mse_concept,
        "r2_concept": r2_concept,
        "times": times,
    }   

    return results

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

    data_dir = 'data/causal3d_time_dep_all7_conthue_01_coarse'
    arg_names = load_arg_names(data_dir=data_dir)

    if os.path.isfile(encodings_latents_fname):
        arrs = np.load(encodings_latents_fname)
        all_encs = torch.from_numpy(arrs["encs"])
        all_latents = torch.from_numpy(arrs["latents"])
    else:
        # if not we have to calculate the latents
        print("Computing latents and encodings")
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
    return ckpt_dir, model, all_encs, all_latents, arg_names


def load_arg_names(data_dir):
    dataset_args = {
        'coarse_vars': True, 
        'exclude_vars': None, 
        'exclude_objects': None
    }
    test_args = lambda train_set: {'causal_vars': train_set.full_target_names}
    dataset = Causal3DDataset(
        data_folder=data_dir, 
        split='train', 
        single_image=False, 
        triplet=False, 
        seq_len=2,
        **dataset_args
    )
    return (dataset.full_target_names, dataset.target_names_l)


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


