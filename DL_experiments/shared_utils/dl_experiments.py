
from collections import OrderedDict
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], '../..'))

import time
import torch
import torch.utils.data as data
import torch.nn.functional as F
from tqdm.auto import tqdm
import numpy as np
import scipy as sp

import matplotlib.pyplot as plt

from sklearn.preprocessing import SplineTransformer
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics import r2_score, mean_squared_error

from shared_utils.experiment import calc_perm_errors, set_seed
from shared_utils.causal_encoder import  MLP
from permutation_estimator.estimator import FeaturePermutationEstimator, KernelizedPermutationEstimator


def print_params(logger_name, model_args):
    num_chars = max(50, 11+len(logger_name))
    print('=' * num_chars)
    print(f'Experiment {logger_name}')
    print('-' * num_chars)
    for key in sorted(list(model_args.keys())):
        print(f'-> {key}: {model_args[key]}')
    print('=' * num_chars)


def get_device():
    try:
        if torch.backends.mps.is_available():
            # For small networks and loads this cpu seems to be faster 
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    except AttributeError as e:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    return device


def get_nn_baseline(all_encs, all_latents, Ns, groups_gt, repeats=10):

    errors = np.zeros(shape=(len(Ns), repeats))
    mse = np.zeros(shape=(len(Ns), repeats))
    r2 = np.zeros(shape=(len(Ns), repeats))
    times = np.zeros(shape=(len(Ns), repeats))

    seeds = list(range(40, 40 + repeats))

    nr_latents = all_latents.shape[1]
    data_size = all_encs.shape[0]

    for i, N in enumerate(Ns):
        print(f"For baseline NN working on {N}")
        for j in range(repeats):
            rand_perm = np.random.choice(size=nr_latents, a=np.arange(nr_latents),replace=False)
            perm_all_latents = all_latents[:, rand_perm]
            
            # Train network to predict causal factors from latent variables
            train_time, r2_matrix, mse_matrix = train_test_network(
                all_encs,
                perm_all_latents, 
                data_size, 
                N, 
                seeds[j],
                nr_latents,
                groups_gt=groups_gt,
                num_train_epochs=100,
                c_hid=128
            )
            
            groups = sp.optimize.linear_sum_assignment(-1 * r2_matrix.T)[1]
            
            cols = np.arange(nr_latents)

            times[i, j] = train_time
            errors[i, j] = calc_perm_errors(groups, rand_perm)

            mse[i, j] = mse_matrix[groups, cols].mean()
            r2[i, j] = r2_matrix[groups, cols].mean() 

            print("error", errors[i, j])
            print("mse", mse[i, j])
            print("r2", r2[i, j])


    result = {
        "times": times,
        "perm_error": errors,
        "mse": mse, 
        "r2": r2
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
    groups=None,
    repeats: int=10
) -> np.array:
    n = encs.shape[0]

    errors_match = np.zeros(shape=repeats)
    errors_spear = np.zeros(shape=repeats)
    errors_corr = np.zeros(shape=repeats)
    mse_arr = np.zeros(shape=repeats)
    r2_arr = np.zeros(shape=repeats)
    times_match = np.zeros(shape=repeats)
    times_corr = np.zeros(shape=repeats)
    times_spear = np.zeros(shape=repeats)

    nr_latents = latents.shape[1]
    z_dim = encs.shape[1]
    seeds = list(range(40, 40 + repeats))
    
    for i in range(repeats):
        set_seed(seeds[i])

        rand_perm = np.random.choice(size=nr_latents, a=np.arange(nr_latents),replace=False)
        indices = np.random.choice(size=n, a=np.arange(n),replace=False)

        permuted_latents = latents[:, rand_perm]

        # We use as many test data points as train data points
        train_idx, test_idx = indices[:N], indices[-N:]

        train_encs = encs[train_idx, :]
        train_latents = permuted_latents[train_idx, :]
        test_encs = encs[test_idx, :]
        test_latents = permuted_latents[test_idx, :]

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

        if groups is None:
            res = estimator.fit(
                train_encs.T, train_latents.T
            )
        else:
            res = estimator.fit(
                train_encs.T, train_latents.T,
                recover_weights=False
            )
        test_latents_hat = estimator.predict_match(test_encs.T)

        mse_dims = np.zeros(nr_latents)
        r2_dims = np.zeros(nr_latents)

        mse_dims = mean_squared_error(test_latents, test_latents_hat.T, multioutput="raw_values")
        r2_dims = r2_score(test_latents, test_latents_hat.T, multioutput="raw_values")

        mse_arr[i] = mse_dims.mean()
        r2_arr[i] = r2_dims.mean()

        errors_match[i] = calc_perm_errors(rand_perm, res["perm_hat_match"])
        errors_spear[i] = calc_perm_errors(rand_perm, res["perm_hat_spr"])
        errors_corr[i] = calc_perm_errors(rand_perm, res["perm_hat_corr"])

        times_match[i] = res["time_match"]
        times_spear[i] = res["time_spear"]      
        times_corr[i] = res["time_corr"]

    results = {
        "error_match": errors_match,
        "error_spear": errors_spear, 
        "error_corr": errors_corr, 
        "mse": mse_arr,
        "r2_score": r2_arr, 
        "times_match": times_match,
        "times_spear": times_spear,
        "times_corr": times_corr,
    }

    return results


def test_model(
    all_latents: np.array,
    all_encs: np.array,
    alphas, 
    Ns,
    ckpt_dir: str,
    groups=None,
    repeats=10
) -> np.array:
    z_dim = all_encs.shape[1]
    
    n_knots = 6
    degree = 3
    n_features = n_knots + degree - 1 

    splines_all_dims = [SplineTransformer(
        n_knots=n_knots,
        degree=degree) for _ in range(z_dim)]
    rff_all_dims = [RBFSampler(n_components=n_features) for _ in range(z_dim)]

    perm_errors = np.zeros((7, repeats, len(alphas), len(Ns)))
    mse = np.zeros((5, repeats, len(alphas), len(Ns)))
    r2 = np.zeros((5, repeats, len(alphas), len(Ns)))
    times = np.zeros((5, repeats, len(alphas), len(Ns)))

    overlapping_settings = {
        "latents": all_latents, 
        "encs": all_encs,
        "repeats": repeats
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
            print(results_two_stage["r2_score"])
            print(results_two_stage["mse"])

            perm_errors[0, :, i, j] = results_linear["error_match"]
            perm_errors[1, :, i, j] = results_spline["error_match"]
            perm_errors[2, :, i, j] = results_rff["error_match"]
            perm_errors[3, :, i, j] = results_laplacian["error_match"]
            perm_errors[4, :, i, j] = results_two_stage["error_match"]
            perm_errors[5, :, i, j] = results_linear["error_corr"]
            perm_errors[6, :, i, j] = results_linear["error_spear"]

            mse[0, :, i, j] = results_linear["mse"]
            mse[1, :, i, j] = results_spline["mse"]
            mse[2, :, i, j] = results_rff["mse"]
            mse[3, :, i, j] = results_laplacian["mse"]
            mse[4, :, i, j] = results_two_stage["mse"]

            r2[0, :, i, j] = results_linear["r2_score"]
            r2[1, :, i, j] = results_spline["r2_score"]
            r2[2, :, i, j] = results_rff["r2_score"]
            r2[3, :, i, j] = results_laplacian["r2_score"]
            r2[4, :, i, j] = results_two_stage["r2_score"]

            times[0, :, i, j] = results_linear["times_match"]
            times[1, :, i, j] = results_spline["times_match"]
            times[2, :, i, j] = results_rff["times_match"]
            times[3, :, i, j] = results_laplacian["times_match"]
            times[4, :, i, j] = results_two_stage["times_match"]

    print("perm Linear: ", perm_errors[0, 0, :, :])
    print("perm Spline: ", perm_errors[1, 0, :, :])
    print("perm rff: ", perm_errors[2, 0, :, :])
    print("perm Laplacian: ", perm_errors[3, 0, :, :])
    print("perm Two Stage: ", perm_errors[6, 0, :, :])

    print("r2 Linear: ", r2[0, 0, :, :])
    print("r2 Spline: ", r2[1, 0, :, :])
    print("r2 rff: ", r2[2, 0, :, :])
    print("r2 Laplacian: ", r2[3, 0, :, :])
    print("r2 Two stage: ", r2[4, 0, :, :])

    results = {
        "perm_error": perm_errors,
        "mse": mse,
        "r2": r2,
        "times": times,
    }   

    return results


def train_test_network(
        all_encs, 
        perm_all_latents, 
        data_size,
        N,
        seed,
        nr_latents,
        groups_gt = None,
        num_train_epochs: int=100, 
        c_hid: int=128
) -> np.array:
        start = time.time()
        device = get_device()

        # In the CITRIS experiments, use all the encodings in the same group for the predictions
        individual_encs = []
        encoders = []
        if groups_gt is not None:
            perm_all_latents = torch.from_numpy(perm_all_latents).to(torch.float32)
            grouped_encs = torch.zeros(size=(data_size, nr_latents))
            for g in range(nr_latents):
                g_idx = groups_gt == g
                grouped_encs = all_encs[:, g_idx].to(torch.float32)
                
                encoders.append(MLP(c_in=grouped_encs.size(1), c_hid=c_hid, c_out=nr_latents, lr=4e-3))
                individual_encs.append(grouped_encs)
        else:
            for g in range(nr_latents):
                encoders.append(MLP(c_in=1, c_hid=c_hid, c_out=nr_latents, lr=4e-3)) 
                individual_encs.append(all_encs[:, g:g+1].to(torch.float32))


        r2_matrix = np.zeros(shape=(nr_latents, nr_latents))
        mse_matrix = np.zeros(shape=(nr_latents, nr_latents))
        for i, enc in enumerate(encoders):
            # Create new tensor dataset for training (50%) and testing (50%)
            full_dataset = data.TensorDataset(individual_encs[i], perm_all_latents)

            train_dataset, test_dataset = data.random_split(
                full_dataset,
                lengths=[int(data_size / 2), int(data_size / 2)], 
                generator=torch.Generator().manual_seed(seed)
            )
            train_dataset = data.Subset(train_dataset, np.arange(N))
            test_dataset = data.Subset(test_dataset, np.arange(N))

            train_loader = data.DataLoader(
                train_dataset, 
                shuffle=True, 
                drop_last=False, 
                batch_size=512
            )

            optimizer = enc.configure_optimizers()
            enc.to(device)
            enc.train()
            for epoch in range(num_train_epochs):
                avg_loss = 0.0
                for inps, latents in train_loader:
                    inps = inps.to(device)
                    latents = latents.to(device)
                    loss = enc.training_step((inps, latents), 1)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    avg_loss += loss.item()
            test_encs = individual_encs[i][test_dataset.indices, :].to(device)
            test_latents = perm_all_latents[test_dataset.indices]

            enc.eval()

            test_latents_hat = enc.forward(test_encs)

            y = test_latents.detach().cpu().numpy()
            y_hat = test_latents_hat.detach().cpu().numpy()

            mse_matrix[i, :] = mean_squared_error(y, y_hat, multioutput="raw_values") 
            r2_matrix[i, :] = r2_score(y, y_hat, multioutput="raw_values") 

        end = time.time()
        return end - start, r2_matrix, mse_matrix


def train_predict_network(
        all_encs, 
        all_latents, 
        num_train_epochs: int=100
    ) -> np.array:
        device = get_device()
        nr_latents = all_encs.shape[1]
        encoders = [MLP(c_in=1, c_hid=128, c_out=nr_latents, lr=4e-3) for _ in range(nr_latents)]

        r2_matrix = np.zeros(shape=(nr_latents, nr_latents))
        for i, enc in enumerate(encoders):
            dataset = data.TensorDataset(
                torch.from_numpy(all_encs[:, i:i+1]), 
                torch.from_numpy(all_latents)
            )
            train_loader = data.DataLoader(
                        dataset, 
                        shuffle=True, 
                        drop_last=False, 
                        batch_size=512
                    )

            optimizer = enc.configure_optimizers()
            enc.to(device)
            enc.train()
            for epoch in range(num_train_epochs):
                avg_loss = 0.0
                for inps, latents in train_loader:
                    inps = inps.to(device)
                    latents = latents.to(device)
                    loss = enc.training_step((inps, latents), 1)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    avg_loss += loss.item()
            enc.eval()

            test_latents_hat = enc.forward(torch.from_numpy(all_encs[:, i:i+1]).to(device))
            y_hat = test_latents_hat.detach().cpu().numpy()

            r2_matrix[i, :] = r2_score(all_latents, y_hat, multioutput="raw_values") 

        
        assignment = sp.optimize.linear_sum_assignment(-1 * r2_matrix.T)[1]

        latent_predictions = np.zeros(shape=all_latents.shape)

        for i in range(nr_latents):
            all_encs_ind = torch.from_numpy(all_encs[:, i:i+1]).to(device)
            y_hat = encoders[i].forward(all_encs_ind)[:, assignment[i]].detach().cpu().numpy()
            latent_predictions[:, assignment[i]] = y_hat
        

        return latent_predictions
