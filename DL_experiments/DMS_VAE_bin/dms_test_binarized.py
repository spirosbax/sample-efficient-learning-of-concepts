from argparse import ArgumentParser
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
import numpy as np

from shared_utils.dl_experiments_binarized import (
    do_vae_test,
)
from shared_utils.dl_experiments import print_params


def load_datasets(data_dir: str) -> np.array:
    z_gt_all = np.load(os.path.join(data_dir, "z_gt_final.npy"))
    # z_hat_all = np.load(os.path.join(data_dir, "z_hat_final.npy"))
    z_hat_all = np.load(os.path.join(data_dir, "z_hat_final_ordered.npy"))
    
    return z_gt_all, z_hat_all


def binarize_latents(all_latents):
    num_latents = all_latents.shape[1]
    bin_all_latents = np.zeros(all_latents.shape)
    for i in range(num_latents):
        min_val = np.min(all_latents[:, i])
        max_val = np.max(all_latents[:, i])

        mid_point = (max_val + min_val) / 2 
        print('concept ', i)
        print(f"min: {min_val}, max: {max_val}, mid: {mid_point}")
        bin_all_latents[all_latents[:, i] > mid_point, i] = 1

    return bin_all_latents


def get_data_dir(model_args, base_dir):
    if model_args["model_type"] == "iVAE":
        model_dir = os.path.join(base_dir, "iVAE")
    elif model_args["model_type"] == "DMS-VAE":
        model_dir = os.path.join(base_dir, "DMS_VAE")
    elif model_args["model_type"] == "TCVAE":
        model_dir = os.path.join(base_dir, "TCVAE")

    if model_args["dataset"] == "action":
        data_dir = os.path.join(model_dir, "action_sparsity_non_trivial")
    elif model_args["dataset"] == "temporal":
        data_dir = os.path.join(model_dir, "temporal_sparsity_non_trivial")

    return data_dir


if __name__ == "__main__":
    model_data_combs = [
        {"model_type": "DMS-VAE", "dataset": "action"},
        {"model_type": "DMS-VAE", "dataset": "temporal"},
        # {"model_type": "iVAE", "dataset": "action"},
        # {"model_type": "TCVAE", "dataset": "temporal"},
    ]
    repeats = 10
    seeds = list(range(100, 100 + repeats))
    alphas = [
            0.0001, 0.0002, 0.0005, 
            0.001, 0.002, 0.005, 
            0.01, 0.02, 0.05, 
            0.1, 0.2, 0.3, 0.4, 0.5, 1.
        ]
    Ns = [20, 100, 1000, 10000]

    parser = ArgumentParser()
    parser.add_argument('--cluster', action='store_true')

    args = parser.parse_args()
    model_args = vars(args)

    if args.cluster:
        base_dir = "/dev/shm/checkpoints"
    else:
        base_dir = "checkpoints"

    all_results = {}
    for model_data_comb in model_data_combs:
        model_args["model_type"] = model_data_comb["model_type"]
        model_args["dataset"] = model_data_comb["dataset"]
        print_params("Permutation estimation experiment", model_args)

        data_dir = get_data_dir(model_args, base_dir)
        z_gt, z_hat = load_datasets(data_dir=data_dir)

        bin_z_gt = binarize_latents(z_gt)

        do_vae_test(
            bin_z_gt, 
            z_hat,
            alphas,
            Ns, 
            groups=None,
            seeds=seeds,
            model_type=model_args["model_type"], 
            dataset=model_args['dataset']
        )
