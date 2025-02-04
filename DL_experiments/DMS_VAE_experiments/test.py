from collections import OrderedDict
from argparse import ArgumentParser
import os
import pickle
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
import numpy as np

import torch

from shared_utils.dl_experiments import test_model, print_params, get_nn_baseline
from shared_utils.experiment import set_seed

from shared_utils.plot_utils import make_aggregate_tables, create_latent_plots, make_time_plot

def load_datasets(
    data_dir: str,
    cluster: bool
) -> np.array:
    z_gt_all = np.load(os.path.join(data_dir, "z_gt_final.npy"))
    z_hat_all = np.load(os.path.join(data_dir, "z_hat_final.npy"))

    if cluster:
        from metrics import mean_corr_coef_np
        _, _, assigments = mean_corr_coef_np(z_gt_all, z_hat_all)
        
        print(f"Assignments were:", assigments[1])
        z_hat_all = z_hat_all[:, assigments[1]]

        np.save(os.path.join(data_dir, "z_hat_final_ordered.npy"), z_hat_all)
    else:
        z_hat_all = np.load(os.path.join(data_dir, "z_hat_final_ordered.npy"))
    
    return z_gt_all, z_hat_all


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


def do_full_test(model_args, base_dir, Ns, alphas, cluster, do_baseline, do_estimator):
    data_dir = get_data_dir(model_args, base_dir)

    z_gt, z_hat = load_datasets(data_dir=data_dir, cluster=cluster)

    if do_baseline:
        baseline = get_nn_baseline(
            all_encs=torch.from_numpy(z_hat), 
            all_latents=torch.from_numpy(z_gt), 
            Ns=Ns, 
            groups_gt=None, 
            repeats=50
        )
    else:
        baseline = None

    if do_estimator:
        result = test_model(
            all_encs=z_hat,
            all_latents=z_gt,
            alphas=alphas,
            Ns=Ns,
            ckpt_dir=data_dir, 
            groups=None, 
            repeats=50
        )
    else:
        result = None

    create_latent_plots(
        data_dir, 
        N=1000,
        alpha=0.01, 
        test_encs=z_hat, 
        test_latents=z_gt
    )

    return baseline, result


if __name__ == "__main__":
    model_data_combs = [
        {"model_type": "DMS-VAE", "dataset": "action"},
        {"model_type": "DMS-VAE", "dataset": "temporal"},
        {"model_type": "iVAE", "dataset": "action"},
        {"model_type": "TCVAE", "dataset": "temporal"},
    ]

    alphas = [
            0.0001, 0.0005, 
            0.001, 0.005, 
            0.01, 0.05, 0.1, 0.2
        ]
    Ns = [5, 10, 20, 100, 1000, 10000]

    parser = ArgumentParser()
    parser.add_argument('--cluster', action='store_true')
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--estimator', action='store_true')
    args = parser.parse_args()
    model_args = vars(args)

    do_baseline = args.baseline
    do_estimator = args.estimator


    if args.cluster:
        base_dir = "checkpoints"

        baseline_all_results = {}
        all_results = {}
        for model_data_comb in model_data_combs:
            model_args["model_type"] = model_data_comb["model_type"]
            model_args["dataset"] = model_data_comb["dataset"]
            print_params("Permutation estimation experiment", model_args)

            baseline, result = do_full_test(model_args, base_dir, Ns, alphas, True, do_baseline, do_estimator)

            comb_str = f"{model_args['model_type']}/{model_args['dataset']}"

            baseline_all_results[comb_str] = baseline
            all_results[comb_str] = result
        
        if do_estimator:
            with open(os.path.join(base_dir, "all_results.pickle"), "wb") as f:
                pickle.dump(all_results, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(os.path.join(base_dir, "all_results.pickle"), "rb") as f:
                all_results = pickle.load(f)

        if do_baseline:
            with open(os.path.join(base_dir, "baseline_results.pickle"), "wb") as f:
                pickle.dump(baseline_all_results, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(os.path.join(base_dir, "baseline_results.pickle"), "rb") as f:
                baseline_all_results = pickle.load(f)

        # Summarizing all the results
        make_aggregate_tables(
            all_results, 
            baseline_all_results,
            Ns,
            alphas,
            ckpt_dir=base_dir
        )
        make_time_plot(
            all_results, 
            baseline_all_results,
            Ns,
            alphas,
            ckpt_dir=base_dir
        )

    else:
        base_dir = "cluster_checkpoints"
        model_args["models"] = [
            (model_data_comb["model_type"], model_data_comb["dataset"]) 
            for model_data_comb in model_data_combs
        ]
        
        print_params("Permutation aggregate results", model_args)

        with open(os.path.join(base_dir, "all_results.pickle"), "rb") as f:
            all_results = pickle.load(f)
        with open(os.path.join(base_dir, "baseline_results.pickle"), "rb") as f:
            baseline_all_results = pickle.load(f)


        ordered_all_results = {
            "DMS-VAE/action": all_results["DMS-VAE/action"],
            "iVAE/action": all_results["iVAE/action"],
            "DMS-VAE/temporal": all_results["DMS-VAE/temporal"],
            "TCVAE/temporal": all_results["TCVAE/temporal"],
        }

        make_aggregate_tables(
            ordered_all_results, 
            baseline_all_results,
            Ns,
            alphas,
            ckpt_dir=base_dir
        )
        make_time_plot(
            all_results, 
            baseline_all_results,
            Ns,
            alphas,
            ckpt_dir=base_dir
        )

        for model_args in model_data_combs:
            print(f"Creating plots for {model_args['model_type']}")
            data_dir = get_data_dir(model_args, base_dir)
            z_gt, z_hat = load_datasets(data_dir=data_dir, cluster=False)
            create_latent_plots(
                data_dir, 
                N=10000,
                alpha=0.01, 
                test_encs=z_hat, 
                test_latents=z_gt
            )
