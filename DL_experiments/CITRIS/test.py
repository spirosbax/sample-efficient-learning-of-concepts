from argparse import ArgumentParser
import os
import pickle
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
import numpy as np

from training.utils import print_params
from utils.dl_experiments import (
    get_groups,
    find_best_model_and_latents
)

from shared_utils.dl_experiments import test_model, get_nn_baseline
from shared_utils.plot_utils import make_aggregate_tables, make_time_plot

from models.citris.citris_vae import CITRISVAE
from models.citris.baselines.ivae import iVAE 


def get_data(model_args, base_dir):
    if model_args["model_type"] == "iVAE":
        model_dir = os.path.join(base_dir, 'iVAE/iVAE_32l_7b_32hid_causal3d')
        ModelClass = iVAE
    elif model_args["model_type"] == "CITRISVAE":    
        model_dir = os.path.join(base_dir, 'CITRISVAE/CITRISVAE_32l_7b_32hid_causal3d')
        ModelClass = CITRISVAE
    ckpt_dir, model, all_encs, all_latents = find_best_model_and_latents(
        model_dir=model_dir,
        model_class=ModelClass
    )

    return ckpt_dir, model, all_encs, all_latents


def do_full_test(model_args, base_dir,  Ns, alphas, do_baseline, do_estimator):
    ckpt_dir, model, all_encs, all_latents = get_data(model_args, base_dir)

    groups = get_groups(
        model=model, 
        all_encs=all_encs, 
        all_latents=all_latents
    )

    all_latents_1 = all_latents[:, :3].sum(axis=-1)
    all_latents_2 = all_latents[:, 3:5].sum(axis=-1)
    all_latents_3 = all_latents[:, 5:]

    if model.__class__.__name__ == "CITRISVAE":
        all_latents_4 = np.random.normal(size=all_latents.shape[0])
        all_latents = np.c_[all_latents_1, all_latents_2, all_latents_3, all_latents_4]
    elif model.__class__.__name__ == "iVAE":
        all_latents = np.c_[all_latents_1, all_latents_2, all_latents_3]

    if do_baseline:
        baseline = get_nn_baseline(
            all_encs=all_encs, 
            all_latents=all_latents, 
            Ns=Ns, 
            groups_gt=groups, 
            repeats=50
        )
    else:
        baseline = None

    if do_estimator:
        result = test_model(
            all_encs=all_encs,
            all_latents=all_latents,
            alphas=alphas,
            Ns=Ns,
            ckpt_dir=ckpt_dir, 
            groups=groups,
            repeats=50
        )
    else:
        result = None
    
    return baseline, result


if __name__ == "__main__":    
    models = [
        {"model_type": "CITRISVAE"},
        {"model_type": "iVAE"},
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
        for model in models:
            model_args["model_type"] = model["model_type"]
            print_params("Permutation estimation experiment", model_args)

            baseline, result = do_full_test(model_args, base_dir, Ns, alphas, do_baseline, do_estimator)

            baseline_all_results[model["model_type"]] = baseline
            all_results[model["model_type"]] = result 
        
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
        model_args["models"] = [model["model_type"] for model in models]

        print_params("Permutation aggregate results", model_args)

        with open(os.path.join(base_dir, "all_results.pickle"), "rb") as f:
            all_results = pickle.load(f)
        with open(os.path.join(base_dir, "baseline_results.pickle"), "rb") as f:
            baseline_all_results = pickle.load(f)

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

        

