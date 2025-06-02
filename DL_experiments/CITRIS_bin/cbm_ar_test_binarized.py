"""
Similar as the other test.py script, but the latents are now binarized by 
threshholding in the middle of the values. 

The metrics are taken from Towards Robust Metrics For Concept Representation Evaluation [Zarlenga et al. 2023]

"""

from argparse import ArgumentParser
import os
import pickle
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
import numpy as np

from training.utils import print_params
from DL_experiments.shared_utils.dl_experiments_binarized import (
    find_best_model_and_latents,
    do_cbm_test, 
    set_seed,
    create_labels
)

from models.citris.citris_vae import CITRISVAE
from models.citris.baselines.ivae import iVAE


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


def get_data(model_args, base_dir):
    if model_args["model_type"] == "iVAE":
        model_dir = os.path.join(base_dir, 'iVAE/iVAE_32l_7b_32hid_causal3d')
        ModelClass = iVAE
    elif model_args["model_type"] == "CITRISVAE":    
        model_dir = os.path.join(base_dir, 'CITRISVAE/CITRISVAE_32l_7b_32hid_causal3d')
        ModelClass = CITRISVAE
    ckpt_dir, model, all_encs, all_latents, arg_names = find_best_model_and_latents(
        model_dir=model_dir,
        model_class=ModelClass
    )

    return ckpt_dir, model, all_encs, all_latents, arg_names



if __name__ == "__main__":
    models_cbm = [
        {"model_type": "HardCBM"},
    ]
    repeats = 10
    seeds = list(range(100, 100 + repeats))
    Ns = [20, 100, 1000, 10000]

    parser = ArgumentParser()
    parser.add_argument('--cluster', action='store_true')

    args = parser.parse_args()
    model_args = vars(args)

    if args.cluster:
        base_dir = "checkpoints"
        # DO CBM based concept learning
        model_args["model_type"] = "iVAE"
        ckpt_dir, model, all_encs, all_latents, args_names = get_data(model_args, base_dir)

        # We use the coarse latents, so combine the needed columns
        all_latents_1 = all_latents[:, :3].sum(axis=-1)
        all_latents_2 = all_latents[:, 3:5].sum(axis=-1)
        all_latents_3 = all_latents[:, 5:]

        all_latents = np.c_[all_latents_1, all_latents_2, all_latents_3]

        bin_all_latents = binarize_latents(all_latents)

        for model in models_cbm:
            model_args["model_type"] = model["model_type"]
            print_params("Permutation Concepts experiment", model_args)


            for i, seed in enumerate(seeds):
                set_seed(seed)

                bin_all_latents = binarize_latents(all_latents)
                y_values = create_labels(bin_all_latents)

                results_cbm = do_cbm_test(
                    None,
                    bin_all_latents, 
                    y_values, 
                    Ns, 
                    seed, 
                    model["model_type"],
                    base_dir, 
                    args.cluster
                )
                
    
    

