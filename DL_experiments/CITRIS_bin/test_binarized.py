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
    get_groups,
    find_best_model_and_latents,
    do_vae_test, 
    do_cbm_test, 
    set_seed
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


def create_labels(binary_latents, seed):
    set_seed(seed)
    num_latents = binary_latents.shape[1]

    options = np.arange(num_latents)
    active_concepts = np.random.choice(options, size=3)
    sum_active = binary_latents[:, active_concepts].sum(axis=1)
    labels = np.zeros(binary_latents.shape[0])
    labels[sum_active > 1] = 1
    return labels


if __name__ == "__main__":
    models = [
        {"model_type": "CITRISVAE"},
        {"model_type": "iVAE"},
    ]
    models_cbm = [
        {"model_type": "CBM"},
        {"model_type": "CEM"},
    ]
    repeats = 10
    seeds = list(range(100, 100 + repeats))

    alphas = [
            0.0001, 0.0005, 
            0.001, 0.005, 
            0.01, 0.05, 0.1, 0.2 
        ]
    Ns = [20, 100, 1000, 10000]

    parser = ArgumentParser()
    parser.add_argument('--cluster', action='store_true')

    args = parser.parse_args()
    model_args = vars(args)

    # do_vae = args.vae
    # do_cbm = args.cbm

    if args.cluster:
        base_dir = "checkpoints"
        all_results = {}
        for model in models:
            model_args["model_type"] = model["model_type"]
            print_params("Permutation Concepts experiment", model_args)


            acc_label_all = np.zeros((repeats, 5, len(alphas), len(Ns)))
            roc_label_all = np.zeros((repeats, 5, len(alphas), len(Ns)))

            acc_concept_all = np.zeros((repeats, 5, len(alphas), len(Ns)))
            roc_concept_all = np.zeros((repeats, 5, len(alphas), len(Ns)))

            ois_concept_all = np.zeros((repeats, 5, len(alphas), len(Ns)))
            nis_concept_all = np.zeros((repeats, 5, len(alphas), len(Ns)))

            times_all = np.zeros((repeats, 5, len(alphas), len(Ns)))

            for i, seed in enumerate(seeds):
                set_seed(seed)

                ckpt_dir, model, all_encs, all_latents, args_names = get_data(model_args, base_dir)
                groups = get_groups(
                    model=model, 
                    all_encs=all_encs, 
                    all_latents=all_latents
                )
                # We use the coarse latents, so combine the needed columns
                all_latents_1 = all_latents[:, :3].sum(axis=-1)
                all_latents_2 = all_latents[:, 3:5].sum(axis=-1)
                all_latents_3 = all_latents[:, 5:]

                if model.__class__.__name__ == "CITRISVAE":
                    all_latents_4 = np.random.normal(size=all_latents.shape[0])
                    all_latents = np.c_[all_latents_1, all_latents_2, all_latents_3, all_latents_4]
                elif model.__class__.__name__ == "iVAE":
                    all_latents = np.c_[all_latents_1, all_latents_2, all_latents_3]

                bin_all_latents = binarize_latents(all_latents)
                y_values = create_labels(bin_all_latents, seed)

                # DO VAE based concept learning
                results_vae = do_vae_test(
                    bin_all_latents,
                    all_encs, 
                    y_values,
                    alphas, 
                    Ns, 
                    groups
                )

                acc_label_all[i, :, :, :] = results_vae["acc_label"]
                roc_label_all[i, :, :, :] = results_vae["roc_label"]

                acc_concept_all[i, :, :, :] = results_vae["acc_concept"]
                roc_concept_all[i, :, :, :] = results_vae["roc_concept"]

                ois_concept_all[i, :, :, :] = results_vae["ois_concept"]
                nis_concept_all[i, :, :, :] = results_vae["nis_concept"]

                times_all[i, :, :, :] = results_vae["times"]
          
            all_results[model_args["model_type"]] = {
                "acc_label": acc_label_all, 
                "roc_label": roc_label_all, 
                "acc_concept": acc_concept_all, 
                "roc_concept": roc_concept_all, 
                "ois_concept": ois_concept_all,
                "nis_concept": nis_concept_all,
                "times": times_all
            }
        
        with open(os.path.join(base_dir, "vae_results_binarized.pickle"), "wb") as f:
            pickle.dump(all_results, f, protocol=pickle.HIGHEST_PROTOCOL)

        # DO CBM based concept learning
        all_results_cbm = {}
        
        model_args["model_type"] = "iVAE"
        ckpt_dir, model, all_encs, all_latents, args_names = get_data(model_args, base_dir)
        # We use the coarse latents, so combine the needed columns
        all_latents_1 = all_latents[:, :3].sum(axis=-1)
        all_latents_2 = all_latents[:, 3:5].sum(axis=-1)
        all_latents_3 = all_latents[:, 5:]

        if model.__class__.__name__ == "CITRISVAE":
            all_latents_4 = np.random.normal(size=all_latents.shape[0])
            all_latents = np.c_[all_latents_1, all_latents_2, all_latents_3, all_latents_4]
        elif model.__class__.__name__ == "iVAE":
            all_latents = np.c_[all_latents_1, all_latents_2, all_latents_3]

        bin_all_latents = binarize_latents(all_latents)

        for model in models_cbm:
            model_args["model_type"] = model["model_type"]
            print_params("Permutation Concepts experiment", model_args)

            acc_label_all_cbm = np.zeros((repeats, len(Ns)))
            roc_label_all_cbm = np.zeros((repeats, len(Ns)))

            acc_concept_all_cbm = np.zeros((repeats, len(Ns)))
            roc_concept_all_cbm = np.zeros((repeats, len(Ns)))

            ois_concept_all_cbm = np.zeros((repeats, len(Ns)))
            nis_concept_all_cbm = np.zeros((repeats, len(Ns)))

            times_all_cbm = np.zeros((repeats, len(Ns)))

            for i, seed in enumerate(seeds):
                set_seed(seed)
                y_values = create_labels(bin_all_latents, seed)

                results_cbm = do_cbm_test(
                    bin_all_latents, 
                    y_values, 
                    Ns, 
                    seed, 
                    model["model_type"],
                    base_dir, 
                    args.cluster
                )
                print(results_cbm)

                acc_label_all_cbm[i, :] = results_cbm["acc_label"]
                roc_label_all_cbm[i, :] = results_cbm["roc_label"]

                acc_concept_all_cbm[i, :] = results_cbm["acc_concept"]
                roc_concept_all_cbm[i, :] = results_cbm["roc_concept"]

                ois_concept_all_cbm[i, :] = results_cbm["ois_concept"]
                nis_concept_all_cbm[i, :] = results_cbm["nis_concept"]

                times_all_cbm[i, :] = results_cbm["times"]

            all_results_cbm[model["model_type"]] = {
                "acc_label": acc_label_all_cbm, 
                "roc_label": roc_label_all_cbm, 
                "acc_concept": acc_concept_all_cbm, 
                "roc_concept": roc_concept_all_cbm, 
                "ois_concept": ois_concept_all_cbm,
                "nis_concept": nis_concept_all_cbm,
                "times": times_all_cbm
            }

        with open(os.path.join(base_dir, "cbm_results_binarized.pickle"), "wb") as f:
            pickle.dump(all_results_cbm, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    

