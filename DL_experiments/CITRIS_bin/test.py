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
    find_best_model_and_latents,
    do_vae_test, 
    do_cbm_test, 
    set_seed
)

from models.citris.citris_vae import CITRISVAE
from models.citris.baselines.ivae import iVAE



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


def create_labels(latents, seed):
    set_seed(seed)

    active_concepts = np.random.randint(1, 5)
    random_weights = 10 * (np.random.random(active_concepts) - 0.5)

    random_weights = np.pad(
        random_weights, 
        pad_width=(0, (latents.shape[1] - active_concepts)),
        mode="constant"
    )
    np.random.shuffle(random_weights)
    concept_labels = np.dot(latents, random_weights)
    return concept_labels


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


            mse_label_all = np.zeros((repeats, 5, len(alphas), len(Ns)))
            r2_label_all = np.zeros((repeats, 5, len(alphas), len(Ns)))

            mse_concept_all = np.zeros((repeats, 5, len(alphas), len(Ns)))
            r2_concept_all = np.zeros((repeats, 5, len(alphas), len(Ns)))

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

                y_values = create_labels(all_latents, seed)

                # DO VAE based concept learning
                results_vae = do_vae_test(
                    all_latents,
                    all_encs, 
                    y_values,
                    alphas, 
                    Ns, 
                    groups
                )

                mse_label_all[i, :, :, :] = results_vae["mse_label"]
                r2_label_all[i, :, :, :] = results_vae["r2_label"]

                mse_concept_all[i, :, :, :] = results_vae["mse_concept"]
                r2_concept_all[i, :, :, :] = results_vae["r2_concept"]

                times_all[i, :, :, :] = results_vae["times"]
            
            all_results[model_args["model_type"]] = {
                "mse_label": mse_label_all, 
                "r2_label": r2_label_all, 
                "mse_concept": mse_concept_all, 
                "r2_concept": r2_concept_all, 
                "times": times_all
            }
        
        with open(os.path.join(base_dir, "vae_results.pickle"), "wb") as f:
            pickle.dump(all_results, f, protocol=pickle.HIGHEST_PROTOCOL)

        # DO CBM based concept learning
        all_results_cbm = {}
        for model in models_cbm:
            model_args["model_type"] = model["model_type"]
            print_params("Permutation Concepts experiment", model_args)

            mse_label_all_cbm = np.zeros((repeats, len(Ns)))
            r2_label_all_cbm = np.zeros((repeats, len(Ns)))

            mse_concept_all_cbm = np.zeros((repeats, len(Ns)))
            r2_concept_all_cbm = np.zeros((repeats, len(Ns)))

            times_all_cbm = np.zeros((repeats, len(Ns)))

            for i, seed in enumerate(seeds):
                set_seed(seed)
                y_values = create_labels(all_latents, seed)

                results_cbm = do_cbm_test(
                    all_latents, 
                    y_values, 
                    Ns, 
                    seed, 
                    model["model_type"],
                    base_dir, 
                    args.cluster
                )
                print(results_cbm)

                mse_label_all_cbm[i, :] = results_cbm["mse_label"]
                r2_label_all_cbm[i, :] = results_cbm["r2_label"]

                mse_concept_all_cbm[i, :] = results_cbm["mse_concept"]
                r2_concept_all_cbm[i, :] = results_cbm["r2_concept"]

                times_all_cbm[i, :] = results_cbm["times"]

            all_results_cbm[model["model_type"]] = {
                "mse_label": mse_label_all_cbm, 
                "r2_label": r2_label_all_cbm, 
                "mse_concept": mse_concept_all_cbm, 
                "r2_concept": r2_concept_all_cbm, 
                "times": times_all_cbm
            }

        with open(os.path.join(base_dir, "cbm_results.pickle"), "wb") as f:
            pickle.dump(all_results_cbm, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    

