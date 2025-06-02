import os
import pickle 
import numpy as np

from shared_utils.plot_utils import make_bin_plots

model_mappings = {
    "CITRISVAE": ("CITRIS-VAE", None),
    "iVAE": ("iVAE", None),
    "DMS-VAE/action": ("DMS-VAE", 'action'),
    "DMS-VAE/temporal": ("DMS-VAE", 'temporal'),
    "iVAE/action": ("iVAE", "action"),
    "TCVAE/temporal": ("TCVAE", 'temporal')
}

concept_dir = "concept_experiments/cluster_checkpoints"

with open(os.path.join(concept_dir, "vae_results_binarized.pickle"), "rb") as f:
    vae_results = pickle.load(f)
with open(os.path.join(concept_dir, "cbm_results_binarized.pickle"), "rb") as f:
    cbm_results = pickle.load(f)

Ns = [20, 100, 1000, 10000]
Ns_idx = [0, 1, 2, 3]

methods = ["Linear", "Spline", "RFF", "Laplacian", "Two Stage"]
methods_idx = [0, 1, 2, 3, 4]

make_bin_plots(
    vae_results, 
    cbm_results, 
    "acc_label", 
    Ns, 
    Ns_idx, 
)
make_bin_plots(
    vae_results, 
    cbm_results, 
    "roc_label", 
    Ns, 
    Ns_idx, 
)

make_bin_plots(
    vae_results, 
    cbm_results, 
    "acc_concept", 
    Ns, 
    Ns_idx, 
)
make_bin_plots(
    vae_results, 
    cbm_results, 
    "roc_concept", 
    Ns, 
    Ns_idx, 
)



    