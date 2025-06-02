import os
import numpy as np
from data.synthetic import ActionToyManifoldDataset, TemporalToyManifoldDataset


base_dir = "checkpoints"
action = "action_sparsity_non_trivial"
temporal = "temporal_sparsity_non_trivial"
manifold = "nn"
model_data_combs = [
    # {"model_type": "DMS-VAE", "dataset": "action"},
    # {"model_type": "DMS-VAE", "dataset": "temporal"},
    {"model_type": "iVAE", "dataset": "action"},
    {"model_type": "TCVAE", "dataset": "temporal"},
]

model = "DMS_VAE"
data_dir = os.path.join(base_dir, "DMS_VAE", action)
z_gt_all = np.load(os.path.join(data_dir, "z_gt_final.npy"))
decoder = ActionToyManifoldDataset(
    manifold=manifold,
    transition_model=action,
    num_samples=int(1e6), 
    seed=100,
    x_dim=int(20),
    z_dim=int(10),
    c_dim=int(10)
)
x_all = decoder.decoder(z_gt_all)
np.save(os.path.join(data_dir, "x_all.npy"), x_all)

model = "DMS_VAE"
data_dir = os.path.join(base_dir, "DMS_VAE", temporal)
z_gt_all = np.load(os.path.join(data_dir, "z_gt_final.npy"))
decoder = TemporalToyManifoldDataset(
    manifold=manifold,
    transition_model=temporal,
    num_samples=int(1e6), 
    seed=100,
    x_dim=int(20),
    z_dim=int(10)
)
x_all = decoder.decoder(z_gt_all)
np.save(os.path.join(data_dir, "x_all.npy"), x_all)


