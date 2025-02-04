import os
import random
import torch
import numpy as np

from typing import List

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def calc_perm_errors(perm, perm_hat):
    return np.sum(perm != perm_hat) / len(perm)
     

def save_error_data(ckpt_dir: str, perm_error:np.array, y_mse: np.array, features: List[str], suffix: str):
    data_dir = os.path.join(ckpt_dir, "data")

    for i, feature in enumerate(features):
        np.savetxt(os.path.join(data_dir, f"{feature}_perm_error_{suffix}"), perm_error[:, i], fmt='%.6f')
        np.savetxt(os.path.join(data_dir, f"{feature}_y_mse_{suffix}"), y_mse[:, i], fmt='%.6f')
