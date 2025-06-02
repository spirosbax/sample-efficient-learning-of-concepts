from tqdm import tqdm
from itertools import product
import os
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from experiment.rff_settings import (
    dim_params, 
    reg_params, 
    entanglement_params, 
    n_total_params
)
from utils.plot_utils import create_synthetic_fig_bin

    
if __name__ == "__main__":
    y_axes = [
        'perm_error_match',
        'acc_label',
        'roc_label',
        'acc_concept',
        'roc_concept',
        'ois_concept', 
        'nis_concept',
        'time'
    ]

    START_DATE = '2025-05-01'
    END_DATE = '2025-05-30'

    # x axis d variables
    print("working in dimension plots")
    for a, e, n in tqdm(dim_params):
        dim_settings = {
            "alpha": a,
            "entanglement": e, 
            "n_total": n
        }
        for y in y_axes:
            create_synthetic_fig_bin(
                x_axis="d_variables", 
                y_axis=y, 
                features=["p_features"],
                settings=dim_settings,
                start_date=START_DATE,
                end_date=END_DATE,
                experiment='features'
            )

    # x axis alphas
    print("working in alpha plots")
    for d, e, n in tqdm(reg_params):
        alpha_settings = {
            "d_variables": d,
            "entanglement": e, 
            "n_total": n
        }
        for y in y_axes:
            create_synthetic_fig_bin(
                x_axis="alpha", 
                y_axis=y, 
                features=["p_features"],
                settings=alpha_settings,
                start_date=START_DATE,
                end_date=END_DATE,
                experiment='features'
            )

    # x axis entanglement
    print("working in entanglement plots")
    for d, a, n in tqdm(entanglement_params):
        entanglement_settings = {
            "d_variables": d,
            "alpha": a, 
            "n_total": n
        }
        for y in y_axes:
            create_synthetic_fig_bin(
                x_axis="entanglement", 
                y_axis=y, 
                features=["p_features"],
                settings=entanglement_settings,
                start_date=START_DATE,
                end_date=END_DATE,
                experiment='features'
            )


    # x axis nr of data points
    print("working in nr datapoint plots")
    for d, a, e in tqdm(n_total_params):
        n_total_settings = {
            "d_variables": d,
            "alpha": a, 
            "entanglement": e,
        }
        for y in y_axes:
            create_synthetic_fig_bin(
                x_axis="n_total", 
                y_axis=y, 
                features=["p_features"],
                settings=n_total_settings,
                start_date=START_DATE,
                end_date=END_DATE,
                experiment='features'
            ) 
