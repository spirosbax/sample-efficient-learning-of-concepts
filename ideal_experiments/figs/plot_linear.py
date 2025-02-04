from tqdm import tqdm
from itertools import product
import os
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from experiment.linear_settings import (
    dim_params, 
    reg_params, 
    entanglement_params, 
    n_total_params
)
from utils.plot_utils import create_synthetic_fig 

if __name__ == "__main__":
    y_axes = [
        'perm_error_match',
        'mse_match',
        'r2_match',
        'time_match'
    ]
    regularizers = ['group', 'lasso']

    START_DATE = '2025-01-20'
    END_DATE = '2025-01-31'

    # x axis d variables
    print("working in dimension plots")
    for a, e, m, n in tqdm(dim_params):
        dim_settings = {
            "alpha": a,
            "entanglement": e, 
            "miss_well": 'true' if m else 'false',
            "n_total": n
        }
        for y in y_axes:
            create_synthetic_fig(
                x_axis="d_variables", 
                y_axis=y, 
                regularizers=regularizers,
                features=[],
                settings=dim_settings,
                start_date=START_DATE,
                end_date=END_DATE,
                experiment='linear'
            )



    # x axis alphas
    print("working in alpha plots")
    for d, e, m, n in tqdm(reg_params):
        alpha_settings = {
            "d_variables": d,
            "entanglement": e, 
            "miss_well": 'true' if m else 'false',
            "n_total": n
        }
        for y in y_axes:
            create_synthetic_fig(
                x_axis="alpha", 
                y_axis=y, 
                regularizers=regularizers,
                features=[],
                settings=alpha_settings,
                start_date=START_DATE,
                end_date=END_DATE,
                experiment='linear'
            )


    # x axis entanglement
    print("working in entanglement plots")
    for d, a, m, n in tqdm(entanglement_params):
        entanglement_settings = {
            "d_variables": d,
            "alpha": a, 
            "n_total": n,
            "miss_well": 'true' if m else 'false',
        }
        for y in y_axes:
            create_synthetic_fig(
                x_axis="entanglement", 
                y_axis=y, 
                regularizers=regularizers,
                features=[],
                settings=entanglement_settings,
                start_date=START_DATE,
                end_date=END_DATE,
                experiment='linear'
            )

    # x axis nr of data points
    print("working in nr datapoint plots")
    for d, a, e, m in tqdm(n_total_params):
        n_total_settings = {
            "d_variables": d,
            "alpha": a, 
            "entanglement": e,
            "miss_well": 'true' if m else 'false',
        }
        for y in y_axes:
            create_synthetic_fig(
                x_axis="n_total", 
                y_axis=y, 
                regularizers=regularizers,
                features=[],
                settings=n_total_settings,
                start_date=START_DATE,
                end_date=END_DATE,
                experiment='linear'
            ) 