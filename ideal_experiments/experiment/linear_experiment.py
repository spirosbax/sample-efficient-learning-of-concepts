"""
Linear experiments for a sanity check
"""
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], '../..'))

import duckdb
import numpy as np
from sklearn.metrics import r2_score

from utils.utils import (
    get_basic_parser,
    set_seed, 
)
from utils.experiment import *

from permutation_estimator.estimator import FeaturePermutationEstimator

def do_linear_experiment(
    n_total: int, 
    test_frac: float, 
    d_variables: int, 
    alpha: float, 
    regularizer: str,
    entanglement: float,
    spec: str,
    seed: int
):

    N_TRAIN = int(n_total * (1 - test_frac))
    N_TEST = int(n_total * test_frac)

    set_seed(seed)

    x_train, x_test = sample_x_data(
        dim=d_variables, 
        n_train=N_TRAIN, 
        n_test=N_TEST, 
        entanglement=entanglement
    )

    permutation = np.random.choice(
        size=d_variables, 
        a=np.arange(d_variables), 
        replace=False
    )

    if regularizer == "group":
        optim_kwargs = {"alpha": alpha, "scale_reg": "group_size", "l1_reg": 0}
    else:
        optim_kwargs = {"alpha": alpha}
    
    estimator = FeaturePermutationEstimator(
        regularizer=regularizer,
        optim_kwargs=optim_kwargs, 
        feature_transform=None, 
        d_variables=d_variables, 
        n_features=1,
    )

    y_train, y_test = sample_y_data_features(
        x_train, x_test,
        x_train, x_test, 
        dim=d_variables, 
        n_features=1, 
        permutation=permutation,
        specification=spec
    )

    res = estimator.fit(x_train, y_train)

    perm_hat_match = res["perm_hat_match"]
    perm_hat_corr = res["perm_hat_corr"]
    perm_hat_spr = res["perm_hat_spr"]

    y_hat_match = estimator.predict_match(x_test)

    perm_error_match = calc_perm_errors(permutation, perm_hat_match)
    perm_error_corr = calc_perm_errors(permutation, perm_hat_corr)
    perm_error_spr = calc_perm_errors(permutation, perm_hat_spr)

    # print(permutation)
    # print(perm_error_match)
    # print(perm_error_corr)
    # print(perm_error_spr)

    y_mse_match = calc_mse(y_test, y_hat_match)

    y_r2_match = r2_score(y_test.T, y_hat_match.T)

    # print("mse match", y_mse_match)
    # print("r2 match", y_r2_match)

    result = {
        "regularizer": regularizer,
        "n_total": n_total,
        "test_frac": test_frac,
        "d_variables": d_variables,
        "entanglement": entanglement,
        "alpha": alpha,
        "miss_well": True if spec == "well" else False,
        "seed": seed,
        "perm_error_match": perm_error_match,
        "perm_error_corr": perm_error_corr,
        "perm_error_spear": perm_error_spr,
        "mse_match": y_mse_match,
        "r2_match": y_r2_match,
        "time_match": res["time_match"],
        "time_corr": res["time_corr"],
        "time_spear": res["time_spear"]
    }
    return result


if __name__ == '__main__':
    parser = get_basic_parser()
    args = parser.parse_args()

    SEED = args.seed
    N_TOTAL = args.N_experiment
    TEST_FRAC = args.test_frac
    D_VARIABLES = args.d_variables

    ENTANGLEMENT = args.entanglement
    ALPHA = args.alpha
    SPECIFICATION = args.specified
    REGULARIZER = args.regularizer

    result = do_linear_experiment(
        n_total=N_TOTAL,
        test_frac=TEST_FRAC,
        d_variables=D_VARIABLES,
        alpha=ALPHA,
        regularizer=REGULARIZER,
        entanglement=ENTANGLEMENT,
        spec=SPECIFICATION,
        seed=SEED
    )

    with duckdb.connect("data/experiments.duckdb") as con:
        con.execute("""
            INSERT OR REPLACE INTO experiments_linear
            VALUES (
                $regularizer,
                $n_total,
                $test_frac,
                $d_variables,
                $entanglement,
                $alpha,
                $miss_well,
                $seed,
                $perm_error_match,
                $perm_error_corr,
                $perm_error_spear,
                $mse_match,
                $r2_match,
                $time_match,
                $time_corr,
                $time_spear,
                current_timestamp
            );""",
            result
        )
        con.table("experiments_linear").show(max_rows=5)

    
