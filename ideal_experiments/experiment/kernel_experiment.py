"""
Checking if quadratic kernels can be learned with onl (2d) points
"""
import sys
import os
import warnings
sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], '../..'))

import duckdb
import numpy as np
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score
from sklearn.linear_model import SGDClassifier
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter("ignore", category=ConvergenceWarning)

from utils.utils import (
    get_basic_parser,
    set_seed, 
)
from utils.experiment import *
from utils.kernels import *
from permutation_estimator.estimator import (
    KernelizedPermutationEstimator,
    FeaturePermutationEstimator
)


def do_kernel_experiment(
    n_total: int, 
    test_frac: float, 
    d_variables: int, 
    n_func: int,
    kernel_name: str ,
    n_kernel: int,
    alpha: float, 
    regularizer: str,
    entanglement: float,
    spec: str, 
    seed: int, 
):
    N_TRAIN = int(n_total * (1 - test_frac))
    N_TEST = int(n_total * test_frac)

    KERNEL_PARAMETER = {
        "polynomial": 3, 
        "rbf": 1, 
        "laplacian": 1,
        "cosine": None
    }

    set_seed(seed)

    parameter = KERNEL_PARAMETER[kernel_name]
    kernel = kernels[kernel_name]

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
    
    x_kernel_true = np.random.standard_normal(
        size=(d_variables, n_func)
    )

    y_train, y_test = sample_y_data_kernels(
        x_train,
        x_test, 
        x_kernel_true,
        parameter=parameter, 
        kernel=kernel,
        dim=d_variables, 
        permutation=permutation, 
        specification=spec
    )
    
    # Kernel features 
    if regularizer == "group":
        optim_kwargs = {"alpha": alpha, "scale_reg": "group_size", "l1_reg": 0}
    else:
        optim_kwargs = {"alpha": alpha}

    linear = FeaturePermutationEstimator(
        regularizer=regularizer, 
        optim_kwargs=optim_kwargs,
        feature_transform=None, 
        d_variables=d_variables,
        n_features=1
    )

    estimator = KernelizedPermutationEstimator(
        regularizer=regularizer, 
        optim_kwargs=optim_kwargs, 
        kernel=kernel_name, 
        parameter=parameter,
        d_variables=d_variables, 
        n_features=n_kernel
    )

    res = estimator.fit(X=x_train, y=y_train)
    res_linear = linear.fit(X=x_train, y=y_train)

    perm_hat_match = res["perm_hat_match"]
    perm_hat_linear = res_linear["perm_hat_match"]
    perm_hat_corr = res["perm_hat_corr"]
    perm_hat_spr = res["perm_hat_spr"]

    y_hat_match = estimator.predict_match(x_test)
    y_hat_linear = linear.predict_match(x_test)

    perm_error_match = calc_perm_errors(permutation, perm_hat_match)
    perm_error_linear = calc_perm_errors(permutation, perm_hat_linear)
    perm_error_corr = calc_perm_errors(permutation, perm_hat_corr)
    perm_error_spr = calc_perm_errors(permutation, perm_hat_spr)

    # print(permutation)
    # print(perm_error_match)
    # print(perm_error_linear)
    # print(perm_error_corr)
    # print(perm_error_spr)

    y_mse_match = calc_mse(y_test, y_hat_match)
    y_mse_linear = calc_mse(y_test, y_hat_linear)

    y_r2_match = r2_score(y_test.T, y_hat_match.T)
    y_r2_linear = r2_score(y_test.T, y_hat_linear.T)

    # print("mse match", y_mse_match)
    # print("mse linear", y_mse_linear)
    # print("r2 match", y_r2_match)
    # print("r2 linear", y_r2_linear)
    
    result = {
        "regularizer": regularizer,
        "n_total": n_total,
        "test_frac": test_frac,
        "d_variables": d_variables,
        "kernel": kernel_name,
        "n_kernel": n_kernel,
        "entanglement": entanglement,
        "alpha": alpha,
        "miss_well": True if spec == "well" else False,
        "seed": seed,
        "perm_error_match": perm_error_match,
        "perm_error_linear": perm_error_linear,
        "perm_error_corr": perm_error_corr,
        "perm_error_spear": perm_error_spr,
        "mse_match": y_mse_match,
        "mse_linear": y_mse_linear,
        "r2_match": y_r2_match,
        "r2_linear": y_r2_linear,
        "time_match": res["time_match"],
        "time_linear": res_linear["time_match"],
        "time_corr": res["time_corr"],
        "time_spear": res["time_spear"]
    }
    return result



def do_kernel_experiment_bin(
    n_total: int, 
    test_frac: float, 
    d_variables: int, 
    n_func: int,
    kernel_name: str ,
    n_kernel: int,
    alpha: float, 
    regularizer: str,
    entanglement: float,
    seed: int, 
):
    N_TRAIN = int(n_total * (1 - test_frac))
    N_TEST = int(n_total * test_frac)

    KERNEL_PARAMETER = {
        "polynomial": 3, 
        "rbf": 1, 
        "laplacian": 1,
        "cosine": None
    }

    set_seed(seed)

    parameter = KERNEL_PARAMETER[kernel_name]
    kernel = kernels[kernel_name]

    x_train, _ = sample_x_data(
        dim=d_variables, 
        n_train=N_TRAIN, 
        n_test=N_TEST, 
        entanglement=entanglement
    )

    _, x_test = sample_x_data(
        dim=d_variables, 
        n_train=N_TRAIN, 
        n_test=N_TEST, 
        entanglement=0
    )
    
    permutation = np.random.choice(
        size=d_variables,
        a=np.arange(d_variables),
        replace=False
    )

    optim_kwargs = {"alpha": alpha, "scale_reg": "group_size", "l1_reg": 0}


    estimator = KernelizedPermutationEstimator(
        regularizer=regularizer, 
        optim_kwargs=optim_kwargs, 
        kernel=kernel_name, 
        parameter=parameter,
        d_variables=d_variables, 
        n_features=n_kernel, 
        groups=np.arange(d_variables)
    )
    predictor = SGDClassifier(loss="log_loss")

    concept_train, concept_test = sample_concept_labels(
        x_train, x_test,
        dim=d_variables, 
        permutation=permutation,
    )
    y_train, y_test = create_labels(concept_train, concept_test)

    arrays = sample_correctly(
        np.hstack((x_train, x_test)), 
        np.hstack((concept_train, concept_test)), 
        np.hstack((y_train, y_test)), 
        n_total=N_TRAIN + N_TEST,
        N_train=N_TRAIN
    )
    x_train, x_test = arrays["train_encs"], arrays["test_encs"]
    concept_train, concept_test = arrays["train_latents"], arrays["test_latents"]
    y_train, y_test = arrays["train_y"], arrays["test_y"]

    res = estimator.fit(x_train, concept_train)
    
    train_latent_hat = estimator.predict_match(x_train)
    test_latent_hat_score = estimator.predict_match(x_test)

    acc_concept = accuracy_score(concept_test.reshape(-1, 1), (test_latent_hat_score < 0).reshape(-1, 1))
    roc_concept = roc_auc_score(concept_test.T, -test_latent_hat_score.T, multi_class="ovr")

    # OIS score 
    if d_variables <= 20:
        ois_concept = ois_score(concept_test.T, -test_latent_hat_score.T)
        nis_concept = nich_impurity_score(concept_test.T, -test_latent_hat_score.T)
    else:
        ois_concept = 2
        nis_concept = 2

    predictor.fit(train_latent_hat.T, y_train)
    test_y_hat = predictor.predict(test_latent_hat_score.T)
    test_y_hat_score = predictor.predict_proba(test_latent_hat_score.T)[:, 1]
    
    acc_label = accuracy_score(y_test, test_y_hat)
    roc_label = roc_auc_score(y_test, test_y_hat_score)

    perm_hat_match = res["perm_hat_match"]

    perm_error_match = calc_perm_errors(permutation, perm_hat_match)

    result = {
        "regularizer": regularizer,
        "n_total": n_total,
        "test_frac": test_frac,
        "d_variables": d_variables,
        "kernel": kernel_name,
        "n_kernel": n_kernel,
        "entanglement": entanglement,
        "alpha": alpha,
        "seed": seed,
        "perm_error_match": perm_error_match,
        "acc_label": acc_label, 
        "roc_label": roc_label, 
        "acc_concept": acc_concept, 
        "roc_concept": roc_concept, 
        "ois_concept": ois_concept,
        "nis_concept": nis_concept,
        "time_match": res["time_match"],
    }
    return result

  


if __name__ == "__main__":
    parser = get_basic_parser()
    parser.add_argument('--kernel', type=str, 
                        choices=["polynomial", "rbf", "Brownian", "Sobolev"], 
                        default="rbf",
                        help='What kernel to be used.')
    parser.add_argument('--N_kernel', type=int, 
                        default=10,
                        help='How many points in the kernel approximation should be used.')
    parser.add_argument('--N_func', type=int, 
                        default=10, 
                        help='Number of function points in the true function.')
    args = parser.parse_args()

    SEED = args.seed
    N_TOTAL = args.N_experiment
    TEST_FRAC = args.test_frac

    D_VARIABLES = args.d_variables
    N_FUNC = args.N_func
    KERNEL = args.kernel
    N_KERNEL = args.N_kernel

    ENTANGLEMENT = args.entanglement
    ALPHA = args.alpha
    SPECIFICATION = args.specified
    REGULARIZER = args.regularizer
    
    result = do_kernel_experiment(
        n_total=N_TOTAL,
        test_frac=TEST_FRAC,
        d_variables=D_VARIABLES,
        n_func=N_FUNC,
        kernel_name=KERNEL,
        n_kernel=N_KERNEL,
        alpha=ALPHA,
        regularizer=REGULARIZER,
        entanglement=ENTANGLEMENT,
        spec=SPECIFICATION,
        seed=SEED
    )


    with duckdb.connect("data/experiments.duckdb") as con:
        con.execute("""
            INSERT OR REPLACE INTO experiments_kernel
            VALUES (
                $regularizer,
                $n_total,
                $test_frac,
                $d_variables,
                $kernel,
                $n_kernel,
                $entanglement,
                $alpha,
                $miss_well,
                $seed,
                $perm_error_match,
                $perm_error_linear,
                $perm_error_corr,
                $perm_error_spear,
                $mse_match,
                $mse_linear,
                $r2_match,
                $r2_linear,
                $time_match,
                $time_linear,
                $time_corr,
                $time_spear,
                current_timestamp
            );""",
            result
        )
        con.table("experiments_kernel").show(max_rows=5)

   



