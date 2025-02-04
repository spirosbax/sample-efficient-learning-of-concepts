import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import List

from utils.kernels import *


NOISE_STD = 0.5


def theoretical_lambda(
        n: int, 
        sigma: float, 
        p: int, 
        d: int,
):
    if p != 0:
        constant = 2 * sigma / np.sqrt(n)
        log_term_1 = 4 * np.log(n) + 4 * np.log(d)
        log_term = 1 + np.sqrt(log_term_1 / p) + log_term_1 / p
        return constant * np.sqrt(log_term) 
    else:
        return 1000000


def sample_x_data(
        dim: int, 
        n_train: int, 
        n_test: int, 
        entanglement: float
):
    cov_matrix = (1 - entanglement) * np.eye(dim) + entanglement 

    x_train = np.random.multivariate_normal(
        mean=np.zeros(dim), 
        cov=cov_matrix, 
        size=n_train
        ).T
    x_test = np.random.multivariate_normal(
        mean=np.zeros(dim),
        cov=cov_matrix,
        size=n_test
        ).T
    return x_train, x_test


def full_transform(x: np.array, w: np.array, permutation: np.array, p: int):
    """
    x |-> permutation ∘ (w @ phi(x)), with w a weight matrix
    """
    assert w.shape[1] == p, f"weights and powers need to be the same, {w.shape[1]} != {p}"
    x= x.reshape(
        w.shape[0], p, x.shape[1]).transpose(0, 2, 1)
    w_x = w[:, np.newaxis, :] * x
    phi_x = np.sum(w_x, axis=2) / p
    return phi_x[permutation, :]


def sample_weights(
        n: np.array, 
        sigma: float, 
        n_features: int, 
        dim: int
) -> np.array:
    """Sample weights in the linear feature models in such a way that the
    norm of the coefficients is higher than the theoretical threshold, 
    and they are evenly distributed among negative and positve terms.
    """
    lambda_opt = theoretical_lambda(
        n=n, 
        sigma=sigma, 
        p=n_features, 
        d=dim
    )

    w = 2 * (np.random.random(dim * n_features) - 0.5)
    w = w.reshape(dim, n_features)

    w_norms = np.linalg.norm(w, axis=1)[:, np.newaxis]
    scalings = 16 * (np.random.random(dim).reshape(dim, 1) + 1)
    w = (w / w_norms) * scalings * lambda_opt

    return w


def sample_y_data_features(
        x_train: np.array, 
        x_test: np.array,
        phi_x_train: np.array,
        phi_x_test: np.array,
        dim: int, 
        n_features: int, 
        permutation: np.array,
        specification: str
): 
    n_train = x_train.shape[1]

    if specification == "well":
        w = sample_weights(
            n=n_train, 
            sigma=NOISE_STD, 
            n_features=n_features, 
            dim=dim
        )

        # Ensuring all the assumptions are indeed fullfilled
        std_scalars = [StandardScaler() for _ in range(dim)]   
        pcas = [PCA(n_components=n_features) for _ in range(dim)]   

        phi_x_all = np.hstack((phi_x_train, phi_x_test))
        transformed_x = np.zeros(shape=phi_x_all.shape)
        
        for i in range(dim):
            group_slice = slice(i*n_features, (i + 1)*n_features)
            features = std_scalars[i].fit_transform(phi_x_all[group_slice, :].T)
            features = pcas[i].fit_transform(features)

            transformed_x[group_slice, :] = features.T

        transformed_x_train = transformed_x[:, :n_train]
        transformed_x_test = transformed_x[:, n_train:]

        y_train = full_transform(
            x=transformed_x_train, 
            w=w, 
            permutation=permutation, 
            p=n_features
        )
        y_test = full_transform(
            x=transformed_x_test, 
            w=w, 
            permutation=permutation, 
            p=n_features
        )

    elif specification == "miss":
        w = np.random.random((dim, 1))

        w_neg = 2 * (w[w <= 0.5] - 1)
        w_pos = 2 * (w[w > 0.5])

        w = np.concatenate((w_neg, w_pos))
        np.random.shuffle(w)
        w = w.reshape(dim, 1)

        list_of_funcs = np.random.choice(
            diffeomorphisms, 
            size=dim
        )

        y_train = true_alignment(
            x_train,
            scalars=w, 
            list_of_funcs=list_of_funcs, 
            permutation=permutation 
        )       
        y_test = true_alignment(
            x_test,
            scalars=w, 
            list_of_funcs=list_of_funcs, 
            permutation=permutation 
        )       
    elif specification == "linear":
        w = sample_weights(
            n=n_train, 
            sigma=NOISE_STD, 
            n_features=1, 
            dim=dim
        )

        y_train = linear_alignment(
            x_train,
            scalars=w, 
            permutation=permutation 
        )       
        y_test = linear_alignment(
            x_test,
            scalars=w, 
            permutation=permutation 
        )

    noise_train = np.random.normal(
        loc=0, 
        scale=NOISE_STD, 
        size=(dim, x_train.shape[1])
    )
    noise_test = np.random.normal(
        loc=0, 
        scale=NOISE_STD, 
        size=(dim, x_test.shape[1])
    )
    y_train = y_train + noise_train
    y_test = y_test + noise_test


    return y_train, y_test


def sample_y_data_kernels(
        x_train: np.array, 
        x_test: np.array,
        x_kernel: np.array,
        kernel: callable,
        parameter:float,
        dim: int, 
        permutation: np.array,
        specification: str
): 
    if specification == "well":
        w = np.random.random(dim * x_kernel.shape[1])

        w_neg = 2 * (w[w <= 0.5] - 1)
        w_pos = 2 * (w[w > 0.5])

        w = np.concatenate((w_neg, w_pos))
        np.random.shuffle(w)
        w = w.reshape((dim, x_kernel.shape[1]))

        kern_func = true_kernel_function(
            x_kernel,
            w=w,
            kernel=kernel,
            parameter=parameter
        )
        y_train = kern_func(x_train)[permutation, :]
        y_test = kern_func(x_test)[permutation, :]
    else:
        w = np.random.random((dim, 1))

        w_neg = 2 * (w[w <= 0.5] - 1)
        w_pos = 2 * (w[w > 0.5])

        w = np.concatenate((w_neg, w_pos))
        np.random.shuffle(w)
        w = w.reshape(dim, 1)

        list_of_funcs = np.random.choice(
            diffeomorphisms, 
            size=dim
        )

        y_train = true_alignment(
            x_train,
            scalars=w, 
            list_of_funcs=list_of_funcs,
            permutation=permutation 
        )       
        y_test = true_alignment(
            x_test,
            scalars=w, 
            list_of_funcs=list_of_funcs,
            permutation=permutation 
        )
    
    noise_train = np.random.normal(
        loc=0, 
        scale=NOISE_STD, 
        size=(dim, x_train.shape[1])
    )
    noise_test = np.random.normal(
        loc=0, 
        scale=NOISE_STD, 
        size=(dim, x_test.shape[1])
    )
    y_train = y_train + noise_train
    y_test = y_test + noise_test

    return y_train, y_test


def linear_alignment(
        x: np.array, 
        scalars: np.array, 
        permutation: np.array
) -> np.array:
    out = scalars * x
    return out[permutation, :]


def diffeomorphism_1(x):
    x = x + 0.1
    if x >= 0:
        y = np.cbrt(x)
        y = np.cbrt(y)
    else: 
        y = np.cbrt(0.01 * x)
        y = np.cbrt(y)
    return 2 * y

def diffeomorphism_2(x):
    x = x - .1
    if x >= 0:
        y = np.cbrt(x)
        y = np.cbrt(y)
    else: 
        y = np.cbrt(0.01 * x)
        y = np.cbrt(y)
    return 2 * y

def diffeomorphism_3(x):
    x = x - 0.7
    x = 1.5 * np.pi * x
    x = x - np.sin(x)
    x = x - np.sin(x)
    x = x - np.sin(x)
    x = x - np.sin(x)
    return x / (1.25 * np.pi) + 0.5


def diffeomorphism_4(x):
    x = 3 * x
    if x >= 0:
        y = np.power(x, 3)
    else:
        y = np.power(0.5 * x, 3)

    return (1 / 6) * y

def diffeomorphism_5(x):
    x = 3 * x
    if x >= 0:
        y = np.power(x, 4)
    else:
        y = -1 * np.power(0.5 * x, 4)
    return (1 / 24) * y

diffeomorphism_1 = np.vectorize(diffeomorphism_1)
diffeomorphism_2 = np.vectorize(diffeomorphism_2)
diffeomorphism_3 = np.vectorize(diffeomorphism_3)
diffeomorphism_4 = np.vectorize(diffeomorphism_4)
diffeomorphism_5 = np.vectorize(diffeomorphism_5)

diffeomorphisms = [
    diffeomorphism_1, 
    diffeomorphism_2, 
    diffeomorphism_3, 
    diffeomorphism_4, 
    diffeomorphism_5
]

def true_alignment(
        x: np.array, 
        list_of_funcs: List[callable], 
        scalars: np.array, 
        permutation: np.array
):  
    out = scalars * np.array([
        f(row) for f, row in zip(list_of_funcs, x)
    ])
    return out[permutation, :]

def calc_perm_errors(perm, perm_hat):
    return np.sum(perm != perm_hat) / len(perm)
 

def calc_weight_errors(w, w_hat):
    mean_diff = np.mean(w_hat - w)
    mse_diff = np.power(w_hat - w, 2).mean()
    return mean_diff, mse_diff


def calc_mse(y, y_hat):
    return np.power(y - y_hat, 2).mean()
    

def save_error_data(ckpt_dir: str, perm_error:np.array, y_mse: np.array, features: List[str], suffix: str):
    data_dir = os.path.join(ckpt_dir, "data")

    for i, feature in enumerate(features):
        np.savetxt(os.path.join(data_dir, f"{feature}_perm_error_{suffix}"), perm_error[:, i], fmt='%.6f')
        np.savetxt(os.path.join(data_dir, f"{feature}_y_mse_{suffix}"), y_mse[:, i], fmt='%.6f')