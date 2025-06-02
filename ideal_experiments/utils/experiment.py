import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier

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


def binarize_latents(all_latents):
    num_latents = all_latents.shape[0]
    bin_all_latents = np.zeros(all_latents.shape)
    for i in range(num_latents):
        min_val = np.min(all_latents[i, :])
        max_val = np.max(all_latents[i, :])

        mid_point = (max_val + min_val) / 2 
        # print('concept ', i)
        # print(f"min: {min_val}, max: {max_val}, mid: {mid_point}")
        bin_all_latents[i, all_latents[i, :] > mid_point] = 1

    return bin_all_latents


def sample_concept_labels(
    x_train: np.array, 
    x_test: np.array,
    dim: int, 
    permutation: np.array,
): 
    n_train = x_train.shape[1]

    w = np.random.random((dim, 1))

    w_neg = 2 * (w[w <= 0.5] - 1)
    w_pos = 2 * (w[w > 0.5])

    w = np.concatenate((w_neg, w_pos))
    np.random.shuffle(w)
    w = w.reshape(dim, 1)

    list_of_funcs = np.random.choice(
        diffeomorphisms_bin, 
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
    y_all = np.hstack((y_train, y_test))
    bin_labels = binarize_latents(y_all)

    return bin_labels[:, :n_train], bin_labels[:, n_train:]


def create_labels(bin_train, bin_test):
    num_latents = bin_train.shape[0]
    active_size = max(3, int(num_latents / 5))

    options = np.arange(num_latents)
    active_concepts = np.random.choice(options, size=active_size)

    sum_active_train = bin_train[active_concepts, :].sum(axis=0)
    sum_active_test = bin_test[active_concepts, :].sum(axis=0)

    y_train = np.zeros(sum_active_train.shape[0])
    y_test = np.zeros(sum_active_test.shape[0])

    sum_needed = max(1, int(num_latents / 10))
    y_train[sum_active_train > sum_needed] = 1
    y_test[sum_active_test > sum_needed] = 1
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

diffeomorphisms_bin = [
    diffeomorphism_1, 
    diffeomorphism_2, 
    diffeomorphism_3, 
    # diffeomorphism_4,
    # diffeomorphism_5
]

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







def sample_correctly(encs, latents, y_values, n_total, N_train):
    # We use as many test data points as train data points
    # run until at least 1 of each label is found
    i = 0
    while True:
        print(i, "Sampling again")
        i += 1
        indices = np.random.choice(size=n_total, a=np.arange(n_total),replace=False)
        train_idx, test_idx = indices[:N_train], indices[N_train:]

        train_encs = encs[:, train_idx]
        test_encs = encs[:, test_idx]

        train_latents = latents[:, train_idx]
        test_latents = latents[:, test_idx]

        train_y = y_values[train_idx]
        test_y = y_values[test_idx]

        latent_check = np.all([
            np.any(train_latents[col, :] == 0) and
            np.any(train_latents[col, :] == 1) and
            np.any(test_latents[col, :] == 0) and
            np.any(test_latents[col, :] == 1) 
            for col in range(train_latents.shape[0])
        ])

        label_check = np.all([
            np.any(train_y == 0) and
            np.any(train_y == 1) and
            np.any(test_y == 0) and
            np.any(test_y == 1) 
        ])
        if latent_check and label_check:
            return {
                "train_encs": train_encs,
                "test_encs": test_encs,
                "train_latents": train_latents,
                "test_latents": test_latents,
                "train_y": train_y,
                "test_y": test_y,
            }


def split_correctly(y_score, y_true, test_size):
    while True:
        print("sampling")
        y_score_train, y_score_test, y_true_train, y_true_test = train_test_split(
            y_score, y_true, test_size=test_size
        )

        label_check = np.all([
            np.any(y_true_train[:, col] == 0) and
            np.any(y_true_train[:, col] == 1) and
            np.any(y_true_test[:, col] == 0) and
            np.any(y_true_test[:, col] == 1) 
            for col in range(y_true_train.shape[1])
        ])
        if label_check:
            return y_score_train, y_score_test, y_true_train, y_true_test

def concept_correlation_nicher(y_true, y_score, multi_dim=False):
    n_concepts = y_true.shape[1]
    if multi_dim:
        corr_matrix = np.zeros((n_concepts, n_concepts))
        n_emb = y_score.shape[2]
        for i in range(n_concepts):
            for j in range(n_concepts):
                corr_intermediate =  np.corrcoef(np.hstack([y_score[:, i, :], y_true[:, j].reshape(-1, 1)]).T)
                nm = corr_intermediate[:n_emb, n_emb:]
                corr_matrix[i, j] = nm.max()
    else:
        corr_matrix = np.abs(np.corrcoef(y_score.T, y_true.T)[:n_concepts, n_concepts:])
    return corr_matrix


def concept_niche(corr_matrix, beta):
    n_concepts = corr_matrix.shape[1]
    
    niche = []
    for j in range(n_concepts):
        niche_j = [i for i in range(n_concepts) if corr_matrix[i, j] > beta]
        niche.append(niche_j)

    return niche


def niche_impurity(y_true, y_score, corr_matrix, beta, mlp, multi_dim=False):
    n_concepts = y_true.shape[1]

    niche = concept_niche(corr_matrix, beta) 
    niche_impurities = np.zeros(n_concepts)

    repeats = y_score.shape[1] / n_concepts
    for i in range(n_concepts):
        mask = np.ones(n_concepts)
        mask[niche[i]] = 0
        if multi_dim:
            mask = np.repeat(mask, repeats)
        y_score_masked = y_score * mask[np.newaxis, :]

        y_mlp_score_test = mlp.predict_proba(y_score_masked)[:, i]

        niche_impurities[i] = roc_auc_score(y_true[:, i], y_mlp_score_test)
    return niche_impurities


def nich_impurity_score(y_true, y_score, tensor=False):
    (n_samples, n_concepts) = y_true.shape
    dimensions = y_score.shape
    if tensor:
        y_score = y_score.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()

    y_score_train, y_score_test, y_true_train, y_true_test = split_correctly(
        y_score, y_true, test_size=0.2
    )
    mlp = MLPClassifier(
        hidden_layer_sizes=[20, 20], 
        activation='relu', 
        max_iter=1000, 
        batch_size=min(512, n_samples)
    )

    if len(dimensions) > 2:
        y_score_train2 = y_score_train.reshape(-1, dimensions[2] * n_concepts)
        y_score_test2 = y_score_test.reshape(-1, dimensions[2] * n_concepts)
        corr_matrix = concept_correlation_nicher(y_true, y_score, multi_dim=True)

        mlp.fit(y_score_train2, y_true_train)
    else:
        corr_matrix = concept_correlation_nicher(y_true, y_score, multi_dim=False)

        mlp.fit(y_score_train, y_true_train)
    

    betas = np.linspace(0, 1, 21)
    niche_impurities_betas = np.zeros((21, n_concepts))
    for i, beta in enumerate(betas):
        if len(dimensions) > 2:
            niche_impurities_betas[i, :] = niche_impurity(y_true_test, y_score_test2, corr_matrix, beta, mlp, multi_dim=True)
        else:
            niche_impurities_betas[i, :] = niche_impurity(y_true_test, y_score_test, corr_matrix, beta, mlp, multi_dim=True)

    scores = np.trapz(niche_impurities_betas, dx=0.05, axis=0)
    return scores.sum() / n_concepts


def ois_score(y_true, y_score, tensor=False):
    n_concepts = y_true.shape[1]   

    impurity_matrix_hat = np.zeros((n_concepts, n_concepts))
    impurity_matrix = np.zeros((n_concepts, n_concepts))

    if tensor:
        y_score = y_score.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()
    y_score_train, y_score_test, y_true_train, y_true_test = split_correctly(
        y_score, y_true, test_size=0.2
    )
    dimensions = y_score.shape
    for i in range(n_concepts):
        for j in range(n_concepts):
            if len(dimensions) == 2:
                mlp = MLPClassifier(hidden_layer_sizes=[32], activation='relu', max_iter=32)
                mlp.fit(y_score_train[:, i:i+1], y_true_train[:, j])
                y_mlp_score_test = mlp.predict_proba(y_score_test[:, i:i+1])[:, 1]
                impurity_matrix_hat[i, j] = roc_auc_score(y_true_test[:, j], y_mlp_score_test)

                mlp = MLPClassifier(hidden_layer_sizes=[32], activation='relu', max_iter=32)
                mlp.fit(y_true_train[:, i:i+1], y_true_train[:, j])
                y_mlp_score_test = mlp.predict_proba(y_true_test[:, i:i+1])[:, 1]
                impurity_matrix[i, j] = roc_auc_score(y_true_test[:, j], y_mlp_score_test)

            else:
                mlp = MLPClassifier(hidden_layer_sizes=[32], activation='relu', max_iter=32)
                mlp.fit(y_score_train[:, i, :], y_true_train[:, j])
                y_mlp_score_test = mlp.predict_proba(y_score_test[:, i, :])[:, 1]
                impurity_matrix_hat[i, j] = roc_auc_score(y_true_test[:, j], y_mlp_score_test)

                mlp = MLPClassifier(hidden_layer_sizes=[32], activation='relu', max_iter=32)
                mlp.fit(y_true_train[:, i:i+1], y_true_train[:, j])
                y_mlp_score_test = mlp.predict_proba(y_true_test[:, i:i+1])[:, 1]
                impurity_matrix[i, j] = roc_auc_score(y_true_test[:, j], y_mlp_score_test)


    norm_diff = np.linalg.norm(impurity_matrix_hat - impurity_matrix)
    
    return 2 * norm_diff / n_concepts
