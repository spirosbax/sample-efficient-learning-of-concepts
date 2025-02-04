import numpy as np
from sklearn.metrics.pairwise import (
    rbf_kernel,
    polynomial_kernel,
    laplacian_kernel,
    cosine_similarity
)

def brownian_kernel(s, t, *args):
    k_1 = np.abs(s) + np.abs(s)
    k_2 = np.abs(s - t)
    return (k_1 - k_2) / 2


def sobolev_kernel_first_order(s, t, gamma=0.5):
    return np.exp(-np.abs(s - t) * gamma)


def sobolev_kernel_second_order(s, t, gamma=0.5):
    C = np.sqrt(3) / 3
    exp = np.exp(-gamma * np.abs(s - t) * np.sqrt(3) / 2)
    sin = np.sin(-gamma * np.abs(s - t) / 2 + np.pi / 6)
    return C * exp * sin

kernels = {
    "rbf": rbf_kernel,
    "polynomial": polynomial_kernel,
    "laplacian": laplacian_kernel,
    "cosine": cosine_similarity,
    "Brownian": brownian_kernel, 
    "Silverman": sobolev_kernel_second_order
}

def true_kernel_function(
        x_knots: np.array, 
        w: np.array, 
        kernel: callable, 
        parameter: float):
    """
    x |-> sum_i c_i*k_(x_i, x), with c parameter weights
    """
    n_func = x_knots.shape[1]
    def kernel_func(x: np.array):
        dim = x.shape[0]
        n_samples = x.shape[1]

        output = np.zeros(shape=(dim, n_samples))
        if parameter is not None:
            for i in range(dim):
                K = w[i:i+1, :] * kernel(x[i:i+1, :].T, x_knots[i:i+1, :].T, parameter)
                output[i, :] = K.sum(axis=1)
        else:
            for i in range(dim):
                K = w[i:i+1, :] * kernel(x[i:i+1, :].T, x_knots[i:i+1, :].T)
                output[i, :] = K.sum(axis=1)
        return output

    return kernel_func

