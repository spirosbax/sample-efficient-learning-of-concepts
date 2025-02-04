import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], '../..'))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import SplineTransformer 
from sklearn.metrics import r2_score

from utils.experiment import *
from utils.utils import set_seed
from utils.plot_utils import save_fig
from permutation_estimator.estimator import FeaturePermutationEstimator

from sklearn.linear_model import Ridge, Lasso
from sklearn.pipeline import make_pipeline

set_seed(42)

N_TOTAL = 1250
FRAC = 0.2
N_TRAIN = int(N_TOTAL * (1 - FRAC))
N_TEST = int(N_TOTAL * FRAC)
ALPHA = 0.05


n_plots = len(diffeomorphisms)

x_train, x_test = sample_x_data(
    dim=n_plots, 
    n_train=N_TRAIN, 
    n_test=N_TEST, 
    entanglement=0
)

permutation = np.random.choice(
    size=n_plots, 
    a=np.arange(n_plots), 
    replace=False
)
# permutation = np.arange(n_plots)

y_train = np.zeros((n_plots, N_TRAIN))
y_test = np.zeros((n_plots, N_TEST))

y_hat_train_1 = np.zeros((n_plots, N_TRAIN))
y_hat_test_1 = np.zeros((n_plots, N_TEST))

y_hat_train_2 = np.zeros((n_plots, N_TRAIN))
y_hat_test_2 = np.zeros((n_plots, N_TEST))

mse_group = np.zeros(n_plots)
r2_group = np.zeros(n_plots)

mse_pure = np.zeros(n_plots)
r2_pure = np.zeros(n_plots)

for i in range(n_plots):
    train_noise = np.random.normal(loc=0, scale=0.2, size=N_TRAIN)
    test_noise = np.random.normal(loc=0, scale=0.2, size=N_TEST)
    y_train[i, :] = diffeomorphisms[i](x_train[i, :]) + train_noise
    y_test[i, :] = diffeomorphisms[i](x_test[i, :]) + test_noise

    model = make_pipeline(Lasso(alpha=1e-3, fit_intercept=False, max_iter=5000, tol=1e-3))
    model.fit(
        x_train[i, :].reshape(-1, 1), 
        y_train[i, :].reshape(-1, 1)
    )

    y_hat_train_2[i, :] = model.predict(x_train[i, :].reshape(-1, 1))
    y_hat_test_2[i, :] = model.predict(x_test[i, :].reshape(-1, 1))

    mse_pure[i] = calc_mse(y_test[i, :], y_hat_test_2[i, :])
    r2_pure[i] = r2_score(y_test[i, :], y_hat_test_2[i, :])

y_train = y_train[permutation, :]
y_test = y_test[permutation, :]

for a in [0.005, 0.01, 0.1, 0.5, 0.005]:
    estimator = FeaturePermutationEstimator(
        regularizer="lasso", 
        optim_kwargs={"alpha": a}, 
        feature_transform=None, 
        d_variables=n_plots, 
        n_features=1
    )
    res, perm_hat_corr = estimator.fit(x_train, y_train)
    print(permutation)
    print(res["perm_hat_spr"])
    print(res["perm_hat_match"])
    y_hat_train_1 = estimator.predict_match(x_train)
    y_hat_test_1 = estimator.predict_match(x_test)

    y_mse_match = calc_mse(y_test, y_hat_test_1)
    y_r2_match = r2_score(y_test.T, y_hat_test_1.T)

    print("mse match", y_mse_match)
    print("r2 match", y_r2_match)

for i in range(n_plots):
    mse_group[i] = calc_mse(y_test[i, :], y_hat_test_1[i, :])
    r2_group[i] = r2_score(y_test[i, :], y_hat_test_1[i, :])

inv_perm = estimator._invert_permutation(permutation)
print(inv_perm)
fig, axs = plt.subplots(nrows=2, ncols=n_plots, figsize=(16, 6))
for i in range(n_plots):
    axs[0, i].scatter(x_train[permutation[i], :], y_train[i, :], label="gt train")
    axs[0, i].scatter(x_train[permutation[i], :], y_hat_train_1[i, :], label="estimated train")

    axs[0, i].scatter(x_test[permutation[i], :], y_test[i, :], label="gt test")
    axs[0, i].scatter(x_test[permutation[i], :], y_hat_test_1[i, :], label="estimated test")
    axs[0, i].set_ylim((-4, 4))
    axs[0, i].set_title(f"Group MSE: {mse_group[i]:.2f}, R2: {r2_group[i]:.2f}")
    axs[0, i].legend()


    axs[1, i].scatter(x_train[permutation[i], :], y_train[i, :], label="gt train")
    axs[1, i].scatter(x_train[permutation[i], :], y_hat_train_2[permutation[i], :], label="estimated train")

    axs[1, i].scatter(x_test[permutation[i], :], y_test[i, :], label="gt test")
    axs[1, i].scatter(x_test[permutation[i], :], y_hat_test_2[permutation[i], :], label="estimated test")
    axs[1, i].set_ylim((-4, 4))
    axs[1, i].set_title(f"Pure MSE: {mse_pure[i]:.2f}, R2: {r2_pure[i]:.2f}")
    axs[1, i].legend()


save_fig(fig, save_dir="figs/linear/linear_diffs")