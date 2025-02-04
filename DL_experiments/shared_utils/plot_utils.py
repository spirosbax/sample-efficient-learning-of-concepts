import json
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], '../..'))

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "lines.markersize": 4, 
    "text.usetex": True, 
    "font.family": "serif",
    "font.size": 8
})
from matplotlib.lines import Line2D
from permutation_estimator.estimator import FeaturePermutationEstimator
from sklearn.preprocessing import SplineTransformer

from shared_utils.dl_experiments import train_predict_network

LINESTYLES = ['solid', 'dotted', 'dashdot']
MARKERS = ['v', 's', '*', "+", "D", "1"]
# Colour blind friendly colours (Okabe and Ito)according to
# https://www.nceas.ucsb.edu/sites/default/files/2022-06/Colorblind%20Safe%20Color%20Schemes.pdfa
COLOURS = [
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#D55E00",
    "#0072B2",
    "#F0E442",
    "#CC79A7",
    "#000000"
]
FACECOLOUR = "#E5E5E5"


def create_latent_plots(
    ckpt_dir, 
    N, 
    alpha, 
    test_encs, 
    test_latents
    ):
    n_knots = 20
    degree = 3

    z_dim = test_encs.shape[1]

    test_encs = test_encs[:N, :]
    test_latents = test_latents[:N, :]

    splines_all_dims = [SplineTransformer(
        n_knots=n_knots,
        degree=degree) for _ in range(z_dim)]


    estimator = FeaturePermutationEstimator(
        regularizer="group", 
        optim_kwargs={"alpha": alpha, "scale_reg": "group_size", "l1_reg": 0}, 
        feature_transform=splines_all_dims, 
        n_features=(n_knots + degree - 1),
        d_variables=z_dim,
        groups=None
    )
    _ = estimator.fit(test_encs.T, test_latents.T)
    latent_spline_hat = estimator.predict_match(test_encs.T)

    estimator = FeaturePermutationEstimator(
        regularizer="group", 
        optim_kwargs={"alpha": alpha, "scale_reg": "group_size", "l1_reg": 0}, 
        feature_transform=None, 
        n_features=1,
        d_variables=z_dim,
        groups=None
    )
    _ = estimator.fit(test_encs.T, test_latents.T)
    latent_linear_hat = estimator.predict_match(test_encs.T)

    print("Training Neural Network")
    latent_nn_hat = train_predict_network(test_encs, test_latents)

    fig, axs = plt.subplots(nrows=2, ncols=5)   
    idx = 0
    n_plot = 100
    for row in range(2):
        for col in range(5):
            axs[row, col].set_facecolor(FACECOLOUR)
            axs[row, col].grid(color="white")

            axs[row, col].scatter(
                test_encs[:n_plot, idx], test_latents[:n_plot, idx], label="Data",
                color=COLOURS[0], 
                )
            axs[row, col].scatter(
                test_encs[:n_plot, idx], latent_spline_hat[idx, :n_plot], label="Splines",
                color=COLOURS[1]
                )
            axs[row, col].scatter(
                test_encs[:n_plot, idx], latent_linear_hat[idx, :n_plot], label="Linear",
                color=COLOURS[2]
                )
            axs[row, col].scatter(
                test_encs[:n_plot, idx], latent_nn_hat[:n_plot, idx], label="NN", 
                color=COLOURS[6]
                )

            axs[row, col].set_xlabel("Learned Encodings")
            if col == 0:
                axs[row, col].set_ylabel("Ground Truth")
            idx += 1

            cm = 1/2.54  # centimeters in inches
            set_ax_size(24*cm, 10*cm, axs[row, col])
    handles, labels = axs[row, col].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, prop={'size': 12}) 
    
    check_save_fig(fig, checkpoint_dir=ckpt_dir, fname="latent_plots")
    fig.clf()
    plt.close()

def set_ax_size(w, h, ax):
    """ w, h: width, height in inches """

    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)

def save_fig(
    fig: plt.Figure,
    save_dir: str 
) -> None:
    fig.savefig(
        f"{save_dir}.pdf", 
        dpi=600,
        transparent=False,
        bbox_inches="tight"
    )
    fig.savefig(
        f"{save_dir}.png", 
        dpi=600, 
        transparent=False, 
        bbox_inches="tight"
    )


def check_save_fig(
        fig: plt.Figure,
        checkpoint_dir: str, 
        fname: str
    ) -> None:
    """
    Utility function to save figures
    """
    fig_dir_path = os.path.join(checkpoint_dir, "figs")
    fig_file_path = os.path.join(fig_dir_path, fname)
    os.makedirs(fig_dir_path, exist_ok=True)

    save_fig(fig, save_dir=fig_file_path)


def create_legend(
    labels: dict,
    linestyles=LINESTYLES, 
    markers=MARKERS, 
    colours=COLOURS
) -> plt.Figure: 

    legend_elements = []

    for i in range(len(labels.keys())):
        legend_elements.append(
            Line2D([0], [0], color=colours[i], lw=2, label=labels[i], linestyle='solid'),
        )

    fig = plt.figure(figsize=(4,3))
    fig.legend(handles=legend_elements, loc='center', ncol=6)
    return fig


def write_aggregate_zero_line(f, method, means, stds):  
    f.write(f"{method}")
    for idx in range(len(means)):
        if means[idx] == 0:
            f.write(f" & \\textbf{{{means[idx]:.2f}}} $\\pm$ \\textbf{{{stds[idx]:.2f}}}")
        else:
            f.write(f" & {means[idx]:.2f} $\\pm$ {stds[idx]:.2f}")
    f.write("\\\\\n\n")

def write_aggregate_one_line(f, method, means, stds):  
    f.write(f"{method}")
    for idx in range(len(means)):
        if means[idx] == 1:
            f.write(f" & \\textbf{{{means[idx]:.2f}}} $\\pm$ \\textbf{{{stds[idx]:.2f}}}")
        else:
            f.write(f" & {means[idx]:.2f} $\\pm$ {stds[idx]:.2f}")
    f.write("\\\\\n\n")


def write_aggregate_table_values(
    f, 
    Ns, 
    alphas,
    all_results, 
    baseline_all_results,
    name
    ):
    model_mappings = {
        "CITRISVAE": ("CITRIS-VAE", None),
        "iVAE": ("iVAE", None),
        "DMS-VAE/action": ("DMS-VAE", 'action'),
        "DMS-VAE/temporal": ("DMS-VAE", 'temporal'),
        "iVAE/action": ("iVAE", "action"),
        "TCVAE/temporal": ("TCVAE", 'temporal')
    }

    best_alphas = {}

    if "CITRISVAE" in all_results:
        f.write("Model & ")
        f.write("Method  ")
    else:
        f.write("Dataset & ")
        f.write("Model & ")
        f.write("Method  ")

    for N in Ns:
        f.write(f" & {N}")
    f.write("\\\\\n")
    f.write("\\toprule\n")
    f.write("\\bottomrule\n")

    if name == "perm_error":
        methods = ["Linear", "Spline", "RFF", "Laplacian", "Pearson", "Spearman"]
    else:
        methods = ["Linear", "Spline", "RFF", "Laplacian"]

    for model in all_results:
        best_alphas[model] = {}
        baseline_results = baseline_all_results[model][name]
        baseline_means = baseline_results.mean(axis=1)
        baseline_std = np.std(baseline_results, axis=1)

        model_name, dataset = model_mappings[model]
        if name == "perm_error":
            if dataset is not None:
                if model_name == "DMS-VAE":
                    f.write(f"\\multirow{{14}}*{{{dataset}}} & \\multirow{{7}}*{{{model_name}}} & ")
                else:
                    f.write(f" & \\multirow{{7}}*{{{model_name}}} & ")
            else:
                f.write(f"\\multirow{{7}}*{{{model_name}}} & ")
        else:
            if dataset is not None:
                if model_name == "DMS-VAE":
                    f.write(f"\\multirow{{10}}*{{{dataset}}} & \\multirow{{5}}*{{{model_name}}} & ")
                else:
                    f.write(f" & \\multirow{{5}}*{{{model_name}}} & ")
            else:
                f.write(f"\\multirow{{5}}*{{{model_name}}} & ")

        if name in ["perm_error", "mse", "times"]:
            write_aggregate_zero_line(f, "NN", baseline_means, baseline_std)
        else:
            write_aggregate_one_line(f, "NN", baseline_means, baseline_std)

        N_ids = np.arange(len(Ns))
        for i, method in enumerate(methods):
            results = all_results[model][name][i, :, :, :]
            if dataset is not None:
                f.write(" & & ")
            else:
                f.write(" & ")
            if name in ["perm_error", "mse", "times"]:
                best_lambda_idx = np.argmin(results.mean(axis=0), axis=0)

                if method in ["Pearson", "Spearman"]:
                    result_means = results.mean(axis=(0, 1))
                    result_stds = np.std(results, axis=(0, 1))
                else:
                    result_means = results[:, best_lambda_idx, N_ids].mean(axis=0)
                    result_stds = np.std(results[:, best_lambda_idx, N_ids], axis=0)
                write_aggregate_zero_line(f, method, result_means, result_stds)
            else:
                best_lambda_idx = np.argmax(results.mean(axis=0), axis=0)

                result_means = results[:, best_lambda_idx, N_ids].mean(axis=0)
                result_stds = np.std(results[:, best_lambda_idx, N_ids], axis=0)
                write_aggregate_one_line(f, method, result_means, result_stds)

            best_alphas[model][method] = list(np.array(alphas)[best_lambda_idx.astype(int)])
        if model_name == "iVAE" or model_name == "TCVAE":
            f.write("\\hline")
        elif model_name == "CITRIS-VAE":
            f.write("\\cline{2-8}")
        else:
            f.write("\\cline{2-9}")
    return best_alphas


def make_aggregate_tables(
    all_results, 
    baseline_all_results,
    Ns,
    alphas,
    ckpt_dir):
    table_dir = os.path.join(ckpt_dir, "tables")
    os.makedirs(table_dir, exist_ok=True)
    
    with open(os.path.join(table_dir, f"aggregate_perm_errors.tex"), 'w') as f:
        best_alphas_perm_error = write_aggregate_table_values(
            f, Ns, alphas, all_results, baseline_all_results, "perm_error"
        )
    with open(os.path.join(table_dir, f"aggregate_mse.tex"), 'w') as f:
        best_alphas_mse = write_aggregate_table_values(
            f, Ns, alphas, all_results, baseline_all_results, "mse"
        )
    with open(os.path.join(table_dir, f"aggregate_r2.tex"), 'w') as f:
        best_alphas_r2 = write_aggregate_table_values(
            f, Ns, alphas, all_results, baseline_all_results, "r2"
        )
    with open(os.path.join(table_dir, f"aggregate_times.tex"), 'w') as f:
        best_alphas_times = write_aggregate_table_values(
            f, Ns, alphas, all_results, baseline_all_results, "times"
        )

    with open(os.path.join(table_dir, "best_lambda_perm_erros.json"), 'w') as f:
        json.dump(best_alphas_perm_error, f)
    with open(os.path.join(table_dir, "best_lambda_mse.json"), 'w') as f:
        json.dump(best_alphas_mse, f)
    with open(os.path.join(table_dir, "best_lambda_r2.json"), 'w') as f:
        json.dump(best_alphas_r2, f)   
    with open(os.path.join(table_dir, "best_lambda_times.json"), 'w') as f:
        json.dump(best_alphas_times, f)


def create_dl_plots(
    alphas: np.array,
    Ns: np.array,
    error_arrays: np.array,
    name:str, 
    ckpt_dir: os.PathLike
):

    labels = {
        0: "Group \& Linear",
        1: "Group \& Spline",
        2: "Pearson",
        3: "Spearman",
    }

    fig_dir = os.path.join(ckpt_dir, "figs")
    os.makedirs(fig_dir, exist_ok=True)

    nr_labels = error_arrays.shape[0]
    for j, N in enumerate(Ns):
        fig, ax = plt.subplots()

        ax.set_facecolor(FACECOLOUR)
        ax.grid(color="white")
        for idx in range(nr_labels):
            y_mean = error_arrays[idx, :, :, j].mean(axis=0)
            y_lb = np.quantile(error_arrays[idx, :, :, j], 0.25, axis=0)
            y_ub = np.quantile(error_arrays[idx, :, :, j], 0.75, axis=0)

            ax.plot(alphas, y_mean, color=COLOURS[idx], marker=MARKERS[0], label=labels[idx])
            ax.fill_between(alphas, y_lb, y_ub, alpha=.1, color=COLOURS[idx])
        ax.set_ylim((0, 1))

        ax.set_title(f"Nr of samples={N}", fontsize=6)

        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xlabel("$\lambda$")
        ax.set_xscale("log")

        cm = 1/2.54  # centimeters in inches
        set_ax_size(4*cm, 2.5*cm, ax)
        save_fig(fig, os.path.join(fig_dir, f"{name}_N_{N}"))
        fig.clf()
        plt.close()

    for j, alpha in enumerate(alphas):
        fig, ax = plt.subplots()

        ax.set_facecolor(FACECOLOUR)
        ax.grid(color="white")
        for idx in range(nr_labels):
            y_mean = error_arrays[idx, :, j, :].mean(axis=0)
            y_lb = np.quantile(error_arrays[idx, :, j, :], 0.25, axis=0)
            y_ub = np.quantile(error_arrays[idx, :, j, :], 0.75, axis=0)

            ax.plot(Ns, y_mean, color=COLOURS[idx], marker=MARKERS[0], label=labels[idx])
            ax.fill_between(Ns, y_lb, y_ub, alpha=.1, color=COLOURS[idx])
        ax.set_ylim((0, 1))

        ax.set_title(f"$\lambda$={alpha}", fontsize=6)
        ax.set_xlabel("$N$")

        cm = 1/2.54  # centimeters in inches
        set_ax_size(4*cm, 2.5*cm, ax)
        save_fig(fig, os.path.join(fig_dir, f"{name}_alpha_{alpha}"))
        fig.clf()
        plt.close()

    legend = create_legend(labels)
    save_fig(legend, os.path.join(fig_dir, f"legend"))
    legend.clf()
    plt.close()


def create_dl_tables(
    alphas: np.array,
    Ns: np.array,
    error_arrays: np.array,
    name: str, 
    ckpt_dir: os.PathLike
) -> None:
    table_dir = os.path.join(ckpt_dir, "tables")
    os.makedirs(table_dir, exist_ok=True)

    with open(os.path.join(table_dir, f"table_{name}_group_linear.tex"), 'w') as f:
        write_table_values(f, alphas, Ns, error_arrays[0, :, :, :])
    with open(os.path.join(table_dir, f"table_{name}_group_spline.tex"), 'w') as f:
        write_table_values(f, alphas, Ns, error_arrays[1, :, :, :])


def write_table_values(
    f,
    alphas: np.array, 
    Ns: np.array,
    error_array: np.array
) -> None:
    error_means = error_array.mean(axis=0)
    error_stds = np.std(error_array, axis=0)
    f.write("&")
    for N in Ns:
        f.write(f" & {N}")
    f.write("\\\\\n")
    f.write("\\toprule\n")
    f.write("\\bottomrule\n")
    f.write("\\multirow{8}*{$\\lambda$}\n")
    for i, alpha in enumerate(alphas):
        error_row_mean = error_means[i, :]
        error_row_std = error_stds[i, :]
        write_line(f, alpha, error_row_mean, error_row_std)

def write_line(
    f,
    alpha: float, 
    row_data_mean: np.array,
    row_data_std: np.array,
) -> None:
    f.write(f"& {alpha}")
    for idx in range(len(row_data_mean)):
        lb = row_data_mean[idx] - row_data_std[idx]
        if lb <= 0:
            f.write(f" & \\textbf{{{row_data_mean[idx]:.2f}}} $\\pm$ \\textbf{{{row_data_std[idx]:.2f}}}")
        else:
            f.write(f" & {row_data_mean[idx]:.2f} $\\pm$ {row_data_std[idx]:.2f}")
    f.write("\\\\\n\n")


def make_time_plot(
    all_results, 
    baseline_all_results,
    Ns,
    alphas,
    ckpt_dir  
):
    fig_dir = os.path.join(ckpt_dir, "figs")
    os.makedirs(fig_dir, exist_ok=True)

    model_mappings = {
        "CITRISVAE": ("CITRIS-VAE", None),
        "iVAE": ("iVAE", None),
        "DMS-VAE/action": ("DMS-VAE", 'action'),
        "DMS-VAE/temporal": ("DMS-VAE", 'temporal'),
        "iVAE/action": ("iVAE", "action"),
        "TCVAE/temporal": ("TCVAE", 'temporal')
    }
    methods = [(0, "Linear"), (1, "Spline"), (2, "RFF"), (3, "Laplacian"), (4, "Two Stage")]
    
    legend_elements = []
    legend= plt.figure(figsize=(4,3))
    legend.legend(handles=legend_elements, loc='center', ncol=6)

    fig, ax = plt.subplots(ncols=1, nrows=1)
    ax.set_facecolor(FACECOLOUR)
    ax.grid(color="white")
    N_ids = np.arange(len(Ns))

    baseline_results = [baseline_all_results[model]["times"] for model in baseline_all_results]
    baseline_results = np.hstack(baseline_results)
    baseline_means = baseline_results.mean(axis=1)
    baseline_lb = np.quantile(baseline_results, 0.25, axis=1)
    baseline_ub = np.quantile(baseline_results, 0.75, axis=1)
    # baseline_std = np.std(baseline_results, axis=1)


    ax.plot(Ns, baseline_means, color=COLOURS[-1], marker=MARKERS[-1], label="NN", markersize=8)
    ax.fill_between(Ns, baseline_lb, baseline_ub, color=COLOURS[-1], alpha=.1)   

    legend_elements.append(
        Line2D([0], [0], color=COLOURS[-1], marker=MARKERS[-1], lw=2, label="NN", linestyle='solid', markersize=8)
    )

    idx = 0 
    m_sizes = [4, 4, 8, 8, 4]
    for i, method in methods:
        results = [all_results[model]['times'][i, :, :, :] for model in all_results]
        results = np.concatenate(results, axis=0)
        best_lambda_idx = np.argmin(results.mean(axis=0), axis=0)

        result_means = results[:, best_lambda_idx, N_ids].mean(axis=0)
        results_lb = np.quantile(results[:, best_lambda_idx, N_ids], 0.25, axis=0)
        results_ub = np.quantile(results[:, best_lambda_idx, N_ids], 0.75, axis=0)
        # result_stds = np.std(results[:, best_lambda_idx, N_ids], axis=0)

        ax.plot(Ns, result_means, color=COLOURS[idx], marker=MARKERS[idx], label=method, markersize=m_sizes[i])
        ax.fill_between(
            Ns, 
            results_lb, 
            results_ub, 
            alpha=.1,
            color=COLOURS[idx]
        )   

        legend_elements.append(
            Line2D([0], [0], color=COLOURS[idx], marker=MARKERS[idx], lw=2, label=method,  markersize=m_sizes[i], linestyle='solid')
        )

        idx += 1

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel("Nr. data points $(n)$")
    ax.set_ylabel("Execution time $(s)$")
    cm = 1/2.54  # centimeters in inches
    set_ax_size(6.5*cm, 4*cm, ax)
    save_fig(fig, os.path.join(fig_dir, f"times"))

    legend.legend(handles=legend_elements, loc='upper center', ncol=6, prop={'size': 8}) 
    save_fig(legend, os.path.join(fig_dir, "legend"))

    legend.clf()
    fig.clf()
    plt.close()