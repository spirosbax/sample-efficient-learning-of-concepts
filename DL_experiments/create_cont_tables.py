import os
import pickle 
import numpy as np

from shared_utils.plot_utils import make_continuous_plot

model_mappings = {
    "CITRISVAE": ("CITRIS-VAE", None),
    "iVAE": ("iVAE", None),
    "DMS-VAE/action": ("DMS-VAE", 'action'),
    "DMS-VAE/temporal": ("DMS-VAE", 'temporal'),
    "iVAE/action": ("iVAE", "action"),
    "TCVAE/temporal": ("TCVAE", 'temporal')
}

def write_both_line(f, method, means, stds, best_per_column, i):  
    f.write(f"{{{method}}}")

    for idx in range(len(means)):
        best_row = best_per_column[idx]
        if method == "CEM":
                f.write(" & - ")
        else:
            if best_row == i:
                f.write(f" & \\textbf{{{means[idx]:.2f}}} {{\\scriptsize$\\pm$ \\textbf{{{stds[idx]:.2f}}}}}")
            else:
                if means[idx] is np.nan:
                    f.write(" & - ")
                elif means[idx] == 0:
                    f.write(f" & \\textbf{{{means[idx]:.2f}}} {{\\scriptsize$\\pm$ \\textbf{{{stds[idx]:.2f}}}}}")
                elif means[idx] <= -100:
                    f.write(f"& $\\dagger$")
                else:
                    f.write(f" & {means[idx]:.2f} {{\\scriptsize$\\pm$ {stds[idx]:.2f}}}")
        
    f.write("\\\\\n\n")

def get_mean_std_vae(name, results, idx=None, N_ids=None):

    if idx is None:
        n_samples = results[name].shape[1]
        result_mean = results[name][N_ids, :].mean(axis=1)
        result_std = np.std(results[name][N_ids, :], axis=1) / np.sqrt(n_samples)
    else:
        res = results[name][:, idx, :, :]
        res = res[:, :, N_ids]
        n_samples = res.shape[0]
        cols = np.arange(len(N_ids))
        if name in ["perm_error", "mse", "times"]:
            best_lambda_idx = np.argmin(res.mean(axis=0), axis=0)
        else:
            best_lambda_idx = np.argmax(res.mean(axis=0), axis=0)

        result_mean = res[:, best_lambda_idx, cols].mean(axis=0)
        result_std = np.std(res[:, best_lambda_idx, cols], axis=0) / np.sqrt(n_samples)
    return result_mean, result_std


def get_mean_std_cbm(name, results, N_ids=None):
    res = results[name]
    res = res[:, N_ids]
    n_samples = res.shape[0]
    cols = np.arange(len(N_ids))

    result_mean = res[:, cols].mean(axis=0)
    result_std = np.std(res[:, cols], axis=0) / np.sqrt(n_samples)
    return result_mean, result_std


def aggregate_mean_std(datasets, dataset_results, metric, Ns_idx):

    mean, std = {}, {}
    for dataset in datasets:
        results = dataset_results[dataset]

        if dataset in ["CITRISVAE", "iVAE"]:
            for i in methods_idx:
                r2_mean, r2_std = get_mean_std_vae(metric, results, i, N_ids=Ns_idx)

                if dataset in mean:
                    mean[dataset] = np.vstack((mean[dataset], r2_mean))
                    std[dataset] = np.vstack((std[dataset], r2_std))
                else:
                    mean[dataset] = r2_mean
                    std[dataset] = r2_std
        else:
            r2_mean, r2_std = get_mean_std_cbm(metric, results, N_ids=Ns_idx)

            if dataset in mean:
                mean[dataset] = np.vstack((mean[dataset], r2_mean))
                std[dataset] = np.vstack((std[dataset], r2_std))
            else:
                mean[dataset] = r2_mean
                std[dataset] = r2_std

    return mean, std


def nan_array(shape, dtype=object):
    a = np.empty(shape, dtype)
    a.fill(np.nan)
    return a

def get_best_per_column(results):
    datasets = [results[model] for model in results]
    data_arr = np.vstack(datasets)
    best_per_column = np.argmax(data_arr, axis=0)
    return best_per_column


def write_tex_table(methods, Ns, mean, std, fname):

    best_per_column = get_best_per_column(mean)
    print(best_per_column)
    with open(os.path.join(table_dir, fname), 'w') as f:
        f.write("Model & ")
        f.write("Method  ")
        for N in Ns:
            f.write(f" & {N}")

        n_methods = len(methods)


        f.write("\\\\\n")
        f.write("\\toprule\n")
        f.write("\\bottomrule\n")
        row_idx = 0
        for dataset in mean:

            if dataset in ["CITRISVAE", "iVAE"]:
                for i, method in enumerate(methods):
                    if i == 0: 
                            f.write(f"\\multirow{{{n_methods}}}*{{{dataset}}} & ")
                    else:
                        f.write(" & ")
                    write_both_line(f, method, mean[dataset][i, :], std[dataset][i, :], best_per_column, row_idx)
                    row_idx += 1
            else:
                f.write(f" {dataset } & ")
                write_both_line(f, dataset, mean[dataset], std[dataset], best_per_column, row_idx)
                row_idx += 1
                
            f.write("\\hline\n")


concept_dir = "concept_experiments/cluster_checkpoints"

with open(os.path.join(concept_dir, "vae_results.pickle"), "rb") as f:
    vae_results = pickle.load(f)
with open(os.path.join(concept_dir, "cbm_results.pickle"), "rb") as f:
    cbm_results = pickle.load(f)

table_dir = "tables"
Ns = [20, 100, 1000, 10000]
Ns_idx = [0, 1, 2, 3]

methods = ["Linear", "Spline", "RFF"]
methods_idx = [0, 1, 2]


datasets = ["CITRISVAE", "iVAE", "CBM", "CEM"]
results = vae_results
results["CBM"] = cbm_results["CBM"]
results["CEM"] = cbm_results["CEM"]

mean_label, std_label = aggregate_mean_std(datasets, results, "r2_label", Ns_idx=Ns_idx)
mean_concept, std_concept = aggregate_mean_std(datasets, results, "r2_concept", Ns_idx=Ns_idx)

best_per_column_concept = get_best_per_column(mean_concept)

write_tex_table(methods, Ns, mean_concept, std_concept, f"rebuttal_table_concept.tex")
write_tex_table(methods, Ns, mean_label, std_label, f"rebuttal_table_label.tex")








    