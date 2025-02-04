import os
import pickle 
import numpy as np

model_mappings = {
    "CITRISVAE": ("CITRIS-VAE", None),
    "iVAE": ("iVAE", None),
    "DMS-VAE/action": ("DMS-VAE", 'action'),
    "DMS-VAE/temporal": ("DMS-VAE", 'temporal'),
    "iVAE/action": ("iVAE", "action"),
    "TCVAE/temporal": ("TCVAE", 'temporal')
}

def write_both_line(f, method, means, stds, best_per_column, i):  
    f.write(f"{{\\notsotiny {method}}}")
    half = int(len(means) / 2)

    for idx in range(len(means)):
        best_row = best_per_column[idx]
        if idx < half:
            if best_row == i:
                f.write(f" & \\textbf{{{means[idx]:.2f}}} {{\\tiny$\\pm$ \\textbf{{{stds[idx]:.2f}}}}}")
            else:
                if means[idx] == np.nan:
                    f.write(" & - ")
                elif means[idx] == 0:
                    f.write(f" & \\textbf{{{means[idx]:.2f}}} {{\\tiny$\\pm$ \\textbf{{{stds[idx]:.2f}}}}}")
                else:
                    f.write(f" & {means[idx]:.2f} {{\\tiny$\\pm$ {stds[idx]:.2f}}}")
        else:
            if best_row == i:
                f.write(f" & \\textbf{{{means[idx]:.2f}}} {{\\tiny$\\pm$ \\textbf{{{stds[idx]:.2f}}}}}")
            else:
                if means[idx] is np.nan:
                    f.write(" & - ")
                else:
                    f.write(f" & {means[idx]:.2f} {{\\tiny$\\pm$ {stds[idx]:.2f}}}")

    f.write("\\\\\n\n")

def get_mean_std(name, results, idx=None, N_ids=None):

    if idx is None:
        n_samples = results[name].shape[1]
        result_mean = results[name][N_ids, :].mean(axis=1)
        result_std = np.std(results[name][N_ids, :], axis=1) / np.sqrt(n_samples)
    else:
        res = results[name][idx, :, :, :]
        res = res[:, :, N_ids]
        n_samples = res.shape[0]
        cols = np.arange(len(N_ids))
        print(n_samples)
        print(np.sqrt(n_samples))
        if name in ["perm_error", "mse", "times"]:
            best_lambda_idx = np.argmin(res.mean(axis=0), axis=0)
        else:
            best_lambda_idx = np.argmax(res.mean(axis=0), axis=0)

        result_mean = res[:, best_lambda_idx, cols].mean(axis=0)
        result_std = np.std(res[:, best_lambda_idx, cols], axis=0) / np.sqrt(n_samples)
        print(result_std)
        print(result_std / np.sqrt(n_samples))
    return result_mean, result_std


def aggregate_mean_std(datasets, baselines, dataset_results, Ns_idx):

    mean, std = {}, {}
    for dataset in datasets:

        # citris_baseline = baseline_citris_results["CITRISVAE"]
        baseline = baselines[dataset]
        base_perm_mean, base_perm_std = get_mean_std("perm_error", baseline, N_ids=Ns_idx)
        base_r2_mean, base_r2_std = get_mean_std("r2", baseline, N_ids=Ns_idx)

        mean[dataset] = np.hstack((base_perm_mean, base_r2_mean))
        std[dataset] = np.hstack((base_perm_std, base_r2_std))

        # cit_results = citris_results["CITRISVAE"]
        results = dataset_results[dataset]

        for i in methods_idx:
            perm_mean, perm_std = get_mean_std("perm_error", results, i, N_ids=Ns_idx)
            if i != 6:
                r2_mean, r2_std = get_mean_std("r2", results, i, N_ids=Ns_idx)
            else:
                r2_mean, r2_std = nan_array(len(Ns)), nan_array(len(Ns))
            res_mean = np.hstack((perm_mean, r2_mean))
            res_std = np.hstack((perm_std, r2_std))

            mean[dataset] = np.vstack((mean[dataset], res_mean))
            std[dataset] = np.vstack((std[dataset], res_std))

    return mean, std




def nan_array(shape, dtype=object):
    a = np.empty(shape, dtype)
    a.fill(np.nan)
    return a

citris_dir = "CITRIS/cluster_checkpoints"
dms_dir = "DMS_VAE_experiments/cluster_checkpoints"

with open(os.path.join(citris_dir, "all_results.pickle"), "rb") as f:
    citris_dataset_results = pickle.load(f)
with open(os.path.join(citris_dir, "baseline_results.pickle"), "rb") as f:
    baseline_citris_results = pickle.load(f)

with open(os.path.join(dms_dir, "all_results.pickle"), "rb") as f:
    dms_dataset_results = pickle.load(f)
with open(os.path.join(dms_dir, "baseline_results.pickle"), "rb") as f:
    baseline_dms_results = pickle.load(f)

table_dir = "tables"
Ns = [10, 20, 100, 1000, 10000]
Ns_idx = [1, 2, 3, 4, 5]

methods = ["NN", "Spearman", "Linear", "Spline", "Laplacian"]
methods_idx = [6, 0, 1, 3]

citris_datasets = ["CITRISVAE", "iVAE"]
dms_datasets = ["DMS-VAE/action", "iVAE/action"]

cit_mean, cit_std = aggregate_mean_std(citris_datasets, baseline_citris_results, citris_dataset_results, Ns_idx=Ns_idx)
dms_mean, dms_std = aggregate_mean_std(dms_datasets, baseline_dms_results, dms_dataset_results, Ns_idx=Ns_idx)


with open(os.path.join(table_dir, "sample_table.tex"), 'w') as f:
    f.write("Model & ")
    f.write("Method  ")
    for N in Ns:
        f.write(f" & {N}")
    for N in Ns:
        f.write(f" & {N}")

    n_methods = len(methods)
 

    f.write("\\\\\n")
    f.write("\\toprule\n")
    f.write("\\bottomrule\n")

    f.write("\\multicolumn{12}{c}{\\rule{0pt}{0.3cm}\\cellcolor{gray!30}\\textbf{Action Sparsity Dataset}}\\\\\n")

    for dataset in dms_mean:
        best_per_col_min = np.argmin(dms_mean[dataset][:, :len(Ns)], axis=0)
        best_per_col_max = np.argmax(dms_mean[dataset][:, len(Ns):], axis=0)
        best_per_col = np.concatenate((best_per_col_min, best_per_col_max))
        for i, method in enumerate(methods):
            if i == 0: 
                f.write(f"\\multirow{{{n_methods}}}*{{{model_mappings[dataset][0]}}} & ")
            else:
                f.write(" & ")
            write_both_line(f, method, dms_mean[dataset][i, :], dms_std[dataset][i, :], best_per_col, i)

        f.write("\\hline\n")

    f.write("\\multicolumn{12}{c}{\\rule{0pt}{0.3cm}\\cellcolor{gray!30}\\textbf{Temporal Causal3DIdent Dataset}}\\\\\n")


    for dataset in cit_mean:   
        best_per_col_min = np.argmin(cit_mean[dataset][:, :len(Ns)], axis=0)
        best_per_col_max = np.argmax(cit_mean[dataset][:, len(Ns):], axis=0)
        best_per_col = np.concatenate((best_per_col_min, best_per_col_max))

        for i, method in enumerate(methods):
            if i == 0: 
                print(dataset)
                print(model_mappings[dataset][0])
                f.write(f"\\multirow{{{n_methods}}}*{{{model_mappings[dataset][0]}}} & ")
            else:
                f.write(" & ")

            write_both_line(f, method, cit_mean[dataset][i, :], cit_std[dataset][i, :], best_per_col, i)
        f.write("\\hline\n")





    