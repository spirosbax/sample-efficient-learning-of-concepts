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

    for idx in range(len(means)):
        best_row = best_per_column[idx]
        if best_row == i:
            f.write(f" & \\textbf{{{means[idx]:.2f}}} {{\\tiny$\\pm$ \\textbf{{{stds[idx]:.2f}}}}}")
        else:
            if means[idx] is np.nan:
                f.write(" & - ")
            elif means[idx] == 0:
                f.write(f" & \\textbf{{{means[idx]:.2f}}} {{\\tiny$\\pm$ \\textbf{{{stds[idx]:.2f}}}}}")
            elif means[idx] <= -100:
                f.write(f"& $\\dagger$")
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
        if name in ["perm_error", "mse", "times"]:
            best_lambda_idx = np.argmin(res.mean(axis=0), axis=0)
        else:
            best_lambda_idx = np.argmax(res.mean(axis=0), axis=0)

        result_mean = res[:, best_lambda_idx, cols].mean(axis=0)
        result_std = np.std(res[:, best_lambda_idx, cols], axis=0) / np.sqrt(n_samples)
    return result_mean, result_std


def aggregate_mean_std(name, datasets, baselines, dataset_results, Ns_idx):

    mean, std = {}, {}
    for dataset in datasets:

        # citris_baseline = baseline_citris_results["CITRISVAE"]
        baseline = baselines[dataset]
        base_mean, base_std = get_mean_std(name, baseline, N_ids=Ns_idx)

        mean[dataset] = base_mean
        std[dataset] = base_std

        # cit_results = citris_results["CITRISVAE"]
        results = dataset_results[dataset]

        for i in methods_idx:
            if name != "perm_error":
                # TODO: change this when the order is correct again
                if i not in [4, 5]:
                    if i == 6:
                        name_mean, name_std = get_mean_std(name, results, 4, N_ids=Ns_idx)
                    else:
                        name_mean, name_std = get_mean_std(name, results, i, N_ids=Ns_idx)
                else:
                    name_mean, name_std = nan_array(len(Ns)), nan_array(len(Ns))
            else:
                name_mean, name_std = get_mean_std(name, results, i, N_ids=Ns_idx)

            mean[dataset] = np.vstack((mean[dataset], name_mean))
            std[dataset] = np.vstack((std[dataset], name_std))

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
Ns = [5, 10, 20, 100, 1000, 10000]
Ns_idx = [0, 1, 2, 3, 4, 5]

methods = ["NN", "Pearson", "Spearman", "Linear", "Spline", "RFF", "Laplacian", "Two Stage"]
methods_idx = [4, 5, 0, 1, 2, 3, 6]

citris_datasets = ["CITRISVAE", "iVAE"]
dms_datasets = ["DMS-VAE/action", "iVAE/action"]
dms_datasets_temp = ["DMS-VAE/temporal", "TCVAE/temporal"]


metrics = ["perm_error", "r2", "times"]

for met in metrics:
    cit_mean, cit_std = aggregate_mean_std(met, citris_datasets, baseline_citris_results, citris_dataset_results, Ns_idx=Ns_idx)
    dms_mean, dms_std = aggregate_mean_std(met, dms_datasets, baseline_dms_results, dms_dataset_results, Ns_idx=Ns_idx)

    print("TEMP")
    dms_temp_mean, dms_temp_std = aggregate_mean_std(met, dms_datasets_temp, baseline_dms_results, dms_dataset_results, Ns_idx=Ns_idx)

    with open(os.path.join(table_dir, f"appendix_table_{met}.tex"), 'w') as f:
        f.write("Model & ")
        f.write("Method  ")
        for N in Ns:
            f.write(f" & {N}")

        n_methods = len(methods)
    

        f.write("\\\\\n")
        f.write("\\toprule\n")
        f.write("\\bottomrule\n")

        f.write("\\multicolumn{8}{c}{\\rule{0pt}{0.3cm}\\cellcolor{gray!30}\\textbf{Action Sparsity Dataset}}\\\\\n")

        for dataset in dms_mean:
            if met == "perm_error":
                best_per_column = np.argmin(dms_mean[dataset], axis=0)
            else:
                best_per_column = np.argmax(dms_mean[dataset], axis=0)

            for i, method in enumerate(methods):
                if i == 0: 
                    f.write(f"\\multirow{{{n_methods}}}*{{{model_mappings[dataset][0]}}} & ")
                else:
                    f.write(" & ")
                write_both_line(f, method, dms_mean[dataset][i, :], dms_std[dataset][i, :], best_per_column, i)

            f.write("\\hline\n")

        f.write("\\multicolumn{8}{c}{\\rule{0pt}{0.3cm}\\cellcolor{gray!30}\\textbf{Temporal Sparsity Dataset}}\\\\\n")

        for dataset in dms_temp_mean:   
            if met == "perm_error":
                best_per_column = np.argmin(dms_temp_mean[dataset], axis=0)
            else:
                best_per_column = np.argmax(dms_temp_mean[dataset], axis=0)

            for i, method in enumerate(methods):
                if i == 0: 
                    f.write(f"\\multirow{{{n_methods}}}*{{{model_mappings[dataset][0]}}} & ")
                else:
                    f.write(" & ")

                write_both_line(f, method, dms_temp_mean[dataset][i, :], dms_temp_std[dataset][i, :], best_per_column, i)
            f.write("\\hline\n")



        f.write("\\multicolumn{8}{c}{\\rule{0pt}{0.3cm}\\cellcolor{gray!30}\\textbf{Temporal Causal3DIdent Dataset}}\\\\\n")


        for dataset in cit_mean:   
            if met == "perm_error":
                best_per_column = np.argmin(cit_mean[dataset], axis=0)
            else:
                best_per_column = np.argmax(cit_mean[dataset], axis=0)

            for i, method in enumerate(methods):
                if i == 0: 
                    f.write(f"\\multirow{{{n_methods}}}*{{{model_mappings[dataset][0]}}} & ")
                else:
                    f.write(" & ")

                write_both_line(f, method, cit_mean[dataset][i, :], cit_std[dataset][i, :], best_per_column, i)
            f.write("\\hline\n")
