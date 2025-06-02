import os
import pickle 
import numpy as np
import duckdb
import pandas as pd

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
    for idx in range(len(means)):
        best_row = best_per_column[idx]
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
        if name in ["perm_error", "mse", "times", "ois_concept", "nis_concept"]:
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


def aggregate_mean_std(datasets, dataset_results, metric, Ns_idx, methods_idx):
    mean, std = {}, {}
    for dataset in datasets:
        results = dataset_results[dataset]

        if dataset in ["CITRISVAE", "iVAE", "DMS-VAE", "TCVAE"]:
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

def get_best_per_column(results, min=False):
    datasets = [results[model] for model in results]
    data_arr = np.vstack(datasets)
    if min:
        best_per_column = np.argmin(data_arr, axis=0)
    else:
        best_per_column = np.argmax(data_arr, axis=0)
    return best_per_column


def write_tex_block(
        f, methods, 
        mean, std, 
        min
    ):
    methods_map = ["Linear", "Spline", "RFF", "Laplacian", "Two Stage"]
    best_per_column = get_best_per_column(mean, min=min)
    row_idx = 0
    n_methods = len(methods)

    for dataset in mean:

        if dataset in ["CITRISVAE", "iVAE", "DMS-VAE", "TCVAE"]:
            for i, method_idx in enumerate(methods):
                if i == 0: 
                        f.write(f"\\multirow{{{n_methods}}}*{{{dataset}}} & ")
                else:
                    f.write(" & ")

                f.write(f"{{{methods_map[method_idx]}}}")
                write_both_line(f, methods_map[method_idx], mean[dataset][i, :], std[dataset][i, :], best_per_column, row_idx)
                f.write("\\\\\n\n")
                row_idx += 1
        else:
            f.write(f" {dataset } & ")
            f.write(f" {dataset } ")
            write_both_line(f, dataset, mean[dataset], std[dataset], best_per_column, row_idx)
            f.write("\\\\\n\n")
            row_idx += 1
            
        f.write("\\hline\n")

def write_tex_table(
        methods, Ns, 
        cit_mean, cit_std, 
        dms_mean, dms_std, 
        dms_temp_mean, dms_temp_std, 
        fname, 
        min=False):

    with open(os.path.join(table_dir, fname), 'w') as f:
        f.write("Model & ")
        f.write("Method  ")
        for N in Ns:
            f.write(f" & {N}")

        n_methods = len(methods)


        f.write("\\\\\n")
        f.write("\\toprule\n")
        f.write("\\bottomrule\n")

        f.write("\\multicolumn{6}{c}{\\rule{0pt}{0.3cm}\\cellcolor{gray!30}\\textbf{Action Sparsity Dataset}}\\\\\n")
        write_tex_block(f, methods, dms_mean, dms_std, min)

        f.write("\\multicolumn{6}{c}{\\rule{0pt}{0.3cm}\\cellcolor{gray!30}\\textbf{Temporal Sparsity Dataset}}\\\\\n")
        write_tex_block(f, methods, dms_temp_mean, dms_temp_std, min)

        f.write("\\multicolumn{6}{c}{\\rule{0pt}{0.3cm}\\cellcolor{gray!30}\\textbf{Temporal Causal3DIdent Dataset}}\\\\\n")
        write_tex_block(f, methods, cit_mean, cit_std, min)




def write_double_tex_block(f, methods, mean_l, std_l, mean_r, std_r, min_l, min_r):
    cite_map = {
        "CBM": "CBM \\citep{koh2020conceptbottleneck}",
        "CEM": "CEM \\citep{zarlenga2022concept}",
        "CBM AR": "HardCBM \\citep{havasi2022addressing}",
    }

    methods_map = ["Linear", "Spline", "RFF", "Laplacian", "Two Stage"]
    best_per_column_l = get_best_per_column(mean_l, min=min_l)
    best_per_column_r = get_best_per_column(mean_r, min=min_r)
    row_idx = 0
    n_methods = len(methods)

    for dataset in mean_l:

        if dataset in ["CITRISVAE", "iVAE", "DMS-VAE", "TCVAE"]:
            for i, method_idx in enumerate(methods):
                if i == 0: 
                        f.write(f"\\multirow{{{n_methods}}}*{{{dataset}}} & ")
                else:
                    f.write(" & ")

                f.write(f"{{{methods_map[method_idx]}}}")
                write_both_line(f, methods_map[method_idx], mean_l[dataset][method_idx, :], std_l[dataset][method_idx, :], best_per_column_l, row_idx)
                write_both_line(f, methods_map[method_idx], mean_r[dataset][method_idx, :], std_r[dataset][method_idx, :], best_per_column_r, row_idx)

                f.write("\\\\\n\n")
                row_idx += 1
            f.write("\\hline\n")
        else:
            # f.write(f" \\multicolumn{{2}}{{c}}{{\\flushleft {cite_map[dataset]}}} ")
            f.write(f" {cite_map[dataset]} &  ")
            write_both_line(f, dataset, mean_l[dataset], std_l[dataset], best_per_column_l, row_idx)
            write_both_line(f, dataset, mean_r[dataset], std_r[dataset], best_per_column_r, row_idx)

            f.write("\\\\\n\n")
            row_idx += 1
    
    f.write("\\bottomrule")

def write_double_tex_table(
        methods, Ns, 
        cit_mean_l, cit_std_l, cit_mean_r, cit_std_r, 
        dms_mean_l, dms_std_l, dms_mean_r, dms_std_r, 
        dms_temp_mean_l, dms_temp_std_l, dms_temp_mean_r, dms_temp_std_r, 
        fname, 
        min_l=False, min_r=False):

    with open(os.path.join(table_dir, fname), 'w') as f:
        f.write("Model & ")
        f.write("Method  ")
        for N in Ns:
            f.write(f" & {N}")
        for N in Ns:
            f.write(f" & {N}")


        f.write("\\\\\n")
        f.write("\\toprule\n")
        f.write("\\bottomrule\n")

        f.write("\\multicolumn{10}{c}{\\rule{0pt}{0.3cm}\\cellcolor{gray!30}\\textbf{Action Sparsity Dataset}}\\\\\n")
        write_double_tex_block(f, methods, dms_mean_l, dms_std_l, dms_mean_r, dms_std_r, min_l, min_r)

        f.write("\\multicolumn{10}{c}{\\rule{0pt}{0.3cm}\\cellcolor{gray!30}\\textbf{Temporal Sparsity Dataset}}\\\\\n")
        write_double_tex_block(f, methods, dms_temp_mean_l, dms_temp_std_l, dms_temp_mean_r, dms_temp_std_r, min_l, min_r)

        f.write("\\multicolumn{10}{c}{\\rule{0pt}{0.3cm}\\cellcolor{gray!30}\\textbf{Temporal Causal3DIdent Dataset}}\\\\\n")
        write_double_tex_block(f, methods, cit_mean_l, cit_std_l, cit_mean_r, cit_std_r, min_l, min_r)




def pull_data(concept_dir, results, model_type, db_str, dataset=None):
    if "cbm" in db_str or "cem" in db_str:
        if dataset is None:
            method_cols = "N,"
            condition_str = ""
            n_method = 5
        else:
            method_cols = f"N, dataset,"
            n_method = 4
            condition_str = f"WHERE dataset='{dataset}'"
    else:
        if dataset is None:
            method_cols = "method, alpha, N, perm_error,"
            n_method = 5
            condition_str = ""
        else:
            method_cols = f"method, alpha, N, dataset, perm_error,"
            n_method = 5
            condition_str = f"WHERE dataset='{dataset}'"


    with duckdb.connect(os.path.join(concept_dir, f"{db_str}.duckdb")) as con:
        result_df = con.sql(f"""
            SELECT {method_cols}
                acc_label, roc_label,
                acc_concept, roc_concept,
                ois_concept, nis_concept,
                time, 
                seed,
            FROM experiments 
            {condition_str}
            ;
        """).df()
            # WHERE method in ('Linear', 'RFF', 'Spline', 'Laplacian', 'two_stage');

        results[model_type] = {}
        if "cbm" in db_str or "cem" in db_str:
            result_df = result_df.sort_values(["seed", "N"])
            results[model_type]["acc_label"] = result_df["acc_label"].to_numpy().reshape(10, 4)
            results[model_type]["roc_label"] = result_df["roc_label"].to_numpy().reshape(10, 4)

            results[model_type]["acc_concept"] = result_df["acc_concept"].to_numpy().reshape(10, 4)
            results[model_type]["roc_concept"] = result_df["roc_concept"].to_numpy().reshape(10, 4)

            results[model_type]["ois_concept"] = result_df["ois_concept"].to_numpy().reshape(10, 4)
            results[model_type]["nis_concept"] = result_df["nis_concept"].to_numpy().reshape(10, 4)
        else:
            result_df["method"] = pd.Categorical(
                result_df["method"], 
                ["Linear", "Spline", "RFF", "Laplacian", "two_stage"]
                ).rename_categories({"two_stage": "Two Stage"})
            result_df = result_df.sort_values(["seed", "method", "alpha", "N"])
            results[model_type]["acc_label"] = result_df["acc_label"].to_numpy().reshape(10, n_method, -1, 4)
            results[model_type]["roc_label"] = result_df["roc_label"].to_numpy().reshape(10, n_method, -1, 4)

            results[model_type]["acc_concept"] = result_df["acc_concept"].to_numpy().reshape(10, n_method, -1, 4)
            results[model_type]["roc_concept"] = result_df["roc_concept"].to_numpy().reshape(10, n_method, -1, 4)

            results[model_type]["ois_concept"] = result_df["ois_concept"].to_numpy().reshape(10, n_method, -1, 4)
            results[model_type]["nis_concept"] = result_df["nis_concept"].to_numpy().reshape(10, n_method, -1, 4)

            results[model_type]["perm_error"] = result_df["perm_error"].to_numpy().reshape(10, n_method, -1, 4)

    return results


concept_dirs = [
    "CITRIS_bin/cluster_checkpoints",
    "DMS_VAE_bin/cluster_checkpoints"
]

table_dir = "tables"
Ns = [20, 100, 1000, 10000]
Ns_idx = [0, 1, 2, 3]

methods = ["Linear", "Spline", "RFF", "Laplacian", "Two Stage"]
methods_idx = [0, 1, 2, 3, 4]

datasets = ["CITRISVAE", "iVAE", "CBM", "CEM", "CBM AR"]

results_citris = {}
results_citris = pull_data(concept_dirs[0], results_citris, "iVAE", "experiment_ivae")
results_citris = pull_data(concept_dirs[0], results_citris, "CITRISVAE", "experiment_citris")
results_citris = pull_data(concept_dirs[0], results_citris, "CBM", "experiment_cbm")
results_citris = pull_data(concept_dirs[0], results_citris, "CEM", "experiment_cem")
results_citris = pull_data(concept_dirs[0], results_citris, "CBM AR", "experiment_cbm_ar")

results_dms = {}
results_dms = pull_data(concept_dirs[1], results_dms, "DMS-VAE", "experiment_dms", dataset="action")
results_dms = pull_data(concept_dirs[1], results_dms, "iVAE", "experiment_ivae", dataset="action")
results_dms = pull_data(concept_dirs[1], results_dms, "CBM", "experiment_cbm", dataset="action")
results_dms = pull_data(concept_dirs[1], results_dms, "CEM", "experiment_cem", dataset="action")
results_dms = pull_data(concept_dirs[1], results_dms, "CBM AR", "experiment_cbm_ar", dataset="action")

results_dms_temp = {}
results_dms_temp = pull_data(concept_dirs[1], results_dms_temp, "DMS-VAE", "experiment_dms", dataset="temporal")
results_dms_temp = pull_data(concept_dirs[1], results_dms_temp, "TCVAE", "experiment_ivae", dataset="temporal")
results_dms_temp = pull_data(concept_dirs[1], results_dms_temp, "CBM", "experiment_cbm", dataset="temporal")
results_dms_temp = pull_data(concept_dirs[1], results_dms_temp, "CEM", "experiment_cem", dataset="temporal")
results_dms_temp = pull_data(concept_dirs[1], results_dms_temp, "CBM AR", "experiment_cbm_ar", dataset="temporal")

# results = vae_results
# results["CBM"] = cbm_results["CBM"]
# results["CEM"] = cbm_results["CEM"]
mertic_pairs = [
    ("acc_concept", "acc_label", False), 
    ("ois_concept", "nis_concept", True), 
    ] 

for i, (met_1, met_2, b) in enumerate(mertic_pairs):

    datasets = ["CITRISVAE", "iVAE", "CBM", "CEM", "CBM AR"]
    cit_mean_met_1, cit_std_met_1 = aggregate_mean_std(datasets, results_citris, met_1, Ns_idx=Ns_idx, methods_idx=methods_idx)
    cit_mean_met_2, cit_std_met_2 = aggregate_mean_std(datasets, results_citris, met_2, Ns_idx=Ns_idx, methods_idx=methods_idx)

    datasets = ["DMS-VAE", "iVAE", "CBM", "CEM", "CBM AR"]
    dms_mean_met_1, dms_std_met_1 = aggregate_mean_std(datasets, results_dms, met_1, Ns_idx=Ns_idx, methods_idx=methods_idx)
    dms_mean_met_2, dms_std_met_2 = aggregate_mean_std(datasets, results_dms, met_2, Ns_idx=Ns_idx, methods_idx=methods_idx)

    datasets = ["DMS-VAE", "TCVAE", "CBM", "CEM", "CBM AR"]
    dms_temp_mean_met_1, dms_temp_std_met_1 = aggregate_mean_std(datasets, results_dms_temp, met_1, Ns_idx=Ns_idx, methods_idx=methods_idx)
    dms_temp_mean_met_2, dms_temp_std_met_2 = aggregate_mean_std(datasets, results_dms_temp, met_2, Ns_idx=Ns_idx, methods_idx=methods_idx)


    # Concept acc & ROC AUC
    write_double_tex_table(
        methods_idx, Ns, 
        cit_mean_met_1, cit_std_met_1, 
        cit_mean_met_2, cit_std_met_2,
        dms_mean_met_1, dms_std_met_1,
        dms_mean_met_2, dms_std_met_2,
        dms_temp_mean_met_1, dms_temp_std_met_1,
        dms_temp_mean_met_2, dms_temp_std_met_2,
        f"table_bin_{i}.tex",
        min_l=b, min_r=b
    )


datasets = ["CITRISVAE", "iVAE"]
cit_mean_perm_error, cit_std_perm_error = aggregate_mean_std(datasets, results_citris, "perm_error", Ns_idx=Ns_idx, methods_idx=methods_idx)

datasets = ["DMS-VAE", "iVAE"]
dms_mean_perm_error, dms_std_perm_error = aggregate_mean_std(datasets, results_dms, "perm_error", Ns_idx=Ns_idx, methods_idx=methods_idx)

datasets = ["DMS-VAE", "TCVAE"]
dms_temp_mean_perm_error, dms_temp_std_perm_error = aggregate_mean_std(datasets, results_dms_temp, "perm_error", Ns_idx=Ns_idx, methods_idx=methods_idx)

write_tex_table(
    methods_idx, Ns, 
    cit_mean_perm_error, cit_std_perm_error, 
    dms_mean_perm_error, dms_std_perm_error,
    dms_temp_mean_perm_error, dms_temp_std_perm_error,
    f"table_bin_{2}.tex",
    min=True
)



    