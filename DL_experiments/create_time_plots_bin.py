import os
import duckdb
import pandas as pd

from shared_utils.plot_utils import make_time_plot_bin

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
            results[model_type]["times"] = result_df["time"].to_numpy().reshape(10, 4)
        else:
            result_df["method"] = pd.Categorical(
                result_df["method"], 
                ["Linear", "Spline", "RFF", "Laplacian", "two_stage"]
                ).rename_categories({"two_stage": "Two Stage"})
            result_df = result_df.sort_values(["seed", "method", "alpha", "N"])
            results[model_type]["times"] = result_df["time"].to_numpy().reshape(10, n_method, -1, 4)

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
# results_citris = pull_data(concept_dirs[0], results_citris, "iVAE", "experiment_ivae")
results_citris = pull_data(concept_dirs[0], results_citris, "CITRISVAE", "experiment_citris")

baseline_citris = {}
baseline_citris = pull_data(concept_dirs[0], baseline_citris, "CBM", "experiment_cbm")
baseline_citris = pull_data(concept_dirs[0], baseline_citris, "CEM", "experiment_cem")
baseline_citris = pull_data(concept_dirs[0], baseline_citris, "CBM AR", "experiment_cbm_ar")

make_time_plot_bin(
    results_citris,
    baseline_citris,
    Ns=Ns, 
    ckpt_dir="./",
    fname="citris"
)

results_dms = {}
results_dms = pull_data(concept_dirs[1], results_dms, "DMS-VAE", "experiment_dms", dataset="action")
# results_dms = pull_data(concept_dirs[1], results_dms, "iVAE", "experiment_ivae", dataset="action")

baseline_dms = {}
baseline_dms = pull_data(concept_dirs[1], baseline_dms, "CBM", "experiment_cbm", dataset="action")
baseline_dms = pull_data(concept_dirs[1], baseline_dms, "CEM", "experiment_cem", dataset="action")
baseline_dms = pull_data(concept_dirs[1], baseline_dms, "CBM AR", "experiment_cbm_ar", dataset="action")

make_time_plot_bin(
    results_dms,
    baseline_dms,
    Ns=Ns, 
    ckpt_dir="./",
    fname="dms"
)