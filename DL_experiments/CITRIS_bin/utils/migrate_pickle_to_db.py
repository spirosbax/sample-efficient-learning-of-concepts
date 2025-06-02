import duckdb
import pickle
import os

def migrate_results_estimator(db_str, result):
    experiments_list = []
    print(result.keys())
    for i in range(len(seed)):
        for j in range(len(methods)):
            for k in range(len(alphas)):
                for l in range(len(Ns)):
                    one_experiment_res = {
                        "method": methods[j],
                        "alpha": alphas[k],
                        "N": Ns[l],
                        "acc_label": result["acc_label"][i, j, k, l],
                        "roc_label": result["roc_label"][i, j, k, l],
                        "acc_concept": result["acc_concept"][i, j, k, l],
                        "roc_concept": result["roc_concept"][i, j, k, l],
                        "ois_score": result["ois_concept"][i, j, k, l],
                        "nis_score": result["nis_concept"][i, j, k, l],
                        "time": result["times"][i, j, k, l],
                        "seed": seed[i]
                    }

                    experiments_list.append(one_experiment_res)
    with duckdb.connect(f"checkpoints/{db_str}.duckdb") as con:
        con.executemany("""
            INSERT OR REPLACE INTO experiments
            VALUES (
                $method,
                $alpha,
                $N, 
                $acc_label,
                $roc_label,
                $acc_concept,
                $roc_concept,
                $ois_score,
                $nis_score,
                $time,
                $seed,
                current_timestamp
            )
            """,
            experiments_list
        )
        con.table("experiments").show(max_rows=5)

def migrate_results_cbm(db_str, result):
    experiments_list = []
    for i in range(len(seed)):
        for j in range(len(Ns)):
            one_experiment_res = {
                "N": Ns[j],
                "acc_label": result["acc_label"][i, j],
                "roc_label": result["roc_label"][i, j],
                "acc_concept": result["acc_concept"][i, j],
                "roc_concept": result["roc_concept"][i, j],
                "ois_score": result["ois_concept"][i, j],
                "nis_score": result["nis_concept"][i, j],
                "time": result["times"][i, j],
                "seed": seed[i]
            }

            experiments_list.append(one_experiment_res)

    with duckdb.connect(f"checkpoints/{db_str}.duckdb") as con:
        con.executemany("""
            INSERT OR REPLACE INTO experiments
            VALUES (
                $N,
                $acc_label,
                $roc_label,
                $acc_concept,
                $roc_concept,
                $ois_score,
                $nis_score,
                $time,
                $seed,
                current_timestamp
            )
            """,
            experiments_list
        )
        con.table("experiments").show(max_rows=5)



concept_dir = "checkpoints"
Ns = [20, 100, 1000, 10000]
Ns_idx = [0, 1, 2, 3]

methods = ["Linear", "Spline", "RFF"]
methods_idx = [0, 1, 2]

alphas = [
        0.0001, 0.0005, 
        0.001, 0.005, 
        0.01, 0.05, 0.1, 0.2 
    ]

seed = list(range(100, 110))

with open(os.path.join(concept_dir, "results_ivae.pickle"), "rb") as f:
    result = pickle.load(f)["iVAE"]
migrate_results_estimator("experiment_ivae", result)
with open(os.path.join(concept_dir, "results_citris.pickle"), "rb") as f:
    result = pickle.load(f)["CITRISVAE"]
migrate_results_estimator("experiment_citris", result)

with open(os.path.join(concept_dir, "results_cem.pickle"), "rb") as f:
    result = pickle.load(f)["CEM"]
migrate_results_cbm("experiment_cem", result)
with open(os.path.join(concept_dir, "results_cbm.pickle"), "rb") as f:
    result = pickle.load(f)["CBM"]
migrate_results_cbm("experiment_cbm", result)
with open(os.path.join(concept_dir, "results_cbm_ar.pickle"), "rb") as f:
    result = pickle.load(f)["HardCBM"]
migrate_results_cbm("experiment_cbm_ar", result)


