import os
from itertools import product
from typing import List
from argparse import ArgumentParser

from joblib import Parallel, delayed
import time

import duckdb
import numpy as np

from experiment.rff_experiment import do_rff_experiment_bin
from experiment.rff_settings import parallel_params


def write_result(list_dicts):
    with duckdb.connect("data/experiments_binary_ablation.duckdb") as con:
        con.executemany("""
            INSERT OR REPLACE INTO experiments_features 
            VALUES (
                $regularizer,
                $n_total,
                $test_frac,
                $d_variables,
                $p_features,
                $entanglement,
                $alpha,
                $seed,
                $perm_error_match,
                $acc_label, 
                $roc_label, 
                $acc_concept, 
                $roc_concept, 
                $ois_concept,
                $nis_concept,
                $time_match,
                current_timestamp
            )
            """,
            list_dicts
        )
        con.table("experiments_features").show(max_rows=5)


def write_results_to_db(experiment_results):
    # dict_lists = {k: [dic[k] for dic in experiment_results] for k in experiment_results[0]}
    while True:
        try:
            print("Writing to Database")
            write_result(list_dicts=experiment_results)
            break
        except Exception as e:
            print(e)
            print("Connection failed, waiting 30 seconds")
            time.sleep(np.random.randint(25, 35))


def check_params(
    parallel_params: set,
    start_date: str="2025-05-01",
    end_date: str="2025-05-30",
) -> List:
    while True:
        # Check which settings have already been run
        try:
            with duckdb.connect("data/experiments_binary_ablation.duckdb") as con:
                def round_dec(x: float) -> float:
                    return round(x, 4)

                con.create_function("round_dec", round_dec)
                keys = con.execute(f"""
                    SELECT DISTINCT 
                        seed, 
                        p_features, 
                        d_variables,
                        round_dec(alpha), 
                        round_dec(entanglement),
                        n_total
                    FROM experiments_features
                    WHERE date_trunc('day', performed) >= '{start_date}'
                    AND date_trunc('day', performed) <= '{end_date}';
                    """).fetchall()
                keys = set(keys)
                break
        except Exception as e:
            print(e)
            time.sleep(30)           

    print(f"nr of params before filter {len(parallel_params)}")
    parallel_params = parallel_params - keys
    parallel_params = list(parallel_params)
    print(f"nr of params after filter {len(parallel_params)}")
    return parallel_params

def add_seed(parallel_params: List, seeds: List) -> List:
    modified_list = [
        (seed, p, d, a, e, n) 
        for p, d, a, e, n in parallel_params 
        for seed in seeds
    ]
    return list(set(modified_list))


if __name__ == '__main__':   
    parser = ArgumentParser()
    parser.add_argument('--check', type=int, choices=[0, 1],
                        default=1,
                        help='If the params need to be checked')

    args = parser.parse_args()
    CHECK = args.check

    REPEATS = 10
    seeds=list(range(100, 100 + REPEATS)) 

    parallel_params = add_seed(parallel_params, seeds)

    if CHECK == 1:
        parallel_params = check_params(set(parallel_params))
    else:
        print(f"nr of params {len(parallel_params)}")

    # num_workers = os.sched_getaffinity(0)
    num_workers = [0, 0, 0, 0]
    print(f"Num of workers = {len(num_workers)}")
    print(f"Num of jobs = {len(parallel_params)}")

    start = time.time()
    all_results = Parallel(n_jobs=len(num_workers), verbose=10)(
        delayed(do_rff_experiment_bin)(
            n_total=n, 
            test_frac=0.5,
            d_variables=d,
            p_features=p,
            alpha=a,
            regularizer="logistic_group",
            entanglement=e,
            seed=seed
        ) for seed, p, d, a, e, n in parallel_params
    )

    write_results_to_db(experiment_results=all_results)
    stop = time.time()

    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(stop - start))
    print(f"--- Elapsed time running in parallel {elapsed_str} ---")
