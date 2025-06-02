import os
from itertools import product
from typing import List
from argparse import ArgumentParser

from joblib import Parallel, delayed
import time

import duckdb
import numpy as np

from experiment.spline_experiment import do_spline_experiment
from experiment.spline_settings import series_params 


def write_result(list_dicts):
    with duckdb.connect("data/experiments.duckdb") as con:
        con.executemany("""
            INSERT OR REPLACE INTO experiments_spline
            VALUES (
                $regularizer,
                $n_total,
                $test_frac,
                $d_variables,
                $n_knots,
                $degree,
                $entanglement,
                $alpha,
                $miss_well,
                $seed,
                $perm_error_match,
                $perm_error_linear,
                $perm_error_corr,
                $perm_error_spear,
                $mse_match,
                $mse_linear,
                $r2_match,
                $r2_linear,
                $time_match,
                $time_linear,
                $time_corr,
                $time_spear,
                current_timestamp
            )
            """,
            list_dicts
        )


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
    series_params: set,
    start_date: str="2025-01-28",
    end_date: str="2025-01-31",
) -> List:
    while True:
        # Check which settings have already been run
        try:
            with duckdb.connect("data/experiments.duckdb") as con:
                def round_dec(x: float) -> float:
                    return round(x, 4)

                con.create_function("round_dec", round_dec)
                keys = con.execute(f"""
                    SELECT DISTINCT regularizer, 
                        d_variables,
                        n_knots, 
                        degree, 
                        round_dec(alpha), 
                        round_dec(entanglement),
                        miss_well,
                        n_total
                    FROM experiments_spline
                    WHERE date_trunc('day', performed) >= '{start_date}'
                    AND date_trunc('day', performed) <= '{end_date}';
                    """).fetchall()
                keys = set(keys)
                break
        except Exception as e:
            print(e)
            time.sleep(30)           
    
    print(f"nr of params before filter {len(series_params)}")
    series_params = series_params - keys
    series_params = list(series_params)
    print(f"nr of params after filter {len(series_params)}")
    return series_params


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--check', type=int, choices=[0, 1],
                        default=1,
                        help='If the params need to be checked')

    args = parser.parse_args()
    CHECK = args.check

    seeds=[120, 121, 122, 123, 124, 125, 126, 127, 128, 129]

    if CHECK == 1:
        series_params = check_params(series_params)
    else:
        series_params = list(series_params)
        print(f"nr of params {len(series_params)}")

    num_workers = os.sched_getaffinity(0)
    print(f"Num of workers = {len(num_workers)}")
    print(f"Num of jobs = {len(series_params) * len(seeds)}")
    
    start = time.time()
    all_results = Parallel(n_jobs=len(num_workers), verbose=5)(
        delayed(do_spline_experiment)(
            n_total=n,
            test_frac=0.2,
            d_variables=d,
            n_knots=k,
            degree=deg,
            alpha=a,
            regularizer=r,
            entanglement=e,
            spec="well" if m else "miss",
            seed=seed
        ) 
        for r, d, k, deg, a, e, m, n in series_params
        for seed in seeds
    )
    write_results_to_db(experiment_results=all_results)
    stop = time.time()

    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(stop - start))
    print(f"--- Elapsed time running in parallel {elapsed_str} ---")
    