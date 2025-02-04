from itertools import product
from typing import List
from argparse import ArgumentParser

import multiprocessing 
import time

import duckdb
import numpy as np

from experiment.linear_experiment import do_linear_experiment 
from experiment.linear_settings import series_params 


class Consumer(multiprocessing.Process):

    def __init__(self, task_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        proc_name = self.name
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means shutdown
                print(f"{proc_name}: Exiting")
                self.task_queue.task_done()
                break

            result = next_task()
            self.result_queue.put(result)

            self.task_queue.task_done()
        return


class Task(object):
    def __init__(
            self, 
            n_total, 
            test_frac,
            d_variables,
            alpha,
            regularizer,
            entanglement,
            spec,
            seeds
        ):
        self.n_total      = n_total
        self.test_frac    = test_frac
        self.d_variables  = d_variables
        self.alpha        = alpha
        self.regularizer  = regularizer
        self.entanglement = entanglement
        self.spec         = spec
        self.seed         = seeds
        
    def __call__(self):
        results = [0] * len(self.seed)
        for i, seed in enumerate(self.seed):
            results[i] = do_linear_experiment(
                    n_total=self.n_total, 
                    test_frac=self.test_frac,
                    d_variables=self.d_variables,
                    alpha=self.alpha,
                    regularizer=self.regularizer,
                    entanglement=self.entanglement,
                    spec=self.spec,
                    seed=seed
                )
        return results
        
    def __str__(self):
        name = f"""Running experiment with the following parameters:
    - {self.n_total} data points with test fraction {self.test_frac}
    - {self.regularizer} regularizer with parameter {self.alpha}
    - {self.d_variables} dimensions
    - {self.entanglement} correlation in the data
    - {self.seed} random seed"""
        return name


def write_result(list_dicts):
    with duckdb.connect("data/experiments.duckdb") as con:
        con.executemany("""
            INSERT OR REPLACE INTO experiments_linear
            VALUES (
                $regularizer,
                $n_total,
                $test_frac,
                $d_variables,
                $entanglement,
                $alpha,
                $miss_well,
                $seed,
                $perm_error_match,
                $perm_error_corr,
                $perm_error_spear,
                $mse_match,
                $r2_match,
                $time_match,
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


def check_params(series_params: set) -> List:
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
                        round_dec(alpha), 
                        round_dec(entanglement),
                        miss_well,
                        n_total
                    FROM experiments_linear;
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
    
    start = time.time()
    tasks = multiprocessing.JoinableQueue()
    results = multiprocessing.JoinableQueue()

    num_consumers = multiprocessing.cpu_count() * 2
    print(f"Creating {num_consumers} consumers")
    consumers = [Consumer(tasks, results) for _ in range(num_consumers)]
    for w in consumers:
        w.start()

    experiment_results = []

    for r, d, a, e, m, n in series_params:   
        m = "well" if m is True else "miss"

        tasks.put(Task(
            n_total=n, 
            test_frac=0.2,
            d_variables=d,
            alpha=a,
            regularizer=r,
            entanglement=e,
            spec=m,
            seeds=seeds
        ))

    # Add a poison pill for each consumer
    for i in range(num_consumers):
        tasks.put(None)

    all_jobs = len(series_params)
    num_jobs = len(series_params) 
    print(f"Nr of experiments {num_jobs * len(seeds)}")
    while num_jobs > 0:
        experiment_results.extend(results.get())
        num_jobs -= 1
        print(f" Nr of jobs left is {num_jobs} of {all_jobs}")

        if len(experiment_results) >= 1000:
            write_results_to_db(experiment_results=experiment_results)
            experiment_results = []
            
    tasks.join()

    stop = time.time()
    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(stop - start))

    print(f"--- Elapsed time running in parallel {elapsed_str} ---")
    write_results_to_db(experiment_results=experiment_results)
