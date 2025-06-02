"""
Checking if random fourier features can be learned with onl (2d) points
"""
import sys
import os
import warnings
sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], './../DL_experiments'))
sys.path.insert(1, os.path.join(sys.path[0], '../..'))

import duckdb
import time

import torch
import torch.nn.functional as F
import torch.utils.data as data

import numpy as np
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter("ignore", category=ConvergenceWarning)

from DL_experiments.shared_utils.concept_models.cbm.lightning_module import SimpleCBM
from DL_experiments.shared_utils.concept_models.cem.lightning_module import CEM
from DL_experiments.shared_utils.concept_models.hardcbm.lightning_module import HardCBM
from DL_experiments.shared_utils.concept_models.hardcbm.train import train


from DL_experiments.CITRIS_bin.training.datasets import ConceptDataset
from pytorch_lightning import Trainer


from utils.utils import (
    get_basic_parser,
    set_seed, 
)
from utils.experiment import *
from permutation_estimator.estimator import FeaturePermutationEstimator


def get_device():
    if torch.backends.mps.is_available():
        # For small networks and loads this cpu seems to be faster 
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    return device

def do_cbm_experiment(
    n_total: int, 
    test_frac: float, 
    d_variables: int,
    entanglement: float,
    model_type: str,
    seed: int
):
    device = get_device()
    N_TRAIN = int(n_total * (1 - test_frac))
    N_TEST = int(n_total * test_frac)

    set_seed(seed)

    x_train, _ = sample_x_data(
        dim=d_variables, 
        n_train=N_TRAIN, 
        n_test=N_TEST, 
        entanglement=entanglement
    )

    _, x_test = sample_x_data(
        dim=d_variables, 
        n_train=N_TRAIN, 
        n_test=N_TEST, 
        entanglement=0
    )

    permutation = np.random.choice(
        size=d_variables, 
        a=np.arange(d_variables), 
        replace=False
    )

    concept_train, concept_test = sample_concept_labels(
        x_train, x_test,
        dim=d_variables, 
        permutation=permutation,
    )

    y_train, y_test = create_labels(concept_train, concept_test)
    y_train, y_test = y_train[:, np.newaxis], y_test[:, np.newaxis]

    arrays = sample_correctly(
        np.hstack((x_train, x_test)), 
        np.hstack((concept_train, concept_test)), 
        np.vstack((y_train, y_test)), 
        n_total=N_TRAIN + N_TEST,
        N_train=N_TRAIN
    )
    x_train, x_test = torch.from_numpy(arrays["train_encs"]).to(torch.float32), torch.from_numpy(arrays["test_encs"]).to(torch.float32)
    concept_train, concept_test = torch.from_numpy(arrays["train_latents"]).to(torch.float32), torch.from_numpy(arrays["test_latents"]).to(torch.float32)
    y_train, y_test = torch.from_numpy(arrays["train_y"]).to(torch.float32), torch.from_numpy(arrays["test_y"]).to(torch.float32)


    train_dataset = ConceptDataset(x_train.T, concept_train.T, y_train, tabular=True, indexed=False)
    test_dataset = ConceptDataset(x_test.T, concept_test.T, y_test, tabular=True, indexed=False)

    train_loader = data.DataLoader(
        train_dataset, 
        batch_size=256,
        shuffle=False,
        drop_last=False, 
        num_workers=0
    )
    test_loader = data.DataLoader(
        test_dataset, 
        batch_size=256,
        shuffle=False,
        drop_last=False, 
        num_workers=0
    )

    if model_type == "CBM":
        cbm = SimpleCBM(
            c_hid=32,
            c_in=d_variables, 
            num_concepts=d_variables, 
            num_output=1,
            regression=False,
            conv=False
        )
    elif model_type == "CEM":
        cbm = CEM(
            c_hid=32,
            c_in=d_variables, 
            num_concepts=d_variables,
            num_output=1, 
            regression=False,
            conv=False
        )
    elif model_type == "HardCBM":
        cbm = HardCBM(
            c_hid=32, 
            c_in=d_variables, 
            num_concepts=d_variables,
            num_output=1,
            conv=False
        ).to(device)

    trainer = Trainer(max_epochs=100, enable_progress_bar=False)
    
    if model_type == "HardCBM":
        start = time.time()
        cbm = train(
            model=cbm, 
            train_loader=train_loader, 
            val_loader=test_loader, 
            test_loader=test_loader, 
            epochs=100, 
            validate_per_epoch=1000
        )
        end = time.time()
    else:
        start = time.time()
        trainer.fit(cbm, train_loader, test_loader)
        end = time.time()

    cbm.eval()

    all_c = []
    all_c_hat = []
    all_c_emb = []

    all_y = []
    all_y_hat = []
    for batch in test_loader:
        x, c, y = batch
        if model_type == "CBM":
            c_hat = cbm.concept_encoder(x.to(torch.float32).to(cbm.device))
            y_hat = cbm.label_predictor(c_hat).detach().cpu()
        elif model_type == "CEM":
            c_emb, c_hat = cbm.concept_encoder(x.to(cbm.device))
            y_hat = cbm.label_predictor(c_emb.reshape(len(c_emb), -1)).detach().cpu()
            all_c_emb.append(c_emb)
        else:
            c_hat, y_hat, _ = cbm.forward(
                x=x.to(device), 
                epoch=1, 
                validation=True
            )
            y_hat = y_hat.detach().cpu()
            c_hat = torch.mean(c_hat, dim=-1)

        all_c_hat.append(c_hat.detach().cpu())
        all_c.append(c)

        all_y_hat.append(y_hat)
        all_y.append(y)

    all_c = torch.cat(all_c, dim=0)
    all_c_hat_score = torch.cat(all_c_hat, dim=0)

    all_y = torch.cat(all_y, dim=0)
    all_y_hat_score = torch.cat(all_y_hat, dim=0)

    acc_concept = accuracy_score(all_c.reshape(-1, 1), (all_c_hat_score > 0.5).reshape(-1, 1))
    roc_concept = roc_auc_score(all_c, all_c_hat_score, multi_class="ovr")

    acc_label = accuracy_score(all_y, all_y_hat_score > 0)
    roc_label = roc_auc_score(all_y, all_y_hat_score, multi_class="ovr")

    if model_type == "CBM" or model_type == "HardCBM":
        if d_variables <= 20:
            ois_concept = ois_score(all_c, all_c_hat_score, tensor=True)
            nis_concept = nich_impurity_score(all_c, all_c_hat_score, tensor=True)
        else:
            ois_concept = 2
            nis_concept = 2
    else:
        if d_variables <= 20:
            all_c_emb = torch.cat(all_c_emb, dim=0)
            ois_concept = ois_score(all_c, all_c_emb, tensor=True)
            nis_concept = nich_impurity_score(all_c, all_c_emb, tensor=True)
        else:
            ois_concept = 2
            nis_concept = 2


    result = {
        "train_corr": entanglement,
        "acc_concept": acc_concept,
        "roc_concept": roc_concept,
        "acc_label": acc_label,
        "roc_label": roc_label,
        "ois_score": ois_concept,
        "nis_score": nis_concept,
        "time": end - start
    }
    result["N"] = N_TRAIN
    result["seed"] = seed
    result["d_variables"] = d_variables
    print(result)
    return result


def write_to_db(db_string, results):
    with duckdb.connect(f"data/{db_string}.duckdb") as con:
        con.executemany(f"""
            INSERT OR REPLACE INTO experiments
            VALUES (
                $N,
                $d_variables,
                $train_corr,
                $seed,
                $acc_label,
                $roc_label,
                $acc_concept,
                $roc_concept,
                $ois_score,
                $nis_score,
                $time,
                current_timestamp
            )
            """,
        results)
        con.table("experiments").show(max_rows=5)



REPEATS = 10

seeds = list(range(100, 100 + REPEATS))

N_TOTAL = [100, 200, 2000, 4000, 10000]
d_variables = [10, 20, 30] #  40, 50]


for dim in d_variables:
    results_cbm = []
    results_cem = []
    results_cbm_ar = []
    for seed in seeds:
        for N in N_TOTAL:
            result_cbm = do_cbm_experiment(
                N, 
                test_frac=0.5, 
                d_variables=dim,
                entanglement=0, 
                model_type="CBM", 
                seed=seed
            )
            results_cbm.append(result_cbm)

            result_cem = do_cbm_experiment(
                N, 
                test_frac=0.5, 
                d_variables=dim,
                entanglement=0, 
                model_type="CEM", 
                seed=seed
            )
            results_cem.append(result_cem)

            result_cbm_ar = do_cbm_experiment(
                N, 
                test_frac=0.5, 
                d_variables=dim,
                entanglement=0, 
                model_type="HardCBM", 
                seed=seed
            )
            results_cbm_ar.append(result_cbm_ar)

    print(f"writing results for dimension: {dim}")
    write_to_db("experiments_cbm", results_cbm)   
    write_to_db("experiments_cem", results_cem)   
    write_to_db("experiments_cbm_ar", results_cbm_ar)   

N_TOTAL = [2000]
d_variables = [40, 50]


for dim in d_variables:
    results_cbm = []
    results_cem = []
    results_cbm_ar = []
    for seed in seeds:
        for N in N_TOTAL:
            result_cbm = do_cbm_experiment(
                N, 
                test_frac=0.5, 
                d_variables=dim,
                entanglement=0.5, 
                model_type="CBM", 
                seed=seed
            )
            results_cbm.append(result_cbm)

            result_cem = do_cbm_experiment(
                N, 
                test_frac=0.5, 
                d_variables=dim,
                entanglement=0.5, 
                model_type="CEM", 
                seed=seed
            )
            results_cem.append(result_cem)

            result_cbm_ar = do_cbm_experiment(
                N, 
                test_frac=0.5, 
                d_variables=dim,
                entanglement=0.5, 
                model_type="HardCBM", 
                seed=seed
            )
            results_cbm_ar.append(result_cbm_ar)

    print(f"writing results for dimension: {dim}")
    write_to_db("experiments_cbm", results_cbm)   
    write_to_db("experiments_cem", results_cem)   
    write_to_db("experiments_cbm_ar", results_cbm_ar)   
