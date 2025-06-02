import numpy as np
import os
import sys
import warnings
sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], '../..'))

import re
import time
import multiprocessing 
from joblib import Parallel, delayed
from itertools import product 

import duckdb

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import Ridge, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import SplineTransformer
from sklearn.kernel_approximation import RBFSampler
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter("ignore", category=ConvergenceWarning)

import torch
import torch.nn.functional as F
import torch.utils.data as data

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger

from CITRIS_bin.training.datasets import Causal3DDataset, ConceptDataset
from shared_utils.concept_models.cbm.lightning_module import SimpleCBM
from shared_utils.concept_models.cem.lightning_module import CEM 
from shared_utils.concept_models.hardcbm.lightning_module import HardCBM

from shared_utils.concept_models.hardcbm.train import train

from shared_utils.experiment import calc_perm_errors, set_seed
from permutation_estimator.estimator import FeaturePermutationEstimator, KernelizedPermutationEstimator

def get_device():
    if torch.backends.mps.is_available():
        # For small networks and loads this cpu seems to be faster 
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    return device

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
            method,
            regularizer, 
            optim_kwargs,
            N,
            feature_transform,
            n_features,
            encs,
            latents,
            y_values,
            seed,
            groups=None
        ):
        self.method = method
        self.regularizer = regularizer       
        self.optim_kwargs = optim_kwargs
        self.N = N
        self.feature_transform = feature_transform
        self.n_features = n_features
        self.encs = encs
        self.latents = latents
        self.y_values = y_values
        self.groups = groups 
        self.seed = seed

    def __call__(self):
        set_seed(self.seed)
        result = run_permutation_estimation(
                    regularizer=self.regularizer,
                    optim_kwargs=self.optim_kwargs,
                    N=self.N,
                    feature_transform=self.feature_transform,
                    n_features=self.n_features,
                    encs=self.encs,
                    latents=self.latents,
                    y_values=self.y_values,
                    groups=self.groups
                )
        result["method"] = self.method
        result["alpha"] = self.optim_kwargs["alpha"]
        result["N"] = self.N
        result["seed"] = self.seed
        return result
        
    def __str__(self):
        name = f"Permutation experiment"
        return name


def do_cbm_test(
    all_x,
    all_latents,
    y_values, 
    Ns, 
    seed, 
    model_type,
    base_dir, 
    cluster,
    dataset_name=None
):  
    device = get_device()

    latent_dataset = data.TensorDataset(torch.from_numpy(all_latents))   
    label_dataset = data.TensorDataset(torch.from_numpy(y_values[:, np.newaxis]))
    
    if dataset_name is None:
        data_dir = 'data/causal3d_time_dep_all7_conthue_01_coarse'
        img_dataset = Causal3DDataset(
            data_folder=data_dir, 
            split='test_indep', 
            single_image=True, 
            triplet=False, 
            coarse_vars=True,
            exclude_vars=None,
            exclude_objects=None
        )
        dataset = ConceptDataset(img_dataset, latent_dataset, label_dataset)
        conv = True
        c_in = 3
    else:
        x_dataset = data.TensorDataset(torch.from_numpy(all_x))
        dataset = ConceptDataset(x_dataset, latent_dataset, label_dataset, tabular=True)
        conv = False
        c_in = all_x.shape[1]

    ckpt_dir = os.path.join(base_dir, "CBM")
    num_workers = 0
    all_results = []

    for i, N in enumerate(Ns):
        print(f"working on {N} datapoints")
        logger = CSVLogger(ckpt_dir, name=f"N_{N}")

        train_dataset, test_dataset, _ = data.random_split(
            dataset, 
            [N, N, len(dataset) - 2*N], 
            torch.Generator().manual_seed(seed)
        )

        train_loader = data.DataLoader(
            train_dataset, 
            batch_size=256,
            shuffle=True,
            drop_last=False, 
            num_workers=num_workers
        )
        test_loader = data.DataLoader(
            test_dataset, 
            batch_size=256,
            shuffle=False,
            drop_last=False, 
            num_workers=num_workers
        )

        if model_type == "CBM":
            cbm = SimpleCBM(
                c_hid=32,
                c_in=c_in, 
                num_concepts=all_latents.shape[1], 
                num_output=1,
                regression=False,
                conv=conv
            )
        elif model_type == "CEM":
            cbm = CEM(
                c_hid=32,
                c_in=c_in, 
                num_concepts=all_latents.shape[1],
                num_output=1, 
                regression=False,
                conv=conv
            )
        elif model_type == "HardCBM":
            cbm = HardCBM(
                c_hid=32, 
                c_in=c_in, 
                num_concepts=all_latents.shape[1],
                num_output=1,
                conv=conv
            ).to(device)

        if cluster:
            trainer = Trainer(max_epochs=100, logger=logger, enable_progress_bar=False)
        else:
            trainer = Trainer(max_epochs=100, logger=logger)
        
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
                c_hat = cbm.concept_encoder(x.to(cbm.device))
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
            ois_concept = ois_score(all_c, all_c_hat_score, tensor=True)
            nis_concept = nich_impurity_score(all_c, all_c_hat_score, tensor=True)
        else:
            all_c_emb = torch.cat(all_c_emb, dim=0)
            ois_concept = ois_score(all_c, all_c_emb, tensor=True)
            nis_concept = nich_impurity_score(all_c, all_c_emb, tensor=True)

        result = {
            "acc_concept": acc_concept,
            "roc_concept": roc_concept,
            "acc_label": acc_label,
            "roc_label": roc_label,
            "ois_score": ois_concept,
            "nis_score": nis_concept,
            "time": end - start
        }
        print(result)
        result["N"] = N
        result["seed"] = seed

        all_results.append(result)
    
    if model_type == "CBM":
        db_str = "experiment_cbm"
    elif model_type == "CEM":
        db_str = "experiment_cem"
    elif model_type == "HardCBM":
        db_str = "experiment_cbm_ar"
    write_results_to_db(db_str, result=all_results, dataset=dataset_name)
    return result


def sample_correctly(encs, latents, y_values, n, N):
    # We use as many test data points as train data points
    # run until at least 1 of each label is found

    while True:
        indices = np.random.choice(size=n, a=np.arange(n),replace=False)
        train_idx, test_idx = indices[:N], indices[-N:]

        train_encs = encs[train_idx, :]
        test_encs = encs[test_idx, :]

        train_latents = latents[train_idx, :]
        test_latents = latents[test_idx, :]

        train_y = y_values[train_idx,]
        test_y = y_values[test_idx]

        latent_check = np.all([
            np.any(train_latents[:, col] == 0) and
            np.any(train_latents[:, col] == 1) and
            np.any(test_latents[:, col] == 0) and
            np.any(test_latents[:, col] == 1) 
            for col in range(train_latents.shape[1])
        ])

        label_check = np.all([
            np.any(train_y == 0) and
            np.any(train_y == 1) and
            np.any(test_y == 0) and
            np.any(test_y == 1) 
        ])
        if latent_check and label_check:
            return {
                "train_encs": train_encs,
                "test_encs": test_encs,
                "train_latents": train_latents,
                "test_latents": test_latents,
                "train_y": train_y,
                "test_y": test_y,
            }


def split_correctly(y_score, y_true, test_size):
    while True:
        y_score_train, y_score_test, y_true_train, y_true_test = train_test_split(
            y_score, y_true, test_size=test_size
        )

        label_check = np.all([
            np.any(y_true_train[:, col] == 0) and
            np.any(y_true_train[:, col] == 1) and
            np.any(y_true_test[:, col] == 0) and
            np.any(y_true_test[:, col] == 1) 
            for col in range(y_true_train.shape[1])
        ])
        if label_check:
            return y_score_train, y_score_test, y_true_train, y_true_test

def concept_correlation_nicher(y_true, y_score, multi_dim=False):
    n_concepts = y_true.shape[1]
    if multi_dim:
        corr_matrix = np.zeros((n_concepts, n_concepts))
        n_emb = y_score.shape[2]
        for i in range(n_concepts):
            for j in range(n_concepts):
                corr_intermediate =  np.corrcoef(np.hstack([y_score[:, i, :], y_true[:, j].reshape(-1, 1)]).T)
                nm = corr_intermediate[:n_emb, n_emb:]
                corr_matrix[i, j] = nm.max()
    else:
        corr_matrix = np.abs(np.corrcoef(y_score.T, y_true.T)[:n_concepts, n_concepts:])
    return corr_matrix


def concept_niche(corr_matrix, beta):
    n_concepts = corr_matrix.shape[1]
    
    niche = []
    for j in range(n_concepts):
        niche_j = [i for i in range(n_concepts) if corr_matrix[i, j] > beta]
        niche.append(niche_j)

    return niche


def niche_impurity(y_true, y_score, corr_matrix, beta, mlp, multi_dim=False):
    n_concepts = y_true.shape[1]

    niche = concept_niche(corr_matrix, beta) 
    niche_impurities = np.zeros(n_concepts)

    repeats = y_score.shape[1] / n_concepts
    for i in range(n_concepts):
        mask = np.ones(n_concepts)
        mask[niche[i]] = 0
        if multi_dim:
            mask = np.repeat(mask, repeats)
        y_score_masked = y_score * mask[np.newaxis, :]

        y_mlp_score_test = mlp.predict_proba(y_score_masked)[:, i]

        niche_impurities[i] = roc_auc_score(y_true[:, i], y_mlp_score_test)
    return niche_impurities


def nich_impurity_score(y_true, y_score, tensor=False):
    (n_samples, n_concepts) = y_true.shape
    dimensions = y_score.shape
    if tensor:
        y_score = y_score.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()

    y_score_train, y_score_test, y_true_train, y_true_test = split_correctly(
        y_score, y_true, test_size=0.2
    )
    mlp = MLPClassifier(
        hidden_layer_sizes=[20, 20], 
        activation='relu', 
        max_iter=1000, 
        batch_size=min(512, n_samples)
    )

    if len(dimensions) > 2:
        y_score_train2 = y_score_train.reshape(-1, dimensions[2] * n_concepts)
        y_score_test2 = y_score_test.reshape(-1, dimensions[2] * n_concepts)
        corr_matrix = concept_correlation_nicher(y_true, y_score, multi_dim=True)

        mlp.fit(y_score_train2, y_true_train)
    else:
        corr_matrix = concept_correlation_nicher(y_true, y_score, multi_dim=False)

        mlp.fit(y_score_train, y_true_train)
    

    betas = np.linspace(0, 1, 21)
    niche_impurities_betas = np.zeros((21, n_concepts))
    for i, beta in enumerate(betas):
        if len(dimensions) > 2:
            niche_impurities_betas[i, :] = niche_impurity(y_true_test, y_score_test2, corr_matrix, beta, mlp, multi_dim=True)
        else:
            niche_impurities_betas[i, :] = niche_impurity(y_true_test, y_score_test, corr_matrix, beta, mlp, multi_dim=True)

    scores = np.trapz(niche_impurities_betas, dx=0.05, axis=0)
    return scores.sum() / n_concepts


def ois_score(y_true, y_score, tensor=False):
    n_concepts = y_true.shape[1]   

    impurity_matrix_hat = np.zeros((n_concepts, n_concepts))
    impurity_matrix = np.zeros((n_concepts, n_concepts))

    if tensor:
        y_score = y_score.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()
    y_score_train, y_score_test, y_true_train, y_true_test = split_correctly(
        y_score, y_true, test_size=0.2
    )
    dimensions = y_score.shape
    for i in range(n_concepts):
        for j in range(n_concepts):
            if len(dimensions) == 2:
                mlp = MLPClassifier(hidden_layer_sizes=[32], activation='relu', max_iter=32)
                mlp.fit(y_score_train[:, i:i+1], y_true_train[:, j])
                y_mlp_score_test = mlp.predict_proba(y_score_test[:, i:i+1])[:, 1]
                impurity_matrix_hat[i, j] = roc_auc_score(y_true_test[:, j], y_mlp_score_test)

                mlp = MLPClassifier(hidden_layer_sizes=[32], activation='relu', max_iter=32)
                mlp.fit(y_true_train[:, i:i+1], y_true_train[:, j])
                y_mlp_score_test = mlp.predict_proba(y_true_test[:, i:i+1])[:, 1]
                impurity_matrix[i, j] = roc_auc_score(y_true_test[:, j], y_mlp_score_test)

            else:
                mlp = MLPClassifier(hidden_layer_sizes=[32], activation='relu', max_iter=32)
                mlp.fit(y_score_train[:, i, :], y_true_train[:, j])
                y_mlp_score_test = mlp.predict_proba(y_score_test[:, i, :])[:, 1]
                impurity_matrix_hat[i, j] = roc_auc_score(y_true_test[:, j], y_mlp_score_test)

                mlp = MLPClassifier(hidden_layer_sizes=[32], activation='relu', max_iter=32)
                mlp.fit(y_true_train[:, i:i+1], y_true_train[:, j])
                y_mlp_score_test = mlp.predict_proba(y_true_test[:, i:i+1])[:, 1]
                impurity_matrix[i, j] = roc_auc_score(y_true_test[:, j], y_mlp_score_test)


    norm_diff = np.linalg.norm(impurity_matrix_hat - impurity_matrix)
    
    return 2 * norm_diff / n_concepts

def run_permutation_estimation(
    regularizer: str, 
    optim_kwargs: dict,
    N: int,
    feature_transform: list,
    n_features: int,
    encs: np.array, 
    latents: np.array,
    y_values: np.array,
    groups=None,
) -> np.array:
    n = encs.shape[0]
    nr_latents = latents.shape[1]
    z_dim = encs.shape[1]

    # rand_perm = np.arange(nr_latents)
    rand_perm = np.random.choice(size=nr_latents, a=np.arange(nr_latents),replace=False)
    permuted_latents = latents[:, rand_perm]

    sampled_arrs = sample_correctly(encs, permuted_latents, y_values, n, N)
    train_encs = sampled_arrs["train_encs"]
    test_encs = sampled_arrs["test_encs"]

    train_latents = sampled_arrs["train_latents"]
    test_latents = sampled_arrs["test_latents"]

    train_y = sampled_arrs["train_y"]
    test_y = sampled_arrs["test_y"]

    if isinstance(feature_transform, str):
        estimator = KernelizedPermutationEstimator(
            regularizer=regularizer, 
            optim_kwargs=optim_kwargs, 
            kernel=feature_transform,
            parameter=1,
            n_features=n_features, 
            d_variables=z_dim,
            groups=groups
        )
    else:
        if regularizer == "lasso":
            two_stage = "ridge"
        elif regularizer == "logistic_lasso":
            two_stage = "logistic"
            regularizer = "lasso"
        else:
            two_stage = None


        estimator = FeaturePermutationEstimator(
            regularizer=regularizer, 
            optim_kwargs=optim_kwargs, 
            feature_transform=feature_transform, 
            n_features=n_features,
            d_variables=z_dim,
            groups=groups, 
            two_stage=two_stage
        )

    # To have it the same as the CBM model
    predictor = SGDClassifier(loss="log_loss")
    # predictor = Ridge(alpha=0.05, fit_intercept=True)

    if groups is None:
        res = estimator.fit(
            train_encs.T, train_latents.T
        )
    else:
        res = estimator.fit(
            train_encs.T, train_latents.T,
            recover_weights=False
        )

    train_latent_hat = estimator.predict_match(train_encs.T)
    test_latents_hat_score = estimator.predict_match(test_encs.T)
    
    if two_stage is None:
        acc_concept = accuracy_score(test_latents.reshape(-1, 1), (test_latents_hat_score.T < 0).reshape(-1, 1))
        roc_concept = roc_auc_score(test_latents, -test_latents_hat_score.T, multi_class="ovr")

        # OIS score 
        ois_concept = ois_score(test_latents, -test_latents_hat_score.T)
        nis_concept = nich_impurity_score(test_latents, -test_latents_hat_score.T)
    else:
        acc_concept = accuracy_score(test_latents.reshape(-1, 1), (test_latents_hat_score.T > 0).reshape(-1, 1))
        roc_concept = roc_auc_score(test_latents, test_latents_hat_score.T, multi_class="ovr")

        # OIS score 
        ois_concept = ois_score(test_latents, test_latents_hat_score.T)
        nis_concept = nich_impurity_score(test_latents, test_latents_hat_score.T)


    predictor.fit(train_latent_hat.T, train_y)
    test_y_hat = predictor.predict(test_latents_hat_score.T)
    test_y_hat_score = predictor.predict_proba(test_latents_hat_score.T)[:, 1]
    
    acc_label = accuracy_score(test_y, test_y_hat)
    roc_label = roc_auc_score(test_y, test_y_hat_score)

    print(rand_perm)
    print(res["perm_hat_match"])
    errors_match = calc_perm_errors(rand_perm, res["perm_hat_match"])

    times_match = res["time_match"]

    results = {
        "perm_error": errors_match,
        "acc_concept": acc_concept,
        "roc_concept": roc_concept, 
        "acc_label": acc_label,
        "roc_label": roc_label,
        "ois_score": ois_concept,
        "nis_score": nis_concept,
        "time": times_match,
    }
    print(results)

    return results

def do_vae_test(
    all_latents,
    all_encs,
    alphas,
    Ns, 
    groups, 
    seeds,
    model_type, 
    dataset=None
) -> np.array:

    z_dim = all_encs.shape[1]

    n_knots = 6
    degree = 3
    n_features = n_knots + degree - 1

    splines_all_dims = [SplineTransformer(
        n_knots=n_knots,
        degree=degree) for _ in range(z_dim)]
    rff_all_dims = [RBFSampler(n_components=n_features) for _ in range(z_dim)]

    if dataset == None:
        if model_type == "CITRISVAE":
            db_str = "experiment_citris"
        elif model_type == "iVAE":
            db_str = "experiment_ivae"
    else:
        if model_type == "DMS-VAE":
            db_str = "experiment_dms"
        elif model_type == "iVAE" or model_type == "TCVAE":
            db_str = "experiment_ivae"
 


    def inner_loop(method,seed, alpha, N):
        set_seed(seed)                           
        y_values = create_labels(all_latents)

        if method == "Linear":
            task = Task(
                method="Linear",
                regularizer="logistic_group", 
                optim_kwargs={"alpha": alpha, "scale_reg": "group_size", "l1_reg": 0},
                N=N,
                feature_transform=None,
                n_features=1,
                encs=all_encs,
                latents=all_latents,
                y_values=y_values,
                seed=seed,
                groups=groups)
            result = task()
        elif method == "RFF":
            task = Task(
                method="RFF",
                regularizer="logistic_group", 
                optim_kwargs={"alpha": alpha, "scale_reg": "group_size", "l1_reg": 0},
                N=N,
                feature_transform=rff_all_dims,
                n_features=n_features,
                encs=all_encs,
                latents=all_latents,
                y_values=y_values,
                seed=seed,
                groups=groups)
            result = task()
        elif method == "Spline":
            task = Task(
                method="Spline",
                regularizer="logistic_group", 
                optim_kwargs={"alpha": alpha, "scale_reg": "group_size", "l1_reg": 0},
                N=N,
                feature_transform=splines_all_dims,
                n_features=n_features,
                encs=all_encs,
                latents=all_latents,
                y_values=y_values,
                seed=seed,
                groups=groups)
            result = task()
        elif method == "Laplacian":
            n_kernel = min(N, 20)
            task = Task(
                method="Laplacian",
                regularizer="logistic_group", 
                optim_kwargs={"alpha": alpha, "scale_reg": "group_size", "l1_reg": 0},
                N=N,
                feature_transform="laplacian",
                n_features=n_kernel,
                encs=all_encs,
                latents=all_latents,
                y_values=y_values,
                seed=seed,
                groups=groups)
            result = task()
        elif method == "two_stage":
            task = Task(
                method="two_stage",
                regularizer="logistic_lasso", 
                optim_kwargs={"alpha": alpha},
                N=N,
                feature_transform=splines_all_dims,
                n_features=n_features,
                encs=all_encs,
                latents=all_latents,
                y_values=y_values,
                seed=seed,
                groups=groups)   
            result = task()
        return result

    num_workers = os.sched_getaffinity(0)
    # num_workers = [0, 1, 2, 3]
    print(f"Num of workers = {len(num_workers)}")
    methods = ["two_stage"] # ["Linear", "RFF", "Spline", "Laplacian"] #"two_stage"]
    all_results = Parallel(n_jobs=len(num_workers), verbose=5)(
        delayed(inner_loop)(method, seed, alpha, N)
        for method, seed, alpha, N in product(methods, seeds, alphas, Ns)
        )

    print("Done!")
    write_results_to_db(db_str=db_str, result=all_results, dataset=dataset)
    return 

def get_groups(model, all_encs, all_latents, ckpt_dir):
    ta_saved_fname = os.path.join(ckpt_dir, "ta_ivae_saved.pt")

    if hasattr(model, 'prior_t1'):
        # If a target assignmente exist we will use it to create the groups
        target_assignment = model.prior_t1.get_target_assignment(hard=True)
        groups = target_assignment.argmax(dim=1).cpu().numpy()
        print("prior_t1 groups are:", groups)
    elif hasattr(model, 'target_assignment') and model.target_assignment is not None:
        target_assignment = model.target_assignment.clone()
        groups = target_assignment.argmax(dim=1).cpu().numpy()
        print(groups)
    elif os.path.isfile(ta_saved_fname):
        target_assignment = torch.load(ta_saved_fname)
        groups = target_assignment.argmax(dim=1).cpu().numpy()

        print("Loaded groups are:", groups)
    else:

        print("No groups found")
        groups = None
    return groups

def find_best_model_and_latents(
        model_dir, 
        model_class
    ):
    # Find all subdirectories that match the version pattern
    version_dirs = [d for d in os.listdir(model_dir) if re.match(r'^version_\d+$', d)]
    
    # If no version directories found, return None
    if not version_dirs:
        return None
    
    latest_version = max(version_dirs, key=lambda d: int(d.split('_')[1]))
    print(f"-> version: {latest_version}")
    ckpt_dir = os.path.join(model_dir, latest_version)

    # true_latents are sources
    device = get_device()
    model_dir = os.path.join(ckpt_dir, 'checkpoints/last.ckpt')
    model = model_class.load_from_checkpoint(model_dir).to(device)

    encodings_latents_fname = os.path.join(ckpt_dir, 'encodings_latents.npz')

    data_dir = 'data/causal3d_time_dep_all7_conthue_01_coarse'
    arg_names = load_arg_names(data_dir=data_dir)

    if os.path.isfile(encodings_latents_fname):
        arrs = np.load(encodings_latents_fname)
        all_encs = torch.from_numpy(arrs["encs"])
        all_latents = torch.from_numpy(arrs["latents"])
    else:
        # if not we have to calculate the latents
        print("Computing latents and encodings")
        test_loader = load_datasets(data_dir=data_dir)

        all_encs = []
        all_latents = []
        for batch in test_loader:
            imgs, latents = batch
            encs = model.encode(imgs.to(model.device)).detach().cpu()
            all_encs.append(encs)
            all_latents.append(latents)
        all_encs = torch.cat(all_encs, dim=0)
        all_latents = torch.cat(all_latents, dim=0)
        # 25000 x 32, 25000 x 10

        all_encs = all_encs.numpy()
        all_latents = all_latents.numpy()

        np.savez(
            os.path.join(ckpt_dir, "encodings_latents.npz"), 
            encs=all_encs, 
            latents=all_latents
        )


    # Return the directory with the largest version number
    return ckpt_dir, model, all_encs, all_latents, arg_names


def load_arg_names(data_dir):
    dataset_args = {
        'coarse_vars': True, 
        'exclude_vars': None, 
        'exclude_objects': None
    }
    test_args = lambda train_set: {'causal_vars': train_set.full_target_names}
    dataset = Causal3DDataset(
        data_folder=data_dir, 
        split='train', 
        single_image=False, 
        triplet=False, 
        seq_len=2,
        **dataset_args
    )
    return (dataset.full_target_names, dataset.target_names_l)


def load_datasets(data_dir):
    dataset_args = {
        'coarse_vars': True, 
        'exclude_vars': None, 
        'exclude_objects': None
    }
    test_args = lambda train_set: {'causal_vars': train_set.full_target_names}

    train_dataset = Causal3DDataset(
        data_folder=data_dir, 
        split='train', 
        single_image=False, 
        triplet=False, 
        seq_len=2,
        **dataset_args
    )
    test_dataset = Causal3DDataset(
        data_folder=data_dir,
        split='test_indep', 
        single_image=True, 
        triplet=False, 
        return_latents=True, 
        **dataset_args, 
        **test_args(train_dataset)
    )

    test_loader = data.DataLoader(
        test_dataset, 
        batch_size=256,
        shuffle=False, 
        drop_last=False,
        num_workers=0
    )
    return test_loader


def create_labels(binary_latents):
    num_latents = binary_latents.shape[1]

    options = np.arange(num_latents)
    active_concepts = np.random.choice(options, size=3)
    sum_active = binary_latents[:, active_concepts].sum(axis=1)
    labels = np.zeros(binary_latents.shape[0])
    labels[sum_active > 1] = 1
    return labels


def write_results_to_db(db_str, result, dataset=None):
    print("Check results")
    print(result)
    if "cbm" in db_str or "cem" in db_str:
        if dataset is None:
            method_cols = "$N,"
        else:
            method_cols = f"$N,'{dataset}',"
    else:
        if dataset is None:
            method_cols = "$method, $alpha, $N, $perm_error,"
        else:
            method_cols = f"$method, $alpha, $N, '{dataset}', $perm_error,"
    
    with duckdb.connect(f"checkpoints/{db_str}.duckdb") as con:
        con.executemany(f"""
            INSERT OR REPLACE INTO experiments
            VALUES (
                {method_cols}
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
        result)
        con.table("experiments").show(max_rows=5)
