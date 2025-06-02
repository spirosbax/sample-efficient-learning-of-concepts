from copy import copy
import numpy as np 
import scipy as sp
from time import time
from typing import Union, List, Dict

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.kernel_approximation import Nystroem
from sklearn.utils.validation import check_is_fitted
from group_lasso import GroupLasso, LogisticGroupLasso

class permutationMixin:

    def _setup_optim(
        self,
        regularizer: str, 
        dim: int, 
        n_features: int,
        optim_kwargs: Dict,
        groups: Union[None, np.array]=None
    ):  
        allowed_regs = ["logistic_group", "group", "lasso", "elastic", "ridge", "logistic"]
        assert regularizer in allowed_regs, f"{regularizer} not recognized. Select one of the accepted options."
        if groups is None:
            self.lasso_groups = np.concatenate(
                [n_features * [idx] for idx in range(dim)] 
            )
        else:
            self.lasso_groups = np.concatenate(
                [n_features * [g] for g in groups] 
            )

        if optim_kwargs['alpha'] == 0:
            optim = Ridge()
        else:
            if regularizer == "group":
                optim_kwargs["group_reg"] = optim_kwargs["alpha"]
                optim_kwargs.pop("alpha", None)

                optim = GroupLasso(
                    groups=self.lasso_groups,
                    supress_warning=True,
                    n_iter=20000,
                    tol=1e-4,
                    **optim_kwargs
                )
            elif regularizer == "logistic_group":
                optim_kwargs["group_reg"] = optim_kwargs["alpha"]
                optim_kwargs.pop("alpha", None)

                optim = LogisticGroupLasso(
                    groups=self.lasso_groups,
                    supress_warning=True,
                    n_iter=20000,
                    tol=1e-4,
                    **optim_kwargs
                )
            elif regularizer == "lasso":
                optim = Lasso(
                    max_iter=20000,
                    tol=1e-4, 
                    selection='cyclic',
                    **optim_kwargs
                )
            elif regularizer == "elastic":
                optim = ElasticNet(
                    max_iter=20000,
                    tol=1e-4, 
                    selection='cyclic',
                    **optim_kwargs
                )
            elif regularizer == "ridge":
                optim = Ridge(
                    max_iter=20000, 
                    tol=1e-4,
                    **optim_kwargs
                )
                if groups is None:
                    self.pred_groups = np.concatenate(
                        [n_features * [idx] for idx in range(dim)]
                    )
                else:
                    self.pred_groups = np.concatenate(
                        [n_features * [g] for g in groups]
                    )
            elif regularizer == "logistic":
                if groups is None:
                    self.pred_groups = np.concatenate(
                        [n_features * [idx] for idx in range(dim)]
                    )
                else:
                    self.pred_groups = np.concatenate(
                        [n_features * [g] for g in groups]
                    )
                optim = LogisticRegression(max_iter=20000)

        return optim

    def _invert_permutation(self, permutation):
        return np.argsort(permutation)

    def _get_perm_and_weights(
        self,
        coef_matrix: np.array, 
        dim: int, 
        n_features: int,
        groups: Union[None, np.array]=None, 
        soft_matching: bool=False,
    ) -> np.array:
        ## Getting the permutation ##
        perm_max, perm_match = self._get_perm(coef_matrix, dim, n_features, groups, soft_matching)

        w_hat_max = self._get_weights(perm_max, coef_matrix, n_features)
        w_hat_match = self._get_weights(perm_match, coef_matrix, n_features)

        return perm_max[:, 0], perm_match[:,0], w_hat_max, w_hat_match

    def _get_corr_perm(
        self, 
        x: np.array, 
        y: np.array, 
        groups: Union[None, np.array]=None
    ) -> np.array:
        if groups is None:
            dim = x.shape[0]
            cor_abs = np.abs(np.corrcoef(x, y))[dim:, :dim]
            perm_corr = sp.optimize.linear_sum_assignment(-1 * cor_abs)[1]
        else:
            unique_groups = np.unique(groups, return_index=True)[0]
            grouped_x = np.zeros(shape=(len(unique_groups), x.shape[1]))
            for group in unique_groups:
                group_rows = groups == group
                grouped_x[group, :] = x[group_rows, :].sum(axis=0)

            dim = y.shape[0]
            cor_abs = np.abs(np.corrcoef(grouped_x, y)[dim:, :dim])
            perm_corr = sp.optimize.linear_sum_assignment(-1 * cor_abs)[1]
        return perm_corr

    def _get_spr_perm(
        self, 
        x: np.array, 
        y: np.array, 
        groups: Union[None, np.array]=None
    ) -> np.array:
        if groups is None:
            dim = y.shape[0]
            try:
                spr = np.abs(sp.stats.spearmanr(x.T, y.T).statistic[dim:, :dim])
            except AttributeError:
                spr = np.abs(sp.stats.spearmanr(x.T, y.T)[0][dim:, :dim])
            perm_spr = sp.optimize.linear_sum_assignment(-1 * spr)[1]
        else:
            unique_groups = np.unique(groups, return_index=True)[0]
            grouped_x = np.zeros(shape=(len(unique_groups), x.shape[1]))
            for group in unique_groups:
                group_rows = groups == group
                grouped_x[group, :] = x[group_rows, :].sum(axis=0)

            dim = y.shape[0]
            spr = np.abs(sp.stats.spearmanr(grouped_x.T, y.T).statistic[dim:, :dim])
            perm_spr = sp.optimize.linear_sum_assignment(-1 * spr)[1]
        return perm_spr

    def _get_perm(
        self, 
        coef_matrix: np.array,
        dim: int,
        n_features: int,
        groups: Union[None, np.array]=None,
        soft_matching: bool=False
    ) -> np.array:
        summed_coef_matrix = np.abs(coef_matrix)
        if groups is None:
            summed_coef_matrix = summed_coef_matrix.reshape(dim, n_features, -1, order="F")
            summed_coef_matrix = np.sqrt(
                np.square(summed_coef_matrix).mean(axis=1)
                )

        else:
            unique_groups = np.unique(groups, return_index=True)[0]
            result = np.zeros(shape=(summed_coef_matrix.shape[0], len(unique_groups)))
            for group in unique_groups:
                group_columns = self.lasso_groups == group
                result[:, group] = np.sqrt(
                    np.square(summed_coef_matrix[:, group_columns]).mean(axis=1)
                )

            summed_coef_matrix = result

        if soft_matching:
        # Adapted from https://python.quantecon.org/opt_transport.html
            summed_coef_matrix = np.abs(coef_matrix)
            print(summed_coef_matrix.shape)
            d_out, d_in = summed_coef_matrix.shape
            summed_coef_matrix_vec = summed_coef_matrix.reshape((-1, 1), order="F")

            mat_1 = np.kron(np.ones((1, d_in)), np.identity(d_out))
            mat_2 = np.kron(np.identity((d_in)), np.ones((1, d_out)))
            mat = np.vstack([mat_1, mat_2])

            b_1 = np.ones(d_out) / d_out
            b_2 = np.ones(d_in) / d_in
            b = np.hstack([b_1, b_2])

            res = sp.optimize.linprog(
                -1 * summed_coef_matrix_vec, 
                A_eq=mat,
                b_eq=b
            )
            if res.x is not None:
                self.soft_perm_match = res.x.reshape((d_out, d_in), order="F")
            else:
                self.soft_perm_match = np.ones((d_out, d_in))
        perm_match = sp.optimize.linear_sum_assignment(-1 * summed_coef_matrix)[1].reshape(-1, 1)

        self.summed_coef_matrix = summed_coef_matrix

        perm_max = np.argsort(summed_coef_matrix, axis=1)[:, -1:]
        return perm_max, perm_match

    def _get_weights(
        self,
        perm: np.array,
        coef_matrix: np.array,
        n_features: int
    ) -> np.array:
        rows = np.arange(
            coef_matrix.shape[0], 
            dtype=np.intp
        )[:, np.newaxis]
        idx_to_add = np.repeat(
            np.arange(n_features)[np.newaxis, :], 
            repeats=coef_matrix.shape[0], 
            axis=0
        )
        col_idx = idx_to_add + (perm * n_features).astype(np.intp)
        w_hat = coef_matrix[rows, col_idx]
        
        return w_hat


    def _predict(
        self,
        X: np.array, 
        w: np.array, 
        permutation: np.array
    ):
        """
        x |-> permutation ∘ (w @ phi(x)), with w a weight matrix
        """
        check_is_fitted(self)
        if self.groups is None:
            if self.two_stage is None:
                phi_x = np.zeros(shape=(w.shape[0], X.shape[1]))
                for i, group in enumerate(permutation):
                    group_columns = self.lasso_groups == group
                    w_x = w[i, :].reshape(-1, 1) * X[group_columns, :]
                    phi_x[i, :] = w_x.sum(axis=0)
            else:
                phi_x = np.zeros(shape=(len(w), X.shape[1]))
                for i, group in enumerate(permutation):
                    group_columns = self.pred_groups == group
                    w_x = w[i].T * X[group_columns, :]
                    phi_x[i, :] = w_x.sum(axis=0)

            # X = X.reshape(w.shape[0], self.n_features, X.shape[1]).transpose(0, 2, 1)
            # w_x = w[:, np.newaxis, :] * X
            # phi_x = np.sum(w_x, axis=2) 
        else:
            if self.two_stage is None:
                phi_x = np.zeros(shape=(w.shape[0], X.shape[1]))
                unique_groups = np.unique(self.groups, return_index=True)[0]
                for i, group in enumerate(permutation):
                    group_columns = self.lasso_groups == group
                    w_x = w[i, :].reshape(-1, 1) * X
                    phi_x[i, :] = w_x[group_columns, :].sum(axis=0)
            else:
                unique_groups = np.unique(self.groups, return_index=True)[0]
                phi_x = np.zeros(shape=(len(w), X.shape[1]))
                for i, group in enumerate(permutation):
                    group_columns = self.pred_groups == group
                    w_x = w[i].T * X[group_columns, :]
                    phi_x[i, :] = w_x.sum(axis=0)
        if len(self.intercept_.shape) > 1:
            return phi_x + self.intercept_
        else:
            return phi_x + self.intercept_[:, np.newaxis]

    def _perform_fit(
        self,
        optim,
        phi_x: np.array, 
        y: np.array, 
        n_features: np.array,
        recover_weights: bool=True, 
        soft_matching: bool=False
    ):
        dim = y.shape[0]
        if optim.__class__.__name__ == "LogisticGroupLasso":
            beta_hat_all = np.zeros((dim, phi_x.shape[0]))
            self.intercept_ = np.zeros(dim)
            for i in range(dim):
                optim.fit(phi_x.T, y[i, :].T)
                beta_hat_all[i, :] = optim.coef_[:, 0].T

                self.intercept_[i] = self._optim.intercept_[0]
            # phi_x_kron = sp.sparse.kron(np.eye(dim), phi_x)
            # y_kron = y.reshape(-1, 1)
            # optim.fit(phi_x_kron.T, y_kron)
            # beta_hat_all = optim.coef_[:,0].reshape((dim, phi_x.shape[0]))
            # self.lasso_groups = self.lasso_groups[:phi_x.shape[0]]
        elif optim.__class__.__name__ == "GroupLasso":
            optim.fit(phi_x.T, y.T)
            beta_hat_all = optim.coef_.T
        else:
            optim.fit(phi_x.T, y.T)
            beta_hat_all = optim.coef_

        if recover_weights:
            res = self._get_perm_and_weights(
                coef_matrix=beta_hat_all,
                dim=dim, 
                n_features=n_features,
                groups=self.groups, 
                soft_matching=soft_matching
            )
            return_dict = {
                "perm_hat_max" : res[0],
                "perm_hat_match" : res[1],
                "beta_hat_max" : res[2],
                "beta_hat_match" : res[3],
                "beta_hat_all": beta_hat_all
            }
   
        else:
            res = self._get_perm(
                coef_matrix=beta_hat_all,
                dim=dim, 
                n_features=n_features,
                groups=self.groups,
                soft_matching=soft_matching
            )
            return_dict = {
                "perm_hat_max" : res[0][:, 0],
                "perm_hat_match" : res[1][:, 0],
                "beta_hat_max": None,
                "beta_hat_match": None,
                "beta_hat_all": beta_hat_all, 
            }

        return return_dict

    def _perform_predict_fit(
        self,
        phi_x: np.array, 
        y: np.array
    ):  

        if self.groups is None:
            beta_hat = [0] * self.d_variables
            intercept = [0] * self.d_variables

            for i in range(self.d_variables):
                j_i = self.perm_hat_match_[i]
                group_slice = slice(j_i * self.n_features, (j_i+1) * self.n_features)
                self._predict_optims[i].fit(phi_x[group_slice, :].T, y[i, :].T)
                beta_hat[i] = self._predict_optims[i].coef_
                intercept[i] = self._predict_optims[i].intercept_

            self.beta_hat_match_ = np.array(beta_hat)
            self.intercept_ = np.array(intercept)

        else:
            beta_hat = [0] * len(self.perm_hat_match_)
            intercept = [0] * len(self.perm_hat_match_)

            for i, group in enumerate(self.perm_hat_match_):
                j_i = group
                group_columns = self.pred_groups == j_i
                self._predict_optims[i].fit(phi_x[group_columns, :].T, y[i, :].T)
                beta_hat[i] = self._predict_optims[i].coef_
                intercept[i] = self._predict_optims[i].intercept_

            self.beta_hat_match_ = beta_hat
            self.beta_hat_all_ = beta_hat
            self.intercept_ = np.array(intercept)


            
    def predict_match(self, X):
        check_is_fitted(self)

        phi_x = self.transform(X)
        perm_inv = self._invert_permutation(self.perm_hat_match_)
        
        if self.groups is None:
            y_hat = self._predict(
                X=phi_x, 
                w=self.beta_hat_match_, 
                permutation=self.perm_hat_match_, 
            )
        else:
            if self.two_stage is None:
                y_hat = self._predict(
                    X=phi_x, 
                    w=self.beta_hat_all_, 
                    permutation=self.perm_hat_match_, 
                )
            else:
                y_hat = self._predict(
                    X=phi_x, 
                    w=[self.beta_hat_all_[int(i)] for i in range(len(perm_inv))], 
                    permutation=self.perm_hat_match_, 
                )
        return y_hat

    def predict_max(self, X):
        check_is_fitted(self)

        phi_x = self.transform(X)
        perm_inv = self._invert_permutation(self.perm_hat_max_)
        if self.groups is None:
            y_hat = self._predict(
                X=phi_x, 
                w=self.beta_hat_match_, 
                permutation=self.perm_hat_max_, 
            )
        else:
            if self.two_stage is None:
                y_hat = self._predict(
                    X=phi_x, 
                    w=self.beta_hat_all_, 
                    permutation=self.perm_hat_max_, 
                )
            else:
                y_hat = self._predict(
                    X=phi_x, 
                    w=[self.beta_hat_all_[int(i)] for i in range(len(perm_inv))], 
                    permutation=self.perm_hat_max_, 
                )
        return y_hat
    
    def predict_full(self, X):
        check_is_fitted(self)

        phi_x = self.transform(X)
        y_hat = self._optim.predict(phi_x.T)
        return y_hat.T
    

class FeaturePermutationEstimator(permutationMixin, BaseEstimator):
    """
    Class to perform the permutation estimation, 
    modelled after Scikit-Learn Estimator class
    """
    def __init__(
            self, 
            regularizer: str, 
            optim_kwargs: Dict,
            feature_transform: List[callable],
            d_variables: int,
            n_features: int, 
            groups: Union[None, np.array]=None,
            two_stage: Union[str, None]=None
    ) -> None:
        self.regularizer = regularizer
        assert 'alpha' in optim_kwargs, "No regularization parameter found, this option must be supplied."
        self.optim_kwargs = copy(optim_kwargs)
        self.feature_transform = feature_transform
        self.d_variables = d_variables
        self.n_features = n_features
        self.groups = groups
        self.two_stage = two_stage

    def fit(self, X, y, is_fit_transformed=False, recover_weights=True, soft_matching=False):

        start = time()
        # Apply feature transform

        if self.two_stage is not None:
            if X.shape[1] <= 20:
                X_perm, X_predict, y_perm, y_predict = copy(X), copy(X), copy(y), copy(y)
            else:
                X_perm, X_predict, y_perm, y_predict = train_test_split(
                    X.T, y.T, test_size=0.8
                )
                X_perm, X_predict, y_perm, y_predict = X_perm.T, X_predict.T, y_perm.T, y_predict.T
            if not is_fit_transformed:
                phi_x_perm = X_perm
                phi_x_predict = self.fit_transform(X_predict)
            else:
                phi_x_perm = X_perm
                phi_x_predict = self.transform(X_predict)

            self._predict_optims = [
                self._setup_optim(
                    regularizer=self.two_stage, 
                    optim_kwargs={"alpha": self.optim_kwargs["alpha"]}, 
                    dim=self.d_variables,
                    n_features=self.n_features, 
                    groups=self.groups
                    ) 
                for _ in range(self.d_variables)
                ]

            _perm_features = 1
        else:
            if not is_fit_transformed:
                phi_x_perm = self.fit_transform(X)
            else:
                phi_x_perm = self.transform(X)

            X_perm = X
            y_perm = y

            _perm_features = self.n_features

        optim = self._setup_optim(
            regularizer=self.regularizer,
            optim_kwargs=self.optim_kwargs,
            dim=self.d_variables, 
            n_features=_perm_features,
            groups=self.groups
        )
        self._optim = optim

        res = self._perform_fit(
            optim=optim,
            phi_x=phi_x_perm, 
            y=y_perm, 
            n_features=_perm_features,
            recover_weights=recover_weights, 
            soft_matching=soft_matching
        )
        stop = time()
        res["time_match"] = stop - start
        
        if self.regularizer != "logistic_group":
            start = time()
            self.perm_hat_corr_ = self._get_corr_perm(X_perm, y_perm, groups=self.groups)
            stop = time()
            res["time_corr"] = stop - start

            start = time()
            self.perm_hat_spr_ = self._get_spr_perm(X_perm, y_perm, groups=self.groups)
            stop = time()
            res["time_spear"] = stop - start 

            self.perm_hat_max_ = res["perm_hat_max"]
            self.perm_hat_match_ = res["perm_hat_match"]
            self.beta_hat_max_ = res["beta_hat_max"]
            self.beta_hat_match_ = res["beta_hat_match"]
            self.beta_hat_all_ = res["beta_hat_all"]

            self.intercept_ = self._optim.intercept_

            res["non_zero_count"] = np.sum(~np.isclose(self.beta_hat_all_, 0, atol=1e-6))

            res["perm_hat_spr"] = self.perm_hat_spr_
            res["perm_hat_corr"] = self.perm_hat_corr_
        else:
            self.perm_hat_max_ = res["perm_hat_max"]
            self.perm_hat_match_ = res["perm_hat_match"]
            self.beta_hat_max_ = res["beta_hat_max"]
            self.beta_hat_match_ = res["beta_hat_match"]
            self.beta_hat_all_ = res["beta_hat_all"]

            res["non_zero_count"] = np.sum(~np.isclose(self.beta_hat_all_, 0, atol=1e-6))

        if self.two_stage:
            self._perform_predict_fit(phi_x_predict, y_predict)
        return res


    def fit_transform(self, X: np.array):
        n_samples = X.shape[1]
        self.std_scalars = [StandardScaler() for _ in range(self.d_variables)]
        if n_samples >= self.n_features:
            self.pcas = [PCA(n_components=self.n_features) for _ in range(self.d_variables)]
        else:
            self.pcas = [DummyPCA() for _ in range(self.d_variables)]

        if isinstance(self.feature_transform, list):
            phi_x = np.zeros(shape=(self.n_features*self.d_variables, n_samples))
            for i in range(self.d_variables):
                features = self.feature_transform[i].fit_transform(X[i, :].reshape(-1, 1))
                features = self.std_scalars[i].fit_transform(features)
                features = self.pcas[i].fit_transform(features)

                phi_x[i*self.n_features:(i+1)*self.n_features, :] = features.T
        elif self.feature_transform is None:
            phi_x = np.zeros(shape=(self.d_variables, n_samples))
            for i in range(self.d_variables):
                features = self.std_scalars[i].fit_transform(X[i, :].reshape(-1, 1))
                phi_x[i, :] = features.T

        return phi_x

    def transform(self, X: np.array):

        n_samples = X.shape[1]
        if isinstance(self.feature_transform, list):
            phi_x = np.zeros(shape=(self.n_features*self.d_variables, n_samples))

            for i in range(self.d_variables):
                features = self.feature_transform[i].transform(X[i, :].reshape(-1, 1))
                features = self.std_scalars[i].transform(features)
                features = self.pcas[i].transform(features)

                phi_x[i*self.n_features:(i+1)*self.n_features, :] = features.T
        elif self.feature_transform is None:
            phi_x = np.zeros(shape=(self.d_variables, n_samples))
            for i in range(self.d_variables):
                features = self.std_scalars[i].transform(X[i, :].reshape(-1, 1))

                phi_x[i, :] = features.T

        return phi_x


class KernelizedPermutationEstimator(permutationMixin, BaseEstimator):
    """
    Class to perform the permutation estimation, 
    modelled after Scikit-Learn Estimator class
    """
    def __init__(
            self, 
            regularizer: str, 
            optim_kwargs: Dict, 
            kernel: str,
            parameter: float,
            d_variables: int,
            n_features: int,
            groups=None,
            two_stage: Union[str, None]=None
    ) -> None:
        self.regularizer = regularizer
        assert 'alpha' in optim_kwargs, "No regularization parameter found, this option must be supplied."
        self.optim_kwargs = copy(optim_kwargs)
        self.kernel = kernel
        self.parameter = parameter
        self.d_variables = d_variables
        self.n_features = n_features
        self.groups = groups
        self.two_stage = two_stage

    def fit(self, X, y, is_fit_transformed=False, recover_weights=True):
        start = time()
        # Apply feature transform
        if not is_fit_transformed:
            phi_x_full = self.fit_transform(X)
        else:
            phi_x_full = self.transform(X)

        optim = self._setup_optim(
            regularizer=self.regularizer,
            optim_kwargs=self.optim_kwargs,
            dim=self.d_variables, 
            groups=self.groups,
            n_features=self.n_features,
        )
        self._optim = optim

        res = self._perform_fit(
            optim=optim,
            phi_x=phi_x_full, 
            y=y, 
            n_features=self.n_features,
            recover_weights=recover_weights
        )
        stop = time()
        res["time_match"] = stop - start
        
        if self.regularizer != "logistic_group":
            start = time()
            self.perm_hat_corr_ = self._get_corr_perm(X, y, groups=self.groups)
            stop = time()
            res["time_corr"] = stop - start

            start = time()
            self.perm_hat_spr_ = self._get_spr_perm(X, y, groups=self.groups)
            stop = time()
            res["time_spear"] = stop - start 

            self.perm_hat_max_ = res["perm_hat_max"]
            self.perm_hat_match_ = res["perm_hat_match"]
            self.beta_hat_max_ = res["beta_hat_max"]
            self.beta_hat_match_ = res["beta_hat_match"]
            self.beta_hat_all_ = res["beta_hat_all"]

            self.intercept_ = self._optim.intercept_

            res["non_zero_count"] = np.sum(~np.isclose(self.beta_hat_all_, 0, atol=1e-6))

            res["perm_hat_spr"] = self.perm_hat_spr_
            res["perm_hat_corr"] = self.perm_hat_corr_
        else:
            self.perm_hat_max_ = res["perm_hat_max"]
            self.perm_hat_match_ = res["perm_hat_match"]
            self.beta_hat_max_ = res["beta_hat_max"]
            self.beta_hat_match_ = res["beta_hat_match"]
            self.beta_hat_all_ = res["beta_hat_all"]

            res["non_zero_count"] = np.sum(~np.isclose(self.beta_hat_all_, 0, atol=1e-6))


        return res

    def fit_transform(self, X: np.array):
        n_samples = X.shape[1]
        self.nystroms = [
            Nystroem(
                kernel=self.kernel, 
                gamma=self.parameter,
                n_components=self.n_features
            ) for _ in range(self.d_variables)
        ]

        phi_x = np.zeros(shape=(self.n_features*self.d_variables, n_samples))
        for i in range(self.d_variables):
            features = self.nystroms[i].fit_transform(X[i, :].reshape(-1, 1))

            phi_x[i*self.n_features:(i+1)*self.n_features, :] = features.T
        return phi_x

    def transform(self, X: np.array):
        n_samples = X.shape[1]

        phi_x = np.zeros(shape=(self.n_features*self.d_variables, n_samples))
        for i in range(self.d_variables):
            features = self.nystroms[i].transform(X[i, :].reshape(-1, 1))

            phi_x[i*self.n_features:(i+1)*self.n_features, :] = features.T
        return phi_x



class DummyPCA:
    def fit_transform(self, X):
        return X

    def transform(self, X):
        return X
