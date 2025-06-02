import os 
import json
import logging
import random
import torch

from typing import Union
from datetime import datetime
from argparse import ArgumentParser

import numpy as np
from sklearn.linear_model import LinearRegression


def save_settings(args, ckpt_dir):
    # Save the settings of this experiment
    with open(os.path.join(ckpt_dir, "experiment_settings.json"), "w") as f:
        json.dump(vars(args), f, indent=4)


def get_basic_parser():
    """
    Returns argument parser of standard hyperparameters/experiment arguments.
    """
    parser = ArgumentParser()
    parser.add_argument('--N_experiment', type=int, default=12500,
                        help='number of data points.')
    parser.add_argument('--test_frac', type=float, default=0.2,
                        help='Fraction to use as a test set.')
    parser.add_argument('--seed', type=int, default=110,
                        help='Random seed for the experiments.')
    parser.add_argument('--d_variables', type=int, default=100, 
                        help='How many dimensions should be used between the alignments.')
    parser.add_argument('--specified', type=str,
                        choices=["miss", "well"], 
                        default="miss", 
                        help="If the model is well- or misspecified")
    parser.add_argument('--entanglement', type=float, default=0, 
                        help='How big the covariance is between the individual components of x.')
    parser.add_argument('--alpha', type=float, default=0.1, 
                        help="The regularization parameter for the Lasso optimization.")
    parser.add_argument('--regularizer', type=str,
                        default='group', 
                        choices=['lasso', 'group', 'elastic', 'ridge'], 
                        help="Type of regularizer to be used")
    parser.add_argument('--parallel', type=int, 
                        choices=[0, 1], default=0,
                        help="If the permutation is estimated directly or per dimension.")


    return parser


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def setup_checkpoint_dir(
        prefix: str, 
        time_str: bool=True
) -> Union[str, bytes, os.PathLike]:
    # Setup checkpoint dir to log settings and store results
    if time_str: 
        current_date = datetime.now()

        checkpoint_dir = "experiment_checkpoints/{}_{:02d}_{:02d}_{:02d}__{:02d}_{:02d}/".format(
            prefix,
            current_date.year, current_date.month, current_date.day,
            current_date.hour, current_date.minute
        )
    else:
        checkpoint_dir = "experiment_checkpoints/{}/".format(
            prefix
        )
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.join(checkpoint_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(checkpoint_dir, "data"), exist_ok=True)

    return checkpoint_dir


def build_dataset_dir(
        data_folder, 
        N, img_shape, hor_ver_split, 
        red_hor_cor, red_ver_cor
) -> os.PathLike:

    data_dir_name = f"ColoredBars_N{N}" \
        f"_IMG{img_shape}" \
        f"_SPLIT{hor_ver_split}" \
        f"_RHC{red_hor_cor}" \
        f"_RVC{red_ver_cor}"
    
    dir_name = os.path.join(data_folder, data_dir_name)
    return dir_name

def get_device():
    if torch.backends.mps.is_available():
        # For small networks and loads this cpu seems to be faster 
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    return device


