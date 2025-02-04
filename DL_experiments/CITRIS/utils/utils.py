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


def get_device():
    try:
        if torch.backends.mps.is_available():
            # For small networks and loads this cpu seems to be faster 
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    except AttributeError as e:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    return device


