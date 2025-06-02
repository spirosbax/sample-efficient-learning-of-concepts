from tqdm import tqdm
from itertools import product
import os
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import matplotlib.pyplot as plt

from experiment.spline_settings import (
    dim_params, 
    n_total_params
)
from utils.plot_utils import create_ablation_plot, create_legend_ablation, save_fig

START_DATE = '2025-05-18'
END_DATE = '2025-05-30'

y_axis = ["acc_label", "acc_concept", "time", "ois_concept", "nis_concept"]

plots = [
    ("linear", [], [()]), 
    ("features", ['p_features'], [(8,)]), 
    ("spline", ['n_knots', 'degree'], [(8, 3)]), 
    ("kernel", ['kernel'], [("laplacian",)]), 
    ]

cbms = [
    ("CBM", "experiments_cbm"),
    ("CEM", "experiments_cem"),
    ("HardCBM", "experiments_cbm_ar"),
]

legend = create_legend_ablation(plots, cbms)
save_fig(legend, "figs/binary/ablation/legend")
legend.clf()
plt.close()



for y in y_axis:
    for a, e, n in dim_params:
        create_ablation_plot(
            plots=plots,
            cbms=cbms,
            x_axis="d_variables",
            y_axis=y, 
            settings={"alpha": a, "n_total": n},
            start_date=START_DATE,
            end_date=END_DATE
        )

    for d, a, e in n_total_params:
        create_ablation_plot(
            plots=plots,
            cbms=cbms,
            x_axis="n_total",
            y_axis=y, 
            settings={"alpha": a, "d_variables": d},
            start_date=START_DATE,
            end_date=END_DATE
        )


