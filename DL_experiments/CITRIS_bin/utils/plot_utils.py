import os
import functools
import traceback
import logging
from itertools import product
from typing import List, Dict, Iterable, Union

import duckdb
import pandas as pd
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({
    "lines.markersize": 3.5, 
    "lines.linewidth": 1.2,
    "text.usetex": True, 
    "font.family": "serif",
    "font.size": 10
})
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

LINESTYLES = ['solid', 'dotted', 'dashdot']
MARKERS = ['v', 's', '*', 'p']
# Colour blind friendly colours (Okabe and Ito)according to
# https://www.nceas.ucsb.edu/sites/default/files/2022-06/Colorblind%20Safe%20Color%20Schemes.pdfa
COLOURS = [
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#D55E00",
    "#F0E442",
    "#0072B2",
    "#CC79A7",
    "#000000"
]
FACECOLOUR = "#E5E5E5"

def set_ax_size(w, h, ax):
    """ w, h: width, height in inches """

    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)

def save_fig(
    fig: plt.Figure,
    save_dir: Union[str, bytes, os.PathLike]
) -> None:
    fig.savefig(
        f"{save_dir}.pdf", 
        dpi=400,
        transparent=False,
        bbox_inches="tight"
    )
    fig.savefig(
        f"{save_dir}.png", 
        dpi=400, 
        transparent=False, 
        bbox_inches="tight"
    )

def check_save_fig(
        fig: plt.Figure,
        checkpoint_dir: Union[str, bytes, os.PathLike], 
        fname: str
    ) -> None:
    """
    Utility function to save figures
    """
    fig_path = os.path.join(checkpoint_dir, "figures")
    os.makedirs(fig_path, exist_ok=True)

    save_fig(fig, save_dir=fig_path)


def reorder_legend(elements, n_cols):
    # order_idx = [0, 4, 1, 5, 2, 6, 3]
    n_rows = (len(elements) // n_cols) + 1
    order_idx = list(range(n_cols * n_rows))
    idx = 0
    for i in range(n_cols):
        for j in range(n_rows):
            step = j * n_cols
            order_idx[idx] = i + step
            idx += 1
    return order_idx[:len(elements)]


def pull_data_duckdb(
    x_axis: str,
    y_axis: str, 
    features: List[str],
    settings: Dict,
    start_date: str,
    end_date: str,
    experiment='spline',
) -> pd.DataFrame:
    settings_str = [f"{setting}={settings[setting]}" for setting in settings]
    settings_str = " AND ".join(settings_str)
    with duckdb.connect("data/experiments.duckdb") as con:

        def round_dec(x: float) -> float:
            return round(x, 4)

        con.create_function("round_dec", round_dec)
        if features:
            features_str = ",".join(features)
            select_str = f"{features_str},{x_axis},{y_axis},seed,regularizer"
        else:
            select_str = f"{x_axis}, {y_axis},seed,regularizer"
        result = con.sql(f"""
            SELECT 
            {select_str}
            FROM experiments_{experiment}
            WHERE {settings_str} 
            AND date_trunc('day', performed) >= '{start_date}'
            AND date_trunc('day', performed) <= '{end_date}';
            """).df()
    result_agg = result.groupby([x_axis, *features, "regularizer"]).agg(
        mean=(y_axis, np.mean),
        lb=(y_axis, lambda x: np.percentile(x, 25)),
        ub=(y_axis, lambda x: np.percentile(x, 75))
    )
    return result_agg

def create_legend(
    experiment: str, 
    feature_combinations: List, 
    regularizers,
    linestyles=LINESTYLES, 
    markers=MARKERS, 
    colours=COLOURS
) -> plt.Figure: 

    legend_elements = []
    lw = 1.2

    for k, reg in enumerate(regularizers):
        legend_elements.append(
                Line2D([0], [0], color='k', lw=lw, label=reg, linestyle=linestyles[k], marker=markers[k]),
            )
        
    legend_elements.append(
        Line2D([0], [0], color='k', lw=lw, label="Pearson", linestyle="solid", marker=markers[2]),
    )
    legend_elements.append(
        Line2D([0], [0], color='k', lw=lw, label="Spearman", linestyle="solid", marker=markers[3]),
    )

    if experiment == "spline":
        for i, (knot, deg) in enumerate(feature_combinations):
            legend_elements.append(
                Line2D([0], [0], color=colours[i], lw=lw, label=f"{knot} knots \& {deg} degrees", linestyle='solid'),
                )
    elif experiment == "features":
        for i, p in enumerate(feature_combinations):
            legend_elements.append(
                Line2D([0], [0], color=colours[i], lw=lw, label=f"{p[0]} Fourier features", linestyle='solid'),
                )       
    elif experiment == "kernel":
        for i, k in enumerate(feature_combinations):
            legend_elements.append(
                Line2D([0], [0], color=colours[i], lw=lw, label=f"{k[0]}", linestyle='solid'),
                )

    legend_elements.append(
        Line2D([0], [0], color=colours[5], lw=lw, label="Linear", linestyle=linestyles[-1])
    )

    fig = plt.figure(figsize=(4,3))
    fig.legend(handles=legend_elements, loc='center', ncol=5, prop={"size": 5})
    return fig


def add_corr_plot(ax, data_kwargs, feat, i):
    result_agg = pull_data_duckdb(**data_kwargs)

    result_agg = result_agg.xs(feat, level=data_kwargs["features"])
    result_agg = result_agg.xs("group", level="regularizer")
    x = result_agg.index
    y = result_agg["mean"]
    lb = result_agg["lb"]
    ub = result_agg["ub"]

    ax.plot(x, y, color="k", marker=MARKERS[i])
    ax.fill_between(x, lb, ub, alpha=.15, color="k")
    i += 1 

def add_linear_plot(ax, data_kwargs, feat):
    result_agg = pull_data_duckdb(**data_kwargs)

    result_agg = result_agg.xs(feat, level=data_kwargs["features"])
    result_agg = result_agg.xs("group", level="regularizer")
    x = result_agg.index
    y = result_agg["mean"]
    lb = result_agg["lb"]
    ub = result_agg["ub"]

    ax.plot(x, y, color=COLOURS[5], marker=MARKERS[0], linestyle=LINESTYLES[-1])
    ax.fill_between(x, lb, ub, alpha=.15, color=COLOURS[5])


def create_title_from_settings(method: str, settings: Dict) -> str:
    latex_map = {
        "alpha": "$\\lambda$", 
        "entanglement": "$\\rho$",
        "d_variables": "$d$", 
        "miss_well": "spec",
        "n_total": "$n$"
    }

    half = (len(settings) // 2) + (len(settings) % 2)

    settings_fname = [f"{setting}={settings[setting]}" for setting in settings]
    settings_list = [f"{latex_map[setting]}={settings[setting]}" for setting in settings]
    settings_list = settings_list[:half] + ["\n"] + settings_list[half:]
    title = f" ".join(settings_list)
    fname = "_".join(settings_fname)
    return title, fname


def continue_on_error(default_return=None):
    """
    A decorator that catches any exceptions raised by the decorated function,
    optionally logs them, and allows the script to continue execution.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                plt.close()
                print(f"Error in {func.__name__}: {str(e)}")
                print(f"Traceback: {traceback.format_exc()}")
                return default_return
        return wrapper
    return decorator


@continue_on_error(default_return=-1)
def create_synthetic_fig(
    x_axis: str,
    y_axis: str,
    regularizers: List[str],
    features: List[str],
    settings: Dict,
    start_date: str,
    end_date: str,
    experiment='spline',
) -> None:
    
    result_agg = pull_data_duckdb(
        x_axis, 
        y_axis, 
        features,
        settings,
        start_date, 
        end_date,
        experiment
    )

    fig, ax = plt.subplots()

    ax.set_facecolor(FACECOLOUR)
    ax.grid(color="white")

    feature_combinations = []
    for feature in features:
        feature_idx = result_agg.index.get_level_values(feature).unique().to_list()
        feature_combinations.append(feature_idx)
    feature_combinations = list(product(*feature_combinations))
    for i, comb in enumerate(feature_combinations):
        if x_axis == "alpha" and settings["miss_well"] == 'true':
            if experiment == "spline":
                n_features = (comb[0] + comb[1] - 1)
            elif experiment == "features":
                n_features = comb[0]
            elif experiment == "linear":
                n_features = 1
            elif experiment == "kernel":
                n_features = 0
        if comb:
            result_group = result_agg.xs(comb, level=features)
        else:
            result_group = result_agg
        for k, reg in enumerate(regularizers):
            result_plot = result_group.xs(reg, level="regularizer")

            x = result_plot.index
            y = result_plot["mean"]
            lb = result_plot["lb"]
            ub = result_plot["ub"]

            ax.plot(x, y, color=COLOURS[i], linestyle=LINESTYLES[k], marker=MARKERS[k])
            ax.fill_between(x, lb, ub, color=COLOURS[i], alpha=.25)
    
    if y_axis == "perm_error_match":
        data_kwargs = {
            "x_axis": x_axis,
            "features": features,
            "settings": settings,
            "start_date": start_date,
            "end_date": end_date,
            "experiment": experiment,
        }
        i = len(regularizers) 
        if experiment == "spline":
            data_kwargs["y_axis"] = "perm_error_corr"
            add_corr_plot(ax, data_kwargs, (8, 3), i)

            data_kwargs["y_axis"] = "perm_error_spear"
            add_corr_plot(ax, data_kwargs, (8, 3), i)

            data_kwargs["y_axis"] = "perm_error_linear"
            add_linear_plot(ax, data_kwargs, (8, 3))
        elif experiment == "features":
            data_kwargs["y_axis"] = "perm_error_corr"
            add_corr_plot(ax, data_kwargs, (8,), i)

            data_kwargs["y_axis"] = "perm_error_spear"
            add_corr_plot(ax, data_kwargs, (8,), i)

            data_kwargs["y_axis"] = "perm_error_linear"
            add_linear_plot(ax, data_kwargs, (8,))
        elif experiment == "kernel":
            data_kwargs["y_axis"] = "perm_error_corr"
            add_corr_plot(ax, data_kwargs, ("laplacian",), i)

            data_kwargs["y_axis"] = "perm_error_spear"
            add_corr_plot(ax, data_kwargs, ("laplacian",), i)

            data_kwargs["y_axis"] = "perm_error_linear"
            add_linear_plot(ax, data_kwargs, ("laplacian",))


    elif y_axis == "time_match":
        data_kwargs = {
            "x_axis": x_axis,
            "features": features,
            "settings": settings,
            "start_date": start_date,
            "end_date": end_date,
            "experiment": experiment,
        }
        i = len(regularizers) 
        if experiment == "spline":
            data_kwargs["y_axis"] = "time_corr"
            add_corr_plot(ax, data_kwargs, (8, 3), i)

            data_kwargs["y_axis"] = "time_spear"
            add_corr_plot(ax, data_kwargs, (8, 3), i)

            data_kwargs["y_axis"] = "time_linear"
            add_linear_plot(ax, data_kwargs, (8, 3))
        elif experiment == "features":
            data_kwargs["y_axis"] = "time_corr"
            add_corr_plot(ax, data_kwargs, (8,), i)

            data_kwargs["y_axis"] = "time_spear"
            add_corr_plot(ax, data_kwargs, (8,), i)

            data_kwargs["y_axis"] = "time_linear"
            add_linear_plot(ax, data_kwargs, (8,))
        elif experiment == "kernel":
            data_kwargs["y_axis"] = "time_corr"
            add_corr_plot(ax, data_kwargs, ("laplacian",), i)

            data_kwargs["y_axis"] = "time_spear"
            add_corr_plot(ax, data_kwargs, ("laplacian",), i)

            data_kwargs["y_axis"] = "time_linear"
            add_linear_plot(ax, data_kwargs, ("laplacian",))

    elif y_axis == "r2_match":
        data_kwargs = {
            "x_axis": x_axis,
            "features": features,
            "settings": settings,
            "start_date": start_date,
            "end_date": end_date,
            "experiment": experiment,
        }
        i = len(regularizers) 
        data_kwargs["y_axis"] = "r2_linear"
        if experiment == "spline":
            add_linear_plot(ax, data_kwargs, (8, 3))
        elif experiment == "features":
            add_linear_plot(ax, data_kwargs, (8,))
        elif experiment == "kernel":
            add_linear_plot(ax, data_kwargs, ("laplacian",))

    title, fname = create_title_from_settings(y_axis, settings)

    ax.set_title(title, fontsize=8)
    if 'perm' in y_axis:
        ax.set_ylim((-0.05, 1.05))
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_ylabel("Permutation Error")
    elif 'r2' in y_axis:
        ax.set_ylim((-0.05, 1.05))
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_ylabel("$R^2$-score")
    elif 'mse' in y_axis:
        ax.set_ylim((0, 100))
        # ax.set_yscale('log')
        ax.set_ylabel("MSE")
    elif 'time' in y_axis:
        ax.set_ylim((1e-3, 1e4))
        ax.set_yscale("log")
        ax.set_ylabel("Time ($s$)")

    if x_axis == "alpha":
        ax.set_xlabel("Regularization ($\\lambda$)")
        ax.set_xscale("log")
        ax.set_xlim((5e-4, 2e0))
        ax.set_xticks([1e-3, 1e-2, 1e-1, 1e0])
    elif x_axis == "d_variables":
        ax.set_xlabel("Dimension ($d$)")
        ax.set_xlim((0, 110))
        ax.set_xticks([20,40, 60, 80, 100])
    elif x_axis == "entanglement":
        ax.set_xlabel("Correlation ($\\rho$)")
        ax.set_xlim(-0.05, 1.1)
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    elif x_axis == "n_total":
        ax.set_xlabel("Nr. data points ($n$)")
        ax.set_xscale("log")
    else:
        ax.set_xlabel(f"{x_axis}")


    os.makedirs(f"figs/{experiment}/{x_axis}/{y_axis}", exist_ok=True)
    save_dir_wide = f"figs/{experiment}/{x_axis}/{y_axis}/{fname}_wide"
    save_dir_tall = f"figs/{experiment}/{x_axis}/{y_axis}/{fname}"

    cm = 1/2.54  # centimeters in inches
    set_ax_size(4.3*cm, 2.6*cm, ax)
    save_fig(fig, save_dir_wide)

    set_ax_size(4*cm, 4*cm, ax)
    save_fig(fig, save_dir_tall)

    fig.clf()
    plt.close(fig)

    legend = create_legend(
        experiment=experiment,
        feature_combinations=feature_combinations,
        regularizers=regularizers,
        linestyles=LINESTYLES,
        markers=MARKERS,
        colours=COLOURS
    )
    save_fig(legend, f"figs/{experiment}/legend")
    legend.clf()
    plt.close(legend)

def create_diffeomorphism_plots(
    diffeomorphism: List[callable],
    colours: List[str]=COLOURS
) -> None:
    x_min, x_max = -4, 4
    y_min, y_max = -3, 3
    x = np.linspace(x_min, x_max, num=1000)

    fig, ax = plt.subplots()
    ax.set_facecolor(FACECOLOUR)
    ax.grid(color="white")
    ax.set_xlim((x_min, x_max))
    ax.set_ylim((y_min, y_max))

    for i, f in enumerate(diffeomorphism):
        ax.plot(x, f(x), color=colours[i])

    cm = 1/2.54  # centimeters in inches
    set_ax_size(7*cm, 3*cm, ax)

    save_dir = "figs/diffeomorphisms"
    os.makedirs(save_dir, exist_ok=True)
    save_fig(fig, os.path.join(save_dir, "diffeomorphisms"))
    fig.clf()
    plt.close(fig)
    
