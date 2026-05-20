import os
from pathlib import Path

import numpy as np

import yaml
from collections import OrderedDict

import seaborn as sns
from matplotlib import pyplot as plt


def safe_for_yaml(data):
    if isinstance(data, dict):
        # Recursively apply conversion to dictionary values
        return {k: safe_for_yaml(v) for k, v in data.items()}
    elif isinstance(data, (np.float64, np.float32)):
        return float(data)
    elif isinstance(data, (np.int64, np.int32)):
        return int(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    # Add other types if needed
    return data


def _resolve_direction(metric_name: str, directions: dict | None) -> str | None:
    """
    Returns 'max', 'min', or None for a given metric name.
    Looks up the directions dict with the full metric name first, then falls
    back to checking whether any key is a substring of the metric name
    (e.g. 'RMSE Test' matches the 'RMSE' key).
    Returns None if directions is not provided or no match is found.
    """
    if not directions:
        return None

    if metric_name in directions:
        return directions[metric_name]

    for key, val in directions.items():
        if key in metric_name:
            return val

    return None


def print_metrics(stats_dict, directions: dict | None = None):
    """
    Prints the statistics (mean, std, best) for each metric in stats_dict
    in a neatly aligned, color-highlighted table format.

    Args:
        stats_dict: OrderedDict of {metric_name: {mean, std, min, max}}
        directions: optional dict mapping metric name (or substring) to
                    'max', 'min', or None.
                    - 'max' --> column header 'Best (Max)', shows np.max
                    - 'min' --> column header 'Best (Min)', shows np.min
                    -  None --> prints '----' in the best column
                    When directions is None entirely, falls back to 'min'
                    for all metrics (original behaviour).
    """
    RESET = "\033[0m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    print(
        f"{BOLD}{'Metric':<25} {'Index':<8} {'Mean':>12} {'Std':>12} {'Best':>15}{RESET}"
    )
    print("-" * 82)

    for metric_type, metric_values in stats_dict.items():
        mean = np.array(metric_values["mean"]).flatten()
        std = np.array(metric_values["std"]).flatten()

        direction = _resolve_direction(metric_type, directions)

        # Compute the "best" value according to direction
        if direction == "max":
            best_vals = np.array(metric_values["max"]).flatten()
            best_label = "(Max)"
        elif direction == "min":
            best_vals = np.array(metric_values["min"]).flatten()
            best_label = "(Min)"
        else:
            best_vals = None
            best_label = None

        for i, (m, s) in enumerate(zip(mean, std)):
            metric_label = metric_type if i == 0 else ""

            if best_vals is not None:
                best_str = f"{GREEN}{best_vals[i]:>12.3f} {best_label:<3}{RESET}"
            else:
                best_str = f"{DIM}{'----':>16}{RESET}"

            print(
                f"{metric_label:<25} "
                f"{i:<8d} "
                f"{BLUE}{m:>12.3f}{RESET} "
                f"{YELLOW}{s:>12.3f}{RESET} "
                f"{best_str}"
            )

    print("-" * 82)


def store_metrics(
    stats_path,
    store_stats,
    *metric_lists,
    filename="general_stats",
    directions: dict | None = None,
    legacy_save=False,
):
    """
    Takes in arbitrary metric lists and calculates statistics (mean, std, min, max)
    for each list.  Now stores both min and max so print_metrics can use either.

    Args:
        directions: forwarded to print_metrics (see its docstring).
    """
    stats_dict = OrderedDict()

    for metric_list in metric_lists:
        metric_name = metric_list[0]
        values = metric_list[1]

        stats_dict[metric_name] = {
            "mean": np.mean(values, axis=0),
            "std": np.std(values, axis=0),
            "min": np.min(values, axis=0),
            "max": np.max(values, axis=0),
        }

    stats_dict = safe_for_yaml(stats_dict)
    print_metrics(stats_dict, directions=directions)

    if store_stats:
        # Build a save-safe copy that omits 'max' to preserve the legacy schema
        save_dict = stats_dict
        if legacy_save:
            save_dict = OrderedDict(
                {
                    metric: {k: v for k, v in values.items() if k != "max"}
                    for metric, values in stats_dict.items()
                }
            )
            save_dict = safe_for_yaml(save_dict)
        _stats_path = Path(stats_path)
        os.makedirs(_stats_path, exist_ok=True)
        with open(os.path.join(_stats_path, f"{filename}.yaml"), "w") as stats_f:
            yaml.safe_dump(save_dict, stats_f, sort_keys=False)


def plot_calibration_curve(
    results_np,
    index_quantiles,
    stats_path,
    filename,
    title="Calibration Curve",
    palette_name="Navy",
):
    PALETTES = {
        "Navy": {"line": "#1B3A6B", "band": "#3A6BC4", "dot": "#FFFFFF"},
        "Carmine": {"line": "#8B0000", "band": "#D64545", "dot": "#FFFFFF"},
        "Evergreen": {"line": "#1B4D3E", "band": "#529471", "dot": "#FFFFFF"},
        "Amber": {"line": "#C99700", "band": "#FFD24A", "dot": "#1A1A1A"},
        "Amethyst": {"line": "#4B0082", "band": "#8A5BB1", "dot": "#FFFFFF"},
    }

    colors = PALETTES.get(palette_name, PALETTES["Navy"])

    # Set publication-style parameters
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 12,
            "axes.linewidth": 1.2,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
        }
    )

    fig, ax = plt.subplots(figsize=(7, 7), dpi=300)

    ax.plot(
        [0, 1],
        [0, 1],
        color="#7F8C8D",
        linestyle="--",
        linewidth=1.5,
        label="Perfectly Calibrated",
        zorder=1,
    )

    # Calibration Deviation (Shaded Area)
    ax.fill_between(
        index_quantiles,
        index_quantiles,
        results_np,
        color=colors["band"],
        alpha=0.15,
        label="Calibration Deviation",
        zorder=2,
    )

    # Observed Coverage Line
    ax.plot(
        index_quantiles,
        results_np,
        color=colors["line"],
        linewidth=2,
        marker="o",
        markersize=7,
        markerfacecolor=colors["dot"],
        markeredgewidth=1.5,
        label="Observed Coverage",
        zorder=3,
    )

    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    ax.set_xlabel("Expected Quantile ($q$)", fontsize=13, labelpad=10)
    ax.set_ylabel("Observed Coverage", fontsize=13, labelpad=10)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle=":", alpha=0.6, color="#BDC3C7", zorder=0)

    ax.legend(frameon=False, loc="upper left", fontsize=11)

    save_path = os.path.join(stats_path, f"{filename}.pdf")
    plt.savefig(save_path, bbox_inches="tight", transparent=True)
    plt.close()


def plot_histogram(
    distribution, stats_path, filename, title="Distribution of Predicted Uncertainty"
):
    plt.figure(figsize=(10, 6))

    sns.histplot(
        distribution,
        kde=False,
        color="seagreen",
        edgecolor="white",
        linewidth=0.8,
        stat="density",  # Ensures the y-axis is normalized
        # kde_kws={'cut': 3} is included in the original but irrelevant when kde=False
    )
    plt.axvline(np.mean(distribution), color="red", linestyle="--", linewidth=2)

    plt.text(
        0.95,
        0.95,
        f"Sharpness = {np.mean(distribution):.2f} $\\pm$ {np.std(distribution):.2f}",
        transform=plt.gca().transAxes,
        ha="right",  # Horizontal alignment: right edge of text at 0.95
        va="top",  # Vertical alignment: top edge of text at 0.95
        fontsize=12,
        bbox=dict(
            boxstyle="round,pad=0.5", fc="yellow", alpha=0.1
        ),  # Added some transparency (0.1)
    )

    plt.title(title, fontsize=14)
    plt.xlabel("Predicted Uncertainty", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(
        f"{os.path.join(stats_path, filename)}.pdf", dpi=300, bbox_inches="tight"
    )
    plt.close()
