"""
rcParams / tueplots
-------------------
tueplots is used for everything *except* font sizes, which come from the
caller's default rcParams (i.e. whatever is already set in matplotlibrc or
by the surrounding script). This means the figure width, layout padding,
and font family match the target venue exactly, while font sizes stay under
your control.

If tueplots is not installed the script falls back to sane hard-coded defaults.

Available bundles (pass to --bundle):
    neurips2024   NeurIPS 2024  (single column, ~5.5 in)
    icml2024      ICML 2024     (~3.25 in, half-column)
    iclr2024      ICLR 2024
    aistats2023   AISTATS 2023
    tmlr2023      TMLR 2023
    jmlr2001      JMLR

Example
-------
python plot_raw_metrics.py \\
    --dataset_dirs results/boston results/concrete results/energy \\
    --dataset_labels Boston Concrete Energy \\
    --metric "Test NLL" \\
    --models "ensemble" "dropout" "vi" \\
    --models "ensemble" "dropout" \\
    --models "ensemble" "vi" \\
    --draw_arrows ensemble dropout \\
    --bundle neurips2024 \\
    --out figure1.pdf
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import matplotlib as mplt
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import yaml

# ── tueplots ─────────────────────────────────────────────────────────────────
try:
    from tueplots import bundles as _tp_bundles

    _HAS_TUEPLOTS = True
except ImportError:
    _HAS_TUEPLOTS = False

# ── Colorblind-safe palette (Okabe–Ito) ──────────────────────────────────────
_PALETTES = [
    {"color": "#E69F00", "edge": "#9A6A00", "marker": "o"},  # orange
    {"color": "#56B4E9", "edge": "#1A7AAF", "marker": "s"},  # sky blue
    {"color": "#009E73", "edge": "#006B4E", "marker": "^"},  # green
    {"color": "#CC79A7", "edge": "#8C3E6E", "marker": "D"},  # pink
    {"color": "#0072B2", "edge": "#004D80", "marker": "v"},  # blue
    {"color": "#D55E00", "edge": "#943F00", "marker": "P"},  # vermillion
    {"color": "#F0E442", "edge": "#A89B2E", "marker": "*"},  # yellow
    {"color": "#000000", "edge": "#555555", "marker": "X"},  # black
]

_BG = "#FFFFFF"
_GRID = "#E2E2E2"

DATASET_MAP = {
    "pems": ("regression", 75),
    "cora": ("classification", 1000),
    "citeseer": ("classification", 1000),
    "chameleon": ("regression", 456),
    "gapsmallqm9": ("regression", 818),
    "artnetviews": ("regression", 12791),
    "tolokers2": ("classification", 2974),
}

# ── rcParams helpers ──────────────────────────────────────────────────────────

# Font-size keys that belong to the caller — we never let tueplots clobber them.
_FONT_SIZE_KEYS = {
    "font.size",
    "axes.labelsize",
    "axes.titlesize",
    "xtick.labelsize",
    "ytick.labelsize",
    "legend.fontsize",
    "legend.title_fontsize",
}

# Line / tick geometry we always enforce on top of the bundle.
_GEOMETRY_RC: dict = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "lines.linewidth": 0.9,
}

MID_ARROW_LABEL = True

def _get_bundle_rc(bundle_name: str, usetex: bool) -> dict:
    """Return tueplots rcParams for *bundle_name*, stripped of all font-size keys."""
    if not _HAS_TUEPLOTS:
        print(
            "Warning: tueplots not installed — using fallback layout "
            "(pip install tueplots for venue-accurate figure widths)."
        )
        return {
            "font.family": "serif",
            "figure.figsize": (5.5, 3.4),
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 10,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "figure.constrained_layout.use": True,
            "figure.autolayout": False,
            "savefig.pad_inches": 0.015,
        }

    fn = getattr(_tp_bundles, bundle_name, None)
    if fn is None:
        available = [x for x in dir(_tp_bundles) if not x.startswith("_")]
        raise ValueError(f"Unknown bundle {bundle_name!r}. Available: {available}")

    rc = fn()

    # Drop every font-size key — caller controls those.
    for key in list(rc):
        if key in _FONT_SIZE_KEYS:
            del rc[key]

    # text.usetex: bundles default to True (needs type1ec.sty).
    rc["text.usetex"] = usetex
    if not usetex:
        rc.pop("text.latex.preamble", None)
        rc.setdefault("font.family", "serif")

    return rc


def _column_width_from_bundle(bundle_name: str) -> float:
    """Return the figure width (inches) declared by the bundle, or 5.5 as fallback."""
    if not _HAS_TUEPLOTS:
        return 5.5
    fn = getattr(_tp_bundles, bundle_name, None)
    if fn is None:
        return 5.5
    return float(fn()["figure.figsize"][0])


def _figure_size(
    bundle_name: str,
    n_datasets: int,
) -> tuple[float, float]:
    """Compute (width, height) optimised for tight paper column budgets."""
    col_w = _column_width_from_bundle(bundle_name)

    # Tight vertical budget
    top_margin = 0.15  # NOTE this was originally 0.15
    per_row = 0.72  # NOTE this was originally 0.52
    bot_margin = 0.40
    height = top_margin + n_datasets * per_row + bot_margin
    height = max(height, 1.8)

    return col_w, height


# ── Data helpers ──────────────────────────────────────────────────────────────


def discover_stats_files(
    dataset_dir: str, model_name: str, model_size: str
) -> tuple[str, int] | None:
    """Find the stats file for a given model within a dataset directory.

    Looks in dataset_dir/model_name/ for files matching *_stats_{model_size}x{n_reps}*.yaml
    Returns (file_path, n_reps) or None if no matching file is found.
    """
    model_dir = os.path.join(dataset_dir, model_name)
    if not os.path.isdir(model_dir):
        return None

    pattern = re.compile(rf"^.+_stats_{model_size}x(\d+).*\.yaml$")
    for entry in os.scandir(model_dir):
        m = pattern.match(entry.name)
        if m and entry.is_file():
            n_reps = int(m.group(1))
            return (entry.path, n_reps)

    return None


def load_metric(stats_path: str, metric: str) -> dict:
    with open(stats_path) as f:
        data = yaml.safe_load(f)
    if metric not in data:
        raise KeyError(
            f"Metric '{metric}' not found in {stats_path}. "
            f"Available: {list(data.keys())}"
        )
    return data[metric]


def extract_stats(
    metric_data: dict, norm_val: int
) -> tuple[float, float, float, float]:
    """Extract mean, std, min, max from metric data.

    Returns (mean, std, min, max).
    """

    def _get(key):
        val = metric_data.get(key)
        if val is None:
            return None
        if isinstance(val, list):
            if len(val) == 1:
                return float(val[0])
            raise ValueError(f"Expected scalar or single-element list, got {val!r}")
        return float(val)

    mean = _get("mean")
    std = _get("std")
    min_val = _get("min")
    max_val = _get("max")

    if mean is None:
        raise ValueError("Metric data must contain 'mean' field")

    # Provide defaults if std/min/max are missing
    if std is None:
        std = 0.0
    if min_val is None:
        min_val = mean
    if max_val is None:
        max_val = mean

    return mean / norm_val, std / norm_val, min_val / norm_val, max_val / norm_val


def load_dataset(
    dataset_dir: str,
    models: list[str],
    metric: str,
    models_size: list[str],
) -> dict[str, tuple[float, float, float, float]]:
    """Load raw metric values for each model in the dataset.

    Looks for each model in dataset_dir/{model_name}/ subdirectories.
    Returns mapping: model_name → (mean, std, min, max)
    """
    result: dict[str, tuple[float, float, float, float]] = {}

    dataset_name = dataset_dir.split("/")[3]
    norm_val = 1
    if dataset_name in DATASET_MAP:
        ds_type, ds_val = DATASET_MAP[dataset_name]
        if ds_type == "regression" and metric in ("Diagonal NLL"):
            norm_val = ds_val

    for model, model_size in zip(models, models_size):
        file_info = discover_stats_files(dataset_dir, model, model_size)
        if file_info is None:
            raise FileNotFoundError(
                f"No stats file found for model '{model}' in {dataset_dir}/{model}/. "
                f"Expected pattern: *_stats_{{size}}x{{n_reps}}*.yaml"
            )
        path, n_reps = file_info
        metric_data = load_metric(path, metric)
        result[model] = extract_stats(metric_data, norm_val)

    return result


# ── Axis styling ──────────────────────────────────────────────────────────────


def _style_ax(ax, grid: bool = True) -> None:
    ax.set_facecolor(_BG)
    ax.tick_params(axis="both", colors="#333333", length=0)
    if grid:
        ax.minorticks_on()
        ax.yaxis.set_tick_params(which="minor", left=False)
        ax.xaxis.grid(True, color=_GRID, linewidth=0.7, linestyle="-")
        ax.xaxis.grid(
            True, which="minor", color=_GRID, linewidth=0.3, linestyle="-", alpha=0.5
        )
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(_GRID)
    ax.spines["bottom"].set_linewidth(0.8)


# ── Core drawing ──────────────────────────────────────────────────────────────


def _draw_series(
    ax,
    ys: np.ndarray,
    offsets: dict[str, float],
    model_order: list[str],
    model_palette: dict[str, dict],
    data: list[dict[str, tuple[float, float, float, float]]],
    n_datasets: int,
    models_label: dict[str, str],
) -> None:
    """Draw horizontal scatter + std/min-max whiskers for all models onto *ax*."""
    for model in model_order:
        pal = model_palette[model]
        off = offsets[model]

        means = np.full(n_datasets, np.nan)
        std_lower = np.full(n_datasets, np.nan)
        std_upper = np.full(n_datasets, np.nan)
        minmax_lower = np.full(n_datasets, np.nan)
        minmax_upper = np.full(n_datasets, np.nan)

        for ds_idx in range(n_datasets):
            if model in data[ds_idx]:
                mean, std, min_val, max_val = data[ds_idx][model]
                means[ds_idx] = mean
                std_lower[ds_idx] = mean - std
                std_upper[ds_idx] = mean + std
                minmax_lower[ds_idx] = min_val
                minmax_upper[ds_idx] = max_val

        py = ys + off

        # Min-max range (thick, transparent)
        ax.errorbar(
            means,
            py,
            xerr=[means - minmax_lower, minmax_upper - means],
            fmt="none",
            ecolor=pal["color"],
            elinewidth=3.5,
            alpha=0.25,
            capsize=0,
            zorder=2,
        )

        # ±std range (thin, opaque)
        ax.errorbar(
            means,
            py,
            xerr=[means - std_lower, std_upper - means],
            fmt="none",
            ecolor=pal["edge"],
            elinewidth=1.3,
            capsize=3.5,
            zorder=3,
        )

        # Mean marker
        ax.scatter(
            means,
            py,
            s=55,
            marker=pal["marker"],
            facecolors=pal["color"],
            edgecolors=pal["edge"],
            linewidths=1.2,
            zorder=4,
            label=models_label[model],
        )


# ── Arrow annotations ─────────────────────────────────────────────────────────


def _draw_arrows(
    ax,
    ys: np.ndarray,
    offsets: dict[str, float],
    data: list[dict[str, tuple[float, float, float, float]]],
    n_datasets: int,
    arrow_model_a: str,
    arrow_model_b: str,
) -> None:
    """Draw a double-headed arrow between modelA and modelB for every dataset row.

    The arrow sits just above the markers, annotated with "Δ = X.XX" showing
    the absolute horizontal difference between the two means.

    Rows where either model is absent are silently skipped.

    Parameters
    ----------
    ax            : the axes to draw on
    ys            : base y-positions (one per dataset row)
    offsets       : per-model y-offsets keyed by model name
    data          : list of per-dataset dicts mapping model → (mean, std, min, max)
    n_datasets    : number of dataset rows
    arrow_model_a : internal name of the first model  (e.g. "ensemble")
    arrow_model_b : internal name of the second model (e.g. "dropout")
    """
    fontsize = mplt.rcParams.get("legend.fontsize", 10) * 0.95
    arrow_color = "#444444"

    # Vertical clearance: sit above the highest per-model y-offset.
    max_off = 0  # max(offsets.values()) if offsets else 0.0
    arrow_lift = (
        max_off  # + 0.22   # rows are 1 data-unit apart; 0.22 gives breathing room
    )

    x_min, x_max = ax.get_xlim()
    for ds_idx in range(n_datasets):
        ds_data = data[ds_idx]

        if arrow_model_a not in ds_data or arrow_model_b not in ds_data:
            continue  # one or both models absent for this row — skip

        mean_a = ds_data[arrow_model_a][0]  # index 0 = mean
        mean_b = ds_data[arrow_model_b][0]

        x_lo = min(mean_a, mean_b)
        x_hi = max(mean_a, mean_b)
        delta = x_hi - x_lo

        y_arrow = ys[ds_idx] + arrow_lift

        # Double-headed arrow
        print(x_hi - x_lo)

        arrowstyle = (
            mplt.patches.ArrowStyle.BracketAB(widthA=0.85, widthB=0.85)
            if x_hi - x_lo >= 0.005
            else "-"
        )
        ax.annotate(
            "",
            xy=(x_hi, y_arrow),
            xytext=(x_lo, y_arrow),
            arrowprops=dict(
                arrowstyle=arrowstyle,
                color=arrow_color,
                lw=0.9,
                mutation_scale=5,
                shrinkA=0,
                shrinkB=0,
            ),
            zorder=6,
            clip_on=False,
        )

        # "Δ = X.XX" label just above the arrow midpoint
        text_delta_annot = 0.055
        print(x_lo - text_delta_annot * 2, x_min)
        if x_lo - text_delta_annot * 2 < x_min:
            x_mid = x_hi + text_delta_annot
        else:
            x_mid = x_lo - text_delta_annot  # (x_lo + x_hi) / 2.0
        if MID_ARROW_LABEL:
            x_mid = (x_lo + x_hi) / 2.0
            label_y = y_arrow - text_delta_annot - 0.17 # + 0.07
        else:
            label_y = y_arrow - text_delta_annot
        ax.text(
            x_mid,
            label_y,
            f"\u0394 = {np.round(delta, decimals=2):.2f}",
            color=arrow_color,
            fontsize=fontsize,
            ha="center",
            va="bottom",
            zorder=7,
            clip_on=False,
        )


# ── Main plot ─────────────────────────────────────────────────────────────────


def plot(
    *,
    dataset_dirs: list[str],
    dataset_labels: list[str],
    metrics: list[str],
    models_per_dataset: list[list[str]],
    models_size: list[list[str]],
    models_label: dict[str, str],
    out: str | None = None,
    bundle: str = "neurips2024",
    usetex: bool = False,
    arrow_models: tuple[str, str] | None = None,
) -> None:
    """Build and optionally save the raw metrics comparison figure.

    rcParams are applied in this order (later entries win on conflicts):
      1. tueplots bundle  — figure width, constrained layout, font family.
         All font-size keys are stripped so matplotlibrc values flow through.
      2. _GEOMETRY_RC     — thin lines, tick sizes, PDF font embedding.
    """
    # ── Apply rcParams ────────────────────────────────────────────────────────
    rc = {}
    rc.update(_get_bundle_rc(bundle, usetex))
    rc.update(_GEOMETRY_RC)
    mplt.rcParams.update(rc)

    n_datasets = len(dataset_dirs)

    # ── Load data ─────────────────────────────────────────────────────────────
    data: list[dict[str, tuple[float, float, float, float]]] = []

    for ds_dir, metric, models, model_size in zip(
        dataset_dirs, metrics, models_per_dataset, models_size
    ):
        ds_data = load_dataset(ds_dir, models, metric, model_size)
        data.append(ds_data)

    # ── Determine global model order and assign palettes ─────────────────────
    # Collect all unique models across all datasets, preserving first-seen order
    all_models: list[str] = []
    seen: set[str] = set()
    for models in models_per_dataset:
        for m in models:
            if m not in seen:
                all_models.append(m)
                seen.add(m)

    model_palette: dict[str, dict] = {}
    for i, model in enumerate(all_models):
        model_palette[model] = _PALETTES[i % len(_PALETTES)]

    # ── Compute offsets per model ─────────────────────────────────────────────
    n_models = len(all_models)
    cluster_w = min(0.50, 0.20 * n_models)
    if n_models == 1:
        offset_values = [0.0]
    else:
        offset_values = np.linspace(
            -cluster_w / 1.8, cluster_w / 1.8, n_models
        ).tolist()

    offsets: dict[str, float] = {}
    for model, offset in zip(all_models, offset_values):
        offsets[model] = offset

    ys = np.arange(n_datasets, dtype=float)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig_w, fig_h = _figure_size(bundle, n_datasets)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
    fig.patch.set_facecolor(_BG)

    # ── Style ─────────────────────────────────────────────────────────────────
    _style_ax(ax)

    # ── Data series ───────────────────────────────────────────────────────────
    _draw_series(
        ax, ys, offsets, all_models, model_palette, data, n_datasets, models_label
    )

    # ── Arrow annotations between two models ─────────────────────────────────
    if arrow_models is not None:
        model_a, model_b = arrow_models
        _draw_arrows(ax, ys, offsets, data, n_datasets, model_a, model_b)

    ylim = (-0.6, n_datasets - 0.4)

    # ── Determine x-axis limits ───────────────────────────────────────────────
    all_vals: list[float] = []
    for ds_data in data:
        for mean, std, min_val, max_val in ds_data.values():
            all_vals.extend([min_val, max_val])

    if all_vals:
        data_lo = min(all_vals)
        data_hi = max(all_vals)
        pad = max(0.05 * (data_hi - data_lo), 1e-6)
        ax.set_xlim(data_lo - pad, data_hi + pad)

    # ── x-axis ────────────────────────────────────────────────────────────────
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6, prune="both"))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    # ── y-axis ────────────────────────────────────────────────────────────────
    ax.set_yticks(ys)
    ax.set_yticklabels(dataset_labels)
    ax.set_ylim(*ylim)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color(_GRID)
    ax.spines["left"].set_linewidth(0.8)

    # ── x-label ───────────────────────────────────────────────────────────────
    unique_metrics = list(dict.fromkeys(metrics))
    metric_label = unique_metrics[0] if len(unique_metrics) == 1 else "Metric"
    # ax.set_xlabel(metric_label, labelpad=4)

    # ── Legend ────────────────────────────────────────────────────────────────
    handles, labels_leg = ax.get_legend_handles_labels()
    seen_leg: set[str] = set()
    h_dedup, l_dedup = [], []
    for h, l in zip(handles, labels_leg):
        seen_leg.add(l)
        h_dedup.append(h)
        l_dedup.append(l)

    is_lower_left = all(["Accuracy Test" == met for met in metrics])
    ax.legend(
        h_dedup,
        l_dedup,
        frameon=True,
        loc="upper right" if not is_lower_left else "lower left",
        ncols=len(h_dedup),
        framealpha=0.92,
        edgecolor=_GRID,
        facecolor=_BG,
        handletextpad=0.35,
        labelspacing=0.25,
        columnspacing=0.8,
        borderpad=0.45,
        handlelength=1.0,
    )

    if out:
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved to {out}")
    else:
        plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot raw metric values across datasets with mean, ±std, and min-max "
            "whiskers. Each model gets a unique marker and color."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset_dirs",
        nargs="+",
        required=True,
        metavar="DIR",
        help="One directory per dataset. Each should contain model subdirectories with stats files.",
    )
    parser.add_argument(
        "--dataset_labels",
        nargs="+",
        required=True,
        metavar="LABEL",
        help="Display name for each dataset (same order as --dataset_dirs).",
    )
    parser.add_argument(
        "--metric",
        nargs="+",
        required=True,
        metavar="METRIC",
        help=(
            "Metric key(s) inside the yaml files. Pass one value to share "
            "across all datasets, or one per dataset."
        ),
    )
    parser.add_argument(
        "--models",
        action="append",
        nargs="+",
        required=True,
        metavar="MODEL",
        help=(
            "Model names to plot for each dataset. Repeat --models once per dataset. "
            "Example: --models A B C --models A B --models A C"
        ),
    )
    parser.add_argument(
        "--models_size",
        action="append",
        nargs="+",
        required=True,
        metavar="MODELSIZE",
        help=(
            "Size of the model for each model. Repeat --models_size the same amount of --models. "
            "Example: --models A B C --models A B --models_size 10 5 1 --models_size 10 10"
        ),
    )
    parser.add_argument(
        "--models_label",
        nargs="+",
        required=True,
        help=(
            "The name of the models that will be plotted. "
            "Example: --models A B C --models A B --models_label A Ensemble B SVI C NUTS"
        ),
    )
    parser.add_argument(
        "--draw_arrows",
        nargs=2,
        default=None,
        metavar=("MODEL_A", "MODEL_B"),
        help=(
            "Draw a double-headed arrow with a Δ annotation between the two named "
            "models on every dataset row where both are present. Names must match "
            "the internal model directory names (not display labels). "
            "Example: --draw_arrows ensemble dropout"
        ),
    )
    parser.add_argument(
        "--bundle",
        default="neurips2024",
        metavar="BUNDLE",
        help=(
            "tueplots venue bundle for figure width and font family. "
            "Font sizes come from matplotlibrc. "
            "Options: neurips2024, icml2024, iclr2024, aistats2023, tmlr2023, jmlr2001. "
            "Default: neurips2024."
        ),
    )
    parser.add_argument(
        "--usetex",
        action="store_true",
        default=False,
        help=(
            "Render text with LaTeX (requires working LaTeX install with type1ec.sty)."
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output path (png / pdf / svg). Omit to display interactively.",
    )

    args = parser.parse_args()

    if len(args.dataset_dirs) != len(args.dataset_labels):
        parser.error(
            "--dataset_dirs and --dataset_labels must have the same number of entries."
        )

    n_ds = len(args.dataset_dirs)

    if len(args.metric) == 1:
        metrics = args.metric * n_ds
    elif len(args.metric) == n_ds:
        metrics = args.metric
    else:
        parser.error(
            f"--metric must receive 1 or {n_ds} values, got {len(args.metric)}."
        )

    if len(args.models) != n_ds:
        parser.error(
            f"--models must be specified {n_ds} times (once per dataset), "
            f"got {len(args.models)}."
        )
    if len(args.models_size) != n_ds:
        parser.error(
            f"--models_size must be specified {n_ds} times (once per dataset/model), "
            f"got {len(args.models_size)}."
        )

    unique_models = set()
    for _m in args.models:
        for m_m in _m:
            unique_models.add(m_m)

    if len(args.models_label) != len(unique_models) * 2:
        parser.error(
            f"--models_label must have {len(unique_models) * 2} elements "
            f"(alternating internal_name display_label per model), "
            f"got {len(args.models_label)}."
        )

    model_dict_label = {}
    for mli, old_name in enumerate(args.models_label[::2]):
        model_dict_label[old_name] = args.models_label[mli * 2 + 1]

    arrow_models = tuple(args.draw_arrows) if args.draw_arrows else None

    plot(
        dataset_dirs=args.dataset_dirs,
        dataset_labels=args.dataset_labels,
        metrics=metrics,
        models_per_dataset=args.models,
        models_size=args.models_size,
        models_label=model_dict_label,
        out=args.out,
        bundle=args.bundle,
        usetex=args.usetex,
        arrow_models=arrow_models,
    )


if __name__ == "__main__":
    main()
