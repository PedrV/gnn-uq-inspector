"""
Plot a metric from {text}_stats_{models}x{reps}.yaml files.

Usage:
    python plotter.py \
        --run_dir outputs/pems/original/gcn/merlin_2026-03-05 \
        --metric "R2 Test"

    python plotter.py \
        --run_dir outputs/pems/original/gcn/merlin_2026-03-05 \
        --metric "R2 Test" \
        --baselines "Trivial=0.45" "Persistence=0.61" \
        --out r2_plot.png
"""

import os
import re
import math
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams

from matplotlib.patches import Rectangle

DATASET_MAP = {
    "pems": ("regression", 75),
    "cora": ("classification", 1000),
    "citeseer": ("classification", 1000),
    "chameleon": ("regression", 456),
    "gapsmallqm9": ("regression", 818),
    "artnetviews": ("regression", 12791),
    "tolokers2": ("classification", 2974),
}

BASELINES = {
    "pems": {"Diagonal NLL": 102.6190, "RMSE Test": 16.4320},
    "cora": {"NLL": 1.9460, "Accuracy Test": 0.13},
    "citeseer": {"NLL": 1.7920, "Accuracy Test": 0.0770},
    "artnetviews": {"Diagonal NLL": 18223.6030, "R2 Test": 0},
    "tolokers2": {"NLL": 0.5247, "AP Test": 0.2182},
    "chameleon": {"Diagonal NLL": 645.2080, "RMSE Test": 2.1600},
    "gapsmallqm9": {"Diagonal NLL": 1148.1616, "RMSE Test": 1.2681},
}


# ── Font sizes ────────────────────────────────────────────────────────────────
FONT_SIZE_AXIS_LABEL    = 20    # x/y axis labels
FONT_SIZE_TICK_LABEL    = 19.5   # tick labels
FONT_SIZE_LEGEND        = 19     # legend text
FONT_SIZE_DECOMP        = 5.5   # decomposition annotations below x-axis
FONT_SIZE_BASELINE_OOB  = 18     # out-of-bounds baseline pill labels
FONT_SIZE_SUBPLOT_TITLE = 10    # subplot titles (grid mode)

# ── Typography & style ────────────────────────────────────────────────────────
rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Georgia", "Times New Roman", "DejaVu Serif"]
rcParams["mathtext.fontset"] = "dejavuserif"

# ── Palettes ──────────────────────────────────────────────────────────────────
# Supporting an arbitrary number of lines by repeating or extending this list.
PALETTES = [
    {
        "name": "Okabe-Ito Sky Blue",
        "line": "#56B4E9",  # Base Sky Blue
        "band1": "#85C1E9", # Light sky
        "band2": "#AED6F1", # Powder blue
        "dot": "#FFFFFF",
    },
    {
        "name": "Okabe-Ito Orange",
        "line": "#E69F00",  # Base Orange
        "band1": "#F0B27A", # Soft orange
        "band2": "#FAD7A0", # Pale apricot
        "dot": "#FFFFFF",
    },
    {
        "name": "Okabe-Ito Bluish Green",
        "line": "#009E73",  # Base Green
        "band1": "#48C9B0", # Medium mint
        "band2": "#A2D9CE", # Pale teal
        "dot": "#FFFFFF",
    },
    {
        "name": "Okabe-Ito Vermillion",
        "line": "#D55E00",  # Base Vermillion
        "band1": "#EB984E", # Terracotta
        "band2": "#EDBB99", # Soft peach
        "dot": "#FFFFFF",
    },
    {
        "name": "Okabe-Ito Reddish Purple",
        "line": "#CC79A7",  # Base Reddish Purple
        "band1": "#D7A1C4", # Muted orchid
        "band2": "#EBCFE0", # Pale lilac
        "dot": "#FFFFFF",
    },
]

_GRID = "#E8E0DF"  # warm stone grid

_BASELINE_COLORS = [
    "#C0392B",
    "#E67E22",
    "#8E44AD",
    "#16A085",
    "#2C3E50",
]

MAKE_DECOMP = True

DO_CHANGE = False
DO_CHANGE_ABSOLUTE_CONTRIBUTION = True

DO_RATIO = not DO_CHANGE
ONE_MINUS_RATIO = True


def discover_stats_files(run_dir: str) -> dict[int, str]:
    pattern = re.compile(r"^.+_stats_(\d+)x(\d+).*\.yaml$")  # ya?ml$
    hits = {}
    for entry in os.scandir(run_dir):
        m = pattern.match(entry.name)
        if m and entry.is_file():
            hits[int(m.group(1))] = entry.path
    return dict(sorted(hits.items()))


def load_metric(stats_path: str, metric: str) -> dict:
    with open(stats_path) as f:
        data = yaml.safe_load(f)
    if metric not in data:
        raise KeyError(
            f"Metric '{metric}' not found in {stats_path}. "
            f"Available: {list(data.keys())}"
        )
    return data[metric]


def extract_scalar(value) -> float:
    if isinstance(value, list):
        if len(value) == 1:
            return float(value[0])
        raise ValueError(f"Expected scalar or single-element list, got {value}")
    return float(value)


def parse_baselines(raw: list[str]) -> list[tuple[str, float]]:
    """Parse 'Label=value' strings into (label, float) pairs."""
    result = []
    for item in raw:
        if "=" not in item:
            raise ValueError(f"Baseline must be 'Label=value', got: {item!r}")
        label, val = item.split("=", 1)
        result.append((label.strip(), float(val.strip())))
    return result


def plot(
    runs: list[tuple[str, str]],
    metric: str,
    baselines: list[tuple[str, float]],
    out: str | None,
    dataset: str,
    style: str,
    decomp: list[int],
    prune: int,
    *,
    ax: "plt.Axes | None" = None,
    fig: "plt.Figure | None" = None,
    is_subplot: bool = False,
) -> None:

    # ── Canvas ────────────────────────────────────────────────────────────────
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5.5), dpi=180)
        fig.patch.set_facecolor("#F7F8FC")

    ax.set_facecolor("#F7F8FC")

    all_xs = set()
    all_means = []
    all_stds = []

    # ── Plot each run sequentially ────────────────────────────────────────────
    for i, (run_dir, label) in enumerate(runs):
        stats_files = discover_stats_files(run_dir)
        if not stats_files:
            print(f"Warning: No *_stats_*x*.yaml files found in {run_dir}. Skipping.")
            continue

        xs, means, stds, mins, maxs = [], [], [], [], []
        for run_id, (n, path) in enumerate(stats_files.items()):
            if run_id < prune:
                continue
            # print(n, path)
            entry = load_metric(path, metric)
            xs.append(n)
            mean_scalar = extract_scalar(entry["mean"])
            std_scalar = extract_scalar(entry["std"])
            min_scalar = extract_scalar(entry["min"])
            max_scalar = extract_scalar(entry["max"])

            means.append(mean_scalar)
            stds.append(std_scalar)
            mins.append(min_scalar)
            maxs.append(max_scalar)

        xs = np.array(xs)
        means = np.array(means)
        stds = np.array(stds)
        mins = np.array(mins)
        maxs = np.array(maxs)

        all_xs.update(xs)

        # Dataset Specific Normalization
        if dataset in DATASET_MAP:
            ds_type, ds_val = DATASET_MAP[dataset]
            if ds_type == "regression" and metric in ("Diagonal NLL"):
                means /= ds_val
                stds /= ds_val
                mins /= ds_val
                maxs /= ds_val

        # Get color palette (cycle if more runs than palettes)
        print(means)
        all_stds.append(stds)
        all_means.append(means)
        pal = PALETTES[i % len(PALETTES)]

        # Offset x slightly if using bars to prevent overlapping
        x_offset = (i - len(runs) / 2 + 0.5) * 0.15 if style == "bars" else 0
        plot_xs = xs + x_offset

        # ── Variance Visualization ────────────────────────────────────────────
        if style == "band":
            # Min-Max Band
            ax.fill_between(
                plot_xs,
                mins,
                maxs,
                color=pal["band2"],
                alpha=0.35,
                linewidth=0,
                label=f"{label} (min - max)",
            )
            # Std Band
            ax.fill_between(
                plot_xs,
                means - stds,
                means + stds,
                color=pal["band1"],
                alpha=0.30,
                linewidth=0,
                label=f"{label} (±1 std)",
            )
        elif style == "bars":
            # Min-Max Bars (Thinner, lighter line with caps)
            ax.errorbar(
                plot_xs,
                means,
                yerr=[means - mins, maxs - means],
                fmt="none",
                ecolor=pal["band2"],
                elinewidth=1,
                capsize=4,
                zorder=2,
                label=f"{label} (min - max)",
            )
            # Std Bars (Thicker, darker line without caps)
            ax.errorbar(
                plot_xs,
                means,
                yerr=stds,
                fmt="none",
                ecolor=pal["band1"],
                elinewidth=3,
                capsize=0,
                zorder=3,
                label=f"{label} (±1 std)",
            )

        # ── Main line ─────────────────────────────────────────────────────────
        ax.plot(
            plot_xs,
            means,
            color=pal["line"],
            linewidth=2.2,
            zorder=4,
            solid_capstyle="round",
            label=f"{label}",  #  (mean)
        )
        ax.scatter(
            plot_xs,
            means,
            s=52,
            color=pal["line"],
            zorder=5,
            linewidths=1.6,
            edgecolors=pal["dot"],
        )

    # ── Baselines ─────────────────────────────────────────────────────────────────
    y_min, y_max = ax.get_ylim()

    oob_proxy_handles = []
    oob_proxy_labels = []
    for i, (label, value) in enumerate(baselines):
        color = _BASELINE_COLORS[i % len(_BASELINE_COLORS)]
        in_bounds = y_min <= value <= y_max

        if in_bounds:
            ax.axhline(
                value,
                color=color,
                linewidth=1.4,
                linestyle="--",
                alpha=0.85,
                zorder=3,
                label=label,
            )
        else:
            above = value > y_max
            strip_height = (y_max - y_min) * 0.03

            if above:
                edge_y = (
                    y_max - strip_height
                )  # line sits inside, strip fills from line to y_max
                strip_y = edge_y
                arrow_dir = -1
            else:
                edge_y = (
                    y_min + strip_height
                )  # line sits inside, strip fills from y_min to line
                strip_y = y_min
                arrow_dir = 1

            ax.add_patch(
                Rectangle(
                    (0, strip_y),
                    1,
                    strip_height,
                    transform=ax.get_yaxis_transform(),
                    color=color,
                    alpha=0.08,
                    linewidth=0,
                    zorder=2,
                    clip_on=True,
                )
            )

            ax.axhline(
                edge_y,
                color=color,
                linewidth=1.1,
                linestyle=(0, (4, 4)),
                alpha=0.7,
                zorder=3,
            )

            x_frac = 0.18
            y_frac = ax.transAxes.inverted().transform(
                ax.transData.transform((0, edge_y))
            )[1]

            # Arrow points inward (away from the edge, into the strip)
            ax.annotate(
                "",
                xy=(x_frac, y_frac - arrow_dir * 0.035),  # tip points INTO the strip
                xytext=(x_frac, y_frac),  # tail on the dashed line
                xycoords="axes fraction",
                textcoords="axes fraction",
                arrowprops=dict(
                    arrowstyle="-|>", color=color, lw=1.2, mutation_scale=9
                ),
                zorder=5,
                annotation_clip=False,
            )

            pill_text = f"{label}  {value:.3f}"
            ax.annotate(
                pill_text,
                xy=(x_frac + 0.03, y_frac - arrow_dir * 0.022),  # pill inside the strip
                xycoords="axes fraction",
                fontsize=FONT_SIZE_BASELINE_OOB,
                fontfamily="serif",
                color=color,
                va="center",
                ha="left",
                zorder=5,
                annotation_clip=False,
                bbox=dict(
                    boxstyle="round,pad=0.28",
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.12,
                    linewidth=0.8,
                ),
            )

            # Register a proxy artist so it appears in the legend
            from matplotlib.lines import Line2D

            oob_proxy_handles.append(
                Line2D([0], [0], color=color, linewidth=1.4, linestyle="--", alpha=0.85)
            )
            oob_proxy_labels.append(label)

    # ── Axes & grid ───────────────────────────────────────────────────────────
    ax.set_xlabel(
        "Ensemble size (models)",
        fontsize=FONT_SIZE_AXIS_LABEL,
        labelpad=10,
        color="#2C2C3A",
        fontfamily="serif",
    )
    ax.set_ylabel(
        metric,
        fontsize=FONT_SIZE_AXIS_LABEL,
        labelpad=10,
        color="#2C2C3A",
        fontfamily="serif",
    )

    if all_xs:
        unique_xs = sorted(list(all_xs))
        ax.xaxis.set_major_locator(ticker.FixedLocator(unique_xs))
        ax.xaxis.set_major_formatter(ticker.FixedFormatter([str(x) for x in unique_xs]))

    ax.tick_params(axis="both", labelsize=FONT_SIZE_TICK_LABEL, colors="#4A4A5A", length=0)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.tick_params(axis="y", which="minor", length=0)

    ax.yaxis.grid(True, color=_GRID, linewidth=0.8, linestyle="-")
    ax.yaxis.grid(True, which="minor", color=_GRID, linewidth=0.4, linestyle="--")
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(_GRID)
    ax.spines["bottom"].set_linewidth(0.8)

    # ── Legend ────────────────────────────────────────────────────────────────
    handles, labels = ax.get_legend_handles_labels()

    # Keep only entries that are mean lines (one per run) and baselines
    run_labels = {f"{label}" for _, label in runs}  #  (mean)
    base_labels = {label for label, _ in baselines}
    keep = run_labels | base_labels

    by_label = {l: h for l, h in zip(labels, handles) if l in keep}

    leg = ax.legend(
        by_label.values(),
        by_label.keys(),
        frameon=True,
        fontsize=FONT_SIZE_LEGEND,
        loc="best",
        framealpha=0.88,
        edgecolor=_GRID,
        facecolor="#F7F8FC",
    )
    for text in leg.get_texts():
        text.set_color("#2C2C3A")

    fig.tight_layout(pad=1.6)

    # ── Decomposition of change ───────────────────────────────────────────────
    if len(runs) >= 2 and MAKE_DECOMP:
        if decomp is not None:
            decomp_idx = decomp
        else:
            decomp_idx = [0, 1]

        i1, i2 = decomp_idx
        if i1 >= len(all_means) or i2 >= len(all_means):
            print(f"Warning: decomp indices {decomp_idx} out of range, skipping.")
        else:
            m1 = all_means[i1]
            m2 = all_means[i2]
            std1 = all_stds[i1]
            std2 = all_stds[i2]

            total_len = len(unique_xs) - 1 if DO_CHANGE else len(unique_xs)
            for j in range(total_len):
                if DO_CHANGE:
                    total_change = m1[j + 1] - m1[j]
                    subset_change = m2[j + 1] - m2[j]

                    subset_change = 0 if subset_change > 0 else subset_change
                    total_change = 0 if total_change > 0 else total_change
                    exclusive_change = total_change - subset_change

                    if total_change == 0:
                        # text = "—"
                        value = 0 * 100
                    else:
                        if DO_CHANGE_ABSOLUTE_CONTRIBUTION:
                            denom = abs(
                                total_change
                            )  # abs(subset_change) + abs(exclusive_change)
                            value = (
                                (1 - (abs(subset_change) / denom)) * 100
                                if denom != 0 and subset_change != 0
                                else 0
                            )
                        else:
                            value = (subset_change / total_change) * 100
                        text = f"{value:.2f}%"
                    # Place label between the two x points
                    x_mid = (unique_xs[j] + unique_xs[j + 1]) / 2
                elif DO_RATIO:
                    value = (int(ONE_MINUS_RATIO) - m1[j] / m2[j]) * 100
                    text = f"{value:.{int(ONE_MINUS_RATIO)}f}%"
                    x_mid = unique_xs[j]

                ax.text(
                    x_mid,
                    -0.045,
                    text,
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=FONT_SIZE_DECOMP,
                    color="#6A6A7A",
                    fontfamily="serif",
                )

    fig.tight_layout(pad=1.6)
    if not is_subplot:
        if out:
            plt.savefig(out, dpi=600, bbox_inches="tight")
            print(f"Saved to {out}")
        else:
            plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot ensemble size vs metric from *_stats_*x*.yaml files."
    )
    parser.add_argument(
        "--run_dirs",
        nargs="+",
        required=True,
        help="One or more directories containing stats yaml files.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Labels corresponding to each run_dir. Defaults to directory names.",
    )
    parser.add_argument("--metric", required=True, help='Metric name, e.g. "R2 Test"')
    parser.add_argument(
        "--dataset",
        required=True,
        help="The name of the dataset which the data stems from.",
    )
    parser.add_argument(
        "--style",
        choices=["band", "bars"],
        default="band",
        help="Visualization style for the variance (min-max and std). Default is 'band'.",
    )
    parser.add_argument(
        "--baselines",
        nargs="*",
        default=None,
        metavar="LABEL=VALUE",
        help='Horizontal reference lines, e.g. --baselines "Trivial=0.45" "Persistence=0.61". '
        "If omitted, uses the BASELINES dict defined in the script.",
    )
    parser.add_argument(
        "--out", default=None, help="Output file (png/pdf). Omit to show interactively."
    )
    parser.add_argument(
        "--decomp",
        nargs=2,
        type=int,
        default=None,
        metavar=("IDX1", "IDX2"),
        help="Indices (0-based) of the two runs to use for decomposition of change. "
        "Defaults to 0 and 1 when there are exactly 2 runs.",
    )
    parser.add_argument(
        "--prune",
        type=int,
        default=0,
        help="Point to start from the plot from.",
    )
    parser.add_argument(
        "--grid",
        nargs="+",
        metavar="RUN_DIRS_PER_PLOT",
        default=None,
        help=(
            "Grid layout mode. Each argument is a comma-separated list of run_dirs "
            "forming one subplot. E.g. --grid dir1,dir2 dir3 dir4,dir5,dir6 "
            "produces a 3-subplot grid. Labels, metrics, etc. are shared across all."
        ),
    )
    parser.add_argument(
        "--grid_titles",
        nargs="*",
        default=None,
        help="Optional title for each subplot in grid mode.",
    )
    args = parser.parse_args()

    # Match labels to run directories
    if args.labels is None:
        args.labels = [os.path.basename(os.path.normpath(d)) for d in args.run_dirs]
    elif len(args.labels) != len(args.run_dirs):
        parser.error("The number of --labels must match the number of --run_dirs.")

    runs = list(zip(args.run_dirs, args.labels))

    stock_baselines = False
    if args.baselines is None and stock_baselines:
        if args.dataset in BASELINES and args.metric in BASELINES[args.dataset]:
            value = BASELINES[args.dataset][args.metric]
            baselines = [(args.metric, value)]
        else:
            print("Skipping predefined baselines!")
            baselines = []
    elif args.baselines is not None:
        baselines = parse_baselines(args.baselines)
    else:
        baselines = []

    if args.grid is not None:
        # Parse each grid cell: comma-separated run_dirs -> list of (dir, label) pairs
        cells: list[list[tuple[str, str]]] = []
        for cell_spec in args.grid:
            dirs = [d.strip() for d in cell_spec.split(",")]
            cell_labels = [os.path.basename(os.path.normpath(d)) for d in dirs]
            cells.append(list(zip(dirs, cell_labels)))

        n_plots = len(cells)
        n_cols  = min(n_plots, 3)
        n_rows  = math.ceil(n_plots / n_cols)

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(10 * n_cols, 5.5 * n_rows),
            dpi=180,
            squeeze=False,
        )
        fig.patch.set_facecolor("#F7F8FC")

        for idx, (cell_runs, ax_) in enumerate(
            zip(cells, axes.flat)
        ):
            title = (
                args.grid_titles[idx]
                if args.grid_titles and idx < len(args.grid_titles)
                else f"Plot {idx + 1}"
            )
            ax_.set_title(
                title,
                fontsize=FONT_SIZE_SUBPLOT_TITLE,
                color="#2C2C3A",
                fontfamily="serif",
                pad=6,
            )

            plot(
                cell_runs,
                args.metric,
                baselines,
                out=None,           # don't save per-subplot
                dataset=args.dataset,
                style=args.style,
                decomp=args.decomp,
                prune=args.prune,
                ax=ax_,
                fig=fig,
                is_subplot=True,
            )

        # Hide any unused axes (when n_plots < n_rows * n_cols)
        for ax_ in axes.flat[n_plots:]:
            ax_.set_visible(False)

        fig.tight_layout(pad=2.0)

        if args.out:
            fig.savefig(args.out, dpi=600, bbox_inches="tight")
            print(f"Grid saved to {args.out}")
        else:
            plt.show()

    else:
        # original single-plot path
        plot(
            runs,
            args.metric,
            baselines,
            args.out,
            args.dataset,
            args.style,
            args.decomp,
            args.prune,
        )


if __name__ == "__main__":
    main()