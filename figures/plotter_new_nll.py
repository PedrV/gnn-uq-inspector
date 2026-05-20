""""
Available bundles (a subset — run ``python -c "from tueplots import bundles;
print(dir(bundles))"`` for the full list):
    neurips2024   NeurIPS 2024  (5.5 in wide, default)
    icml2024      ICML 2024     (3.25 in wide, half-column)
    iclr2024      ICLR 2024
    aistats2023   AISTATS 2023
    tmlr2023      TMLR 2023
    jmlr2001      JMLR

Example
-------
python plot_ensemble_nll.py \\
    --dataset_dirs  runs/boston runs/concrete runs/energy runs/kin8nm \\
    --dataset_labels Boston Concrete Energy Kin8nm \\
    --metric "Diagonal NLL" \\
    --ensemble_position 2 4 6 \\
    --ensemble_base 0 \\
    --ensemble_base_dir runs/base_model \\
    --baselines "Trivial=85.0" "MLE=12.5" "RANGE!-1Layer=1.81+-0.81" \\
    --dataset_sizes boston=455 concrete=824 energy=691 kin8nm=6996 \\
    --bundle neurips2024 \\
    --out figure1.pdf
"""

from __future__ import annotations

import argparse
import os
import re

import matplotlib as mplt
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import yaml

# ── tueplots ─────────────────────────────────────────────────────────────────
try:
    from tueplots import bundles as _tp_bundles, figsizes as _tp_figsizes
    _HAS_TUEPLOTS = False
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
]

_BASELINE_COLORS = ["#555550", "#8C3E6E", "#004D80", "#943F00"]

_BG   = "#FFFFFF"
_GRID = "#E2E2E2"

# ── Dataset normalisation map ─────────────────────────────────────────────────
DATASET_MAP: dict[str, int] = {}


# ── rcParams helpers ──────────────────────────────────────────────────────────

def _base_rcparams() -> dict:
    """Minimal rcParams that apply regardless of bundle choice."""
    return {
        "pdf.fonttype": 42,   # embed fonts as Type 1/TrueType in PDF
        "ps.fonttype":  42,
        "axes.linewidth":    0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size":  2.5,
        "ytick.major.size":  2.5,
        "xtick.direction":   "out",
        "ytick.direction":   "out",
        "lines.linewidth":   0.9,
    }


def _get_bundle_rc(bundle_name: str, usetex: bool) -> dict:
    """Return the tueplots rcParam dict for *bundle_name*.

    If tueplots is unavailable, fall back to a conservative set of values
    that mimics typical NeurIPS sizing (9 pt fonts, 5.5 in wide).

    Important: tueplots bundles set ``text.usetex = True`` by default.
    This requires a working LaTeX installation with the ``type1ec`` package.
    If you do not have that, saving the figure will crash.  We therefore
    default to ``usetex=False`` and strip the latex preamble; pass
    ``--usetex`` only when your LaTeX install is confirmed to work.
    """
    if not _HAS_TUEPLOTS:
        print(
            "Warning: tueplots not installed. "
            "Using fallback rcParams (pip install tueplots for venue-accurate sizes)."
        )
        return {
            "font.family":     "serif",
            "font.size":       9,
            "axes.labelsize":  9,
            "axes.titlesize":  9,
            "legend.fontsize": 10,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "figure.figsize":  (5.5, 3.4),
            "figure.constrained_layout.use": True,
            "figure.autolayout":             False,
            "savefig.pad_inches":            0.015,
        }

    fn = getattr(_tp_bundles, bundle_name, None)
    if fn is None:
        available = [x for x in dir(_tp_bundles) if not x.startswith("_")]
        raise ValueError(
            f"Unknown bundle {bundle_name!r}. Available: {available}"
        )
    rc = fn()
    # Always override text.usetex — the bundle defaults to True, which
    # requires type1ec.sty and will crash on save if that is missing.
    rc["text.usetex"] = usetex
    if not usetex:
        rc.pop("text.latex.preamble", None)
        rc.setdefault("font.family", "serif")
    return rc


def _figsize_for_bundle(
    bundle_name: str,
    n_datasets: int,
    *,
    need_break: bool,
) -> tuple[float, float]:
    """Return (width, height) for the final figure.

    Height is computed so each dataset row gets ~0.55 in plus margins,
    instead of using the bundle's default golden-ratio height.
    Width comes from the bundle (respects journal column width).
    Bundle width is doubled when we need a broken axis.
    """
    if _HAS_TUEPLOTS:
        fn = getattr(_tp_bundles, bundle_name, None)
        if fn is not None:
            col_w = fn()["figure.figsize"][0]
        else:
            col_w = 5.5
    else:
        col_w = 5.5

    total_w = col_w * (1.33 if need_break else 1.0)

    # Height: fixed margins + per-row height.
    # No top margin needed for labels — they are now drawn inline.
    top_margin   = 0.2   # small breathing room above top row
    bot_margin   = 0.55  # x-label + annotation
    per_row      = 0.60  # inches per dataset — more room so series don't crowd
    h = top_margin + n_datasets * per_row + bot_margin
    h = max(h, 2.0)      # never smaller than 2 in

    return total_w, h


# ── NLL difference + CI ───────────────────────────────────────────────────────

def difference_nll(
    base_mean_nll: float,
    base_std_nll: float,
    other_mean_nll: float,
    other_std_nll: float,
    base_n: int,
    other_n: int,
    ci_q: float = 0.95,
) -> tuple[float, float]:
    """Return (delta_nll, ci_half_width).

    delta_nll > 0 means the ensemble improved over the base model.
    ci_half_width is the half-width of the two-sided confidence interval.
    """
    delta_nll = base_mean_nll - other_mean_nll

    se_base  = base_std_nll  / np.sqrt(base_n)
    se_other = other_std_nll / np.sqrt(other_n)
    se_delta = np.sqrt(se_base ** 2 + se_other ** 2)

    if ci_q == 0.95:
        ci = 1.96 * se_delta
    else:
        raise ValueError(f"Don't know how to calculate CI for confidence level {ci_q}")

    return delta_nll, ci


# ── Data helpers ──────────────────────────────────────────────────────────────

def discover_stats_files(dataset_dir: str) -> dict[int, tuple[str, int]]:
    """Return mapping: ensemble_size → (file_path, n_reps)."""
    pattern = re.compile(r"^.+_stats_(\d+)x(\d+).*\.yaml$")
    hits: dict[int, tuple[str, int]] = {}
    for entry in os.scandir(dataset_dir):
        m = pattern.match(entry.name)
        if m and entry.is_file():
            hits[int(m.group(1))] = (entry.path, int(m.group(2)))
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
        raise ValueError(f"Expected scalar or single-element list, got {value!r}")
    return float(value)


def parse_baselines(raw: list[str]) -> list[tuple[str, float, float | None]]:
    """Parse baseline specs into (label, mean, std_or_None)."""
    result = []
    for item in raw:
        if "=" not in item:
            raise ValueError(f"Baseline must be 'Label=value', got: {item!r}")

        label_part, val_str = item.split("=", 1)
        label_part = label_part.strip()
        val_str    = val_str.strip()

        if label_part.startswith("RANGE!-"):
            label = label_part[len("RANGE!-"):]
            if "+-" not in val_str:
                raise ValueError(
                    f"RANGE! baseline must have 'mean+-std' value, got: {val_str!r}"
                )
            mean_s, std_s = val_str.split("+-", 1)
            result.append((label.strip(), float(mean_s.strip()), float(std_s.strip())))
        else:
            result.append((label_part, float(val_str), None))

    return result


# ── Loading ───────────────────────────────────────────────────────────────────

def load_base_entry(
    base_dir: str,
    base_position: int,
    metric: str,
) -> tuple[float, float, int]:
    stats_files = discover_stats_files(base_dir)
    if not stats_files:
        raise FileNotFoundError(f"No stats yaml files found in {base_dir!r}.")
    sorted_sizes = list(stats_files.keys())
    if base_position >= len(sorted_sizes):
        raise IndexError(
            f"ensemble_base position {base_position} out of range: only "
            f"{len(sorted_sizes)} sizes in {base_dir!r} (sizes: {sorted_sizes})."
        )
    base_size = sorted_sizes[base_position]
    path, n_reps = stats_files[base_size]
    entry = load_metric(path, metric)
    return (
        extract_scalar(entry["mean"]),
        extract_scalar(entry["std"]),
        n_reps,
    )


def load_dataset(
    dataset_dir: str,
    positions: list[int],
    base_position: int,
    metric: str,
    base_dir: str | None,
) -> tuple[
    dict[int, tuple[float, float, int]],
    list[int],
    tuple[float, float, int],
]:
    stats_files = discover_stats_files(dataset_dir)
    if not stats_files:
        raise FileNotFoundError(f"No stats yaml files found in {dataset_dir!r}.")

    sorted_sizes = list(stats_files.keys())

    def _get(pos: int) -> int:
        if pos >= len(sorted_sizes):
            raise IndexError(
                f"Position {pos} out of range: only {len(sorted_sizes)} sizes "
                f"in {dataset_dir!r} (sizes: {sorted_sizes})."
            )
        return sorted_sizes[pos]

    if base_dir is None:
        base_size_local = _get(base_position)
        compare_sizes   = [_get(p) for p in positions if _get(p) != base_size_local]
    else:
        compare_sizes = [_get(p) for p in positions]

    raw: dict[int, tuple[float, float, int]] = {}
    for size in compare_sizes:
        path, n_reps = stats_files[size]
        entry = load_metric(path, metric)
        raw[size] = (
            extract_scalar(entry["mean"]),
            extract_scalar(entry["std"]),
            n_reps,
        )

    if base_dir is not None:
        base_tuple = load_base_entry(base_dir, base_position, metric)
    else:
        base_size_local = _get(base_position)
        path, n_reps = stats_files[base_size_local]
        entry = load_metric(path, metric)
        base_tuple = (
            extract_scalar(entry["mean"]),
            extract_scalar(entry["std"]),
            n_reps,
        )

    return raw, compare_sizes, base_tuple


# ── Break marks ───────────────────────────────────────────────────────────────

def _draw_break_marks(ax_left, ax_right, color: str = "#555555") -> None:
    """Draw diagonal cut marks between two horizontally adjacent axes."""
    fig = ax_left.get_figure()

    # Finalise the layout so bounding-box queries are accurate.
    fig.canvas.draw()
    bbox_l = ax_left.get_position()
    bbox_r = ax_right.get_position()

    gap_x0 = bbox_l.x1 + 0.02
    gap_x1 = bbox_r.x0 + 0.04
    gap_y0 = bbox_l.y0
    gap_y1 = bbox_l.y1
    gap_w  = gap_x1 - gap_x0
    gap_h  = gap_y1 - gap_y0

    # Shaded fill covering the inter-panel gap.
    if gap_w > 0 and gap_h > 0:
        shade = fig.add_axes(
            [gap_x0, gap_y0, gap_w, gap_h],
            label="_break_shade",
        )
        shade.set_navigate(False)
        shade.patch.set_facecolor("#C8C8C8")
        shade.patch.set_alpha(0.15)
        shade.patch.set_hatch('////')
        shade.patch.set_edgecolor('#A0A0A0')
        shade.set_xticks([])
        shade.set_yticks([])
        for spine in shade.spines.values():
            spine.set_visible(False)
    
    d     = 0.018
    angle = 0.5
    kw    = dict(transform=ax_left.transAxes, color=color,
                 linewidth=0.9, clip_on=False, zorder=10)

    for y0 in (0.0, 1.0):
        ax_left.plot(
            (1 - d, 1 + d),
            (y0 - angle * d, y0 + angle * d),
            **kw
        )
        kw2 = dict(kw, transform=ax_right.transAxes)
        ax_right.plot(
            (-d, +d),
            (y0 - angle * d, y0 + angle * d),
            **kw2
        )


# ── Axis styling ──────────────────────────────────────────────────────────────

def _style_ax(ax, grid: bool = True) -> None:
    ax.set_facecolor(_BG)
    ax.tick_params(axis="both", colors="#333333", length=0)
    if grid:
        ax.minorticks_on()
        ax.yaxis.set_tick_params(which="minor", left=False)
        ax.xaxis.grid(True, color=_GRID, linewidth=0.7, linestyle="-")
        ax.xaxis.grid(True, which="minor", color=_GRID, linewidth=0.3,
                      linestyle="-", alpha=0.5)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(_GRID)
    ax.spines["bottom"].set_linewidth(0.8)


# ── Inline line labels ────────────────────────────────────────────────────────

def _label_baseline_inline(
    ax,
    x_data: float,
    label: str,
    color: str,
    ylim: tuple[float, float],
) -> None:
    """Draw *label* centred vertically on the baseline at *x_data*.

    The text sits just to the right of the line, inside the axes, at the
    mid-point of the y range.  A small white background box keeps it legible
    when it overlaps data.
    """
    rng = np.random.default_rng(1111)
    x_offset = -0.012
    if label == "SVHN, VGG":
        x_offset -= 0.012

    if label == "NYUdepthV2, UNet":
        y_mid = (ylim[0] + ylim[1]) / 2
        x_offset += 0.006
    elif label == "UCI, MLP":
        y_mid = (ylim[0] + ylim[1]) / 1.5
    else:
        y_mid = (ylim[0] + ylim[1]) / rng.uniform(1, 3)

    fontsize = mplt.rcParams.get("legend.fontsize", 15)
    ax.text(
        x_data+x_offset, y_mid, f"  {label}",
        color=color, fontsize=fontsize,
        va="center", ha="left",
        rotation=90,
        bbox=dict(
            boxstyle="square,pad=0.15",
            facecolor="white", edgecolor="none", alpha=1,
        ),
        clip_on=True, zorder=5,
    )


def _label_n1_inline(
    ax,
    x_data: float,
    ylim: tuple[float, float],
    color: str = "#999999",
) -> None:
    """Draw the 'N = 1' label inside the axes near the top of the N=1 line."""
    fontsize = mplt.rcParams.get("legend.fontsize", 15)
    x_offset = -0.01  # for regression  # -0.001 # for classification
    # Place near the top: 85 % of the way up the y range.
    y_top = ylim[0] + 0.85 * (ylim[1] - ylim[0])
    ax.text(
        x_data+x_offset, y_top, "  N = 1",
        color=color, fontsize=fontsize,
        va="center", ha="left",
        rotation=90,
        bbox=dict(
            boxstyle="square,pad=0.15",
            facecolor="white", edgecolor="none", alpha=0.75,
        ),
        clip_on=True, zorder=5,
    )


# ── Core drawing ──────────────────────────────────────────────────────────────

def _draw_series(
    ax,
    ys: np.ndarray,
    offsets: list[float],
    compare_sizes: list[int],
    diff_data: list[dict],
    n_datasets: int,
) -> None:
    """Draw horizontal scatter + CI error bars for all ensemble sizes onto *ax*."""
    for size_idx, size in enumerate(compare_sizes):
        pal = _PALETTES[size_idx % len(_PALETTES)]
        off = offsets[size_idx]

        deltas = np.full(n_datasets, np.nan)
        cis    = []

        for ds_idx in range(n_datasets):
            if size in diff_data[ds_idx]:
                delta, cil, ciu = diff_data[ds_idx][size]
                deltas[ds_idx] = delta
                cis.append((cil, ciu))

        py = ys + off

        xerr_asym = [
            [d - low  for d, (low, high) in zip(deltas, cis)],
            [high - d for d, (low, high) in zip(deltas, cis)],
        ]
        # Thick error bars with visible caps — key for legibility at column width.
        ax.errorbar(
            deltas, py,
            xerr=xerr_asym,
            fmt="none", ecolor=pal["color"], elinewidth=2.0,
            capsize=4.0, capthick=2.0, zorder=3,
        )
        # Large marker with contrasting edge so it reads at small sizes.
        ax.scatter(
            deltas, py,
            s=55, marker=pal["marker"],
            facecolors=pal["color"], edgecolors=pal["edge"],
            linewidths=1.2, zorder=4,
            label=f"N = {size}",
        )


def _draw_baseline(
    ax,
    mean: float,
    std: float | None,
    color: str,
) -> None:
    ax.axvline(mean, color=color, linewidth=1.5, linestyle="--", alpha=0.9, zorder=2)
    if std is not None:
        ax.axvspan(
            mean - std, mean + std,
            color=color, alpha=0.15, linewidth=0, zorder=1,
        )


# ── Main plot ─────────────────────────────────────────────────────────────────

def plot(
    *,
    dataset_dirs: list[str],
    dataset_labels: list[str],
    metrics: list[str],
    positions: list[int],
    base_position: int,
    base_dirs: list[str | None],
    baselines: list[tuple[str, float, float | None]],
    out: str | None,
    extra_bottom_annot_text: str = "",
    ci_q: float = 0.95,
    bundle: str = "neurips2024",
    usetex: bool = False,
) -> None:

    # ── Apply rcParams (tueplots bundle + base overrides) ─────────────────────
    # Order matters: base first, then bundle (bundle wins on font sizes/family),
    # then pdf.fonttype etc. that must always be present.
    rc = {}
    rc.update(_get_bundle_rc(bundle, usetex))
    rc.update(_base_rcparams())    # thin lines, pdf embedding — never overridden
    mplt.rcParams.update(rc)

    n_datasets = len(dataset_dirs)

    # ── Load, compute deltas, normalise ──────────────────────────────────────
    diff_data: list[dict[int, tuple[float, float, float]]] = []
    compare_sizes: list[int] | None = None

    for ds_dir, metric, base_dir in zip(dataset_dirs, metrics, base_dirs):
        raw, csizes, base_tuple = load_dataset(
            ds_dir, positions, base_position, metric, base_dir
        )
        if compare_sizes is None:
            compare_sizes = csizes

        _norm = os.path.normpath(ds_dir)
        n_examples = None
        for _part in reversed(_norm.split(os.sep)):
            if _part in DATASET_MAP:
                n_examples = DATASET_MAP[_part]
                break
        if n_examples is None:
            _candidates = _norm.split(os.sep)
            raise KeyError(
                f"None of the path components {_candidates} were found in "
                f"DATASET_MAP. Available keys: {list(DATASET_MAP.keys())}"
            )

        base_mean, base_std, base_n = base_tuple
        ds_diff = {}
        for size in csizes:
            other_mean, other_std, other_n = raw[size]
            delta, ci = difference_nll(
                base_mean, base_std,
                other_mean, other_std,
                base_n, other_n,
                ci_q=ci_q,
            )
            ds_diff[size] = (
                np.exp(delta / n_examples),
                np.exp(delta / n_examples - ci / n_examples),
                np.exp(delta / n_examples + ci / n_examples),
            )
        diff_data.append(ds_diff)

    n_sizes = len(compare_sizes)

    # ── Cluster offsets ───────────────────────────────────────────────────────
    cluster_w = min(0.50, 0.20 * n_sizes)
    offsets = (
        [0.0] if n_sizes == 1
        else np.linspace(-cluster_w / 2, cluster_w / 2, n_sizes).tolist()
    )

    ys = np.arange(n_datasets, dtype=float)

    # ── Determine data x-range ────────────────────────────────────────────────
    all_vals = []
    for ds_diff in diff_data:
        for delta, cil, ciu in ds_diff.values():
            all_vals.extend([delta, cil, ciu])
    # all_vals.append(1.0)  # The baseline

    data_lo = min(all_vals)
    data_hi = max(all_vals)
    pad     = max(0.15 * (data_hi - data_lo), 1e-6)
    main_lo = data_lo - pad
    main_hi = data_hi + pad

    # ── Partition baselines: in-range vs out-of-range ─────────────────────────
    span      = main_hi - main_lo
    threshold = span * 0.15

    def _bl_lo(bl):
        _, mean, std = bl
        return mean - std if std is not None else mean

    def _bl_hi(bl):
        _, mean, std = bl
        return mean + std if std is not None else mean

    indexed_baselines = list(enumerate(baselines))
    in_range_bl  = [(i, bl) for i, bl in indexed_baselines
                    if main_lo - threshold <= _bl_lo(bl) and _bl_hi(bl) <= main_hi + threshold]
    out_range_bl = [(i, bl) for i, bl in indexed_baselines
                    if (i, bl) not in in_range_bl]

    need_break = bool(out_range_bl)

    # ── Figure size from bundle ───────────────────────────────────────────────
    fig_w, fig_h = _figsize_for_bundle(bundle, n_datasets, need_break=need_break)

    if need_break:
        # Width ratio: main panel gets ~75% of the width.
        main_frac = 0.75
        oob_frac  = 1.0 - main_frac
        wr = [main_frac, oob_frac]
        fig, (ax, ax_bl) = plt.subplots(
            1, 2,
            figsize=(fig_w, fig_h),
            dpi=200,
            gridspec_kw={"width_ratios": wr, "wspace": 0.15},
            sharey=True,
        )
    else:
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
        ax_bl = None

    fig.patch.set_facecolor(_BG)

    # ── Style axes ────────────────────────────────────────────────────────────
    _style_ax(ax)
    if ax_bl is not None:
        _style_ax(ax_bl)
        ax_bl.spines["left"].set_visible(False)
        ax_bl.tick_params(left=False, labelleft=False)
        ax.spines["right"].set_visible(False)

    # ── Draw data series (main panel only) ────────────────────────────────────
    _draw_series(ax, ys, offsets, compare_sizes, diff_data, n_datasets)
    # NOTE: do NOT call _draw_series on ax_bl — it is for out-of-range baselines
    # only; drawing data there causes markers to appear twice when the overflow
    # panel x-range overlaps with the main data range.

    # ── N=1 reference line + inline label (top of line, inside axes) ─────────
    # ax.axvline(1, color="#AAAAAA", linewidth=1.5, linestyle="-", zorder=1)

    ylim = (-0.6, n_datasets - 0.4)

    # ── In-range baselines + inline centre labels ─────────────────────────────
    for orig_i, bl in in_range_bl:
        label, mean, std = bl
        color = _BASELINE_COLORS[orig_i % len(_BASELINE_COLORS)]
        _draw_baseline(ax, mean, std, color)
        _label_baseline_inline(ax, mean, label, color, ylim)

    # N=1 label inside the axes, near the top of its line.
    # _label_n1_inline(ax, 1.0, ylim)

    # ── Out-of-range baselines (right overflow panel) ─────────────────────────
    if need_break:
        oob_values = []
        for _, bl in out_range_bl:
            _, mean, std = bl
            oob_values.append(mean)
            if std is not None:
                oob_values.extend([mean - std, mean + std])
        oob_lo  = min(oob_values)
        oob_hi  = max(oob_values)
        oob_pad = max(0.12 * (oob_hi - oob_lo + 1e-9), abs(oob_lo) * 0.08, abs(oob_hi) * 0.08)
        ax_bl.set_xlim(oob_lo - oob_pad, oob_hi + oob_pad)

        oob_label_items: list[tuple[float, str, str]] = []
        for orig_i, bl in out_range_bl:
            label, mean, std = bl
            color = _BASELINE_COLORS[orig_i % len(_BASELINE_COLORS)]
            _draw_baseline(ax_bl, mean, std, color)
            _label_baseline_inline(ax_bl, mean, label, color, ylim)
            oob_label_items.append((mean, label, color))
        _draw_break_marks(ax, ax_bl)

        ax_bl.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3, prune="both"))
        ax_bl.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        leg_bl = ax_bl.get_legend()
        if leg_bl is not None:
            leg_bl.remove()

    # ── Main panel x-axis ─────────────────────────────────────────────────────
    ax.set_xlim(main_lo, main_hi)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6, prune="both"))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    # ── y-axis (datasets) ─────────────────────────────────────────────────────
    ax.set_yticks(ys)
    # y-tick font size comes from rcParams; no manual override needed.
    ax.set_yticklabels(dataset_labels)
    ax.set_ylim(*ylim)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color(_GRID)
    ax.spines["left"].set_linewidth(0.8)

    # ── Single shared x-label ─────────────────────────────────────────────────
    unique_metrics = list(dict.fromkeys(metrics))
    metric_label   = unique_metrics[0] if len(unique_metrics) == 1 else "NLL"
    xlabel = "" # rf"$\Delta$ {metric_label} per datapoint (N=1 minus N, nats)"

    if need_break:
        fig.text(0.5, 0.0, xlabel, ha="center", va="bottom",
                 fontsize=mplt.rcParams["axes.labelsize"])
    else:
        ax.set_xlabel(xlabel, labelpad=5)

    # ── Legend — horizontal strip at the top of the axes ─────────────────────
    # Placed inside at the top-right to match the reference style (image 2).
    handles, labels_leg = ax.get_legend_handles_labels()
    seen, h_dedup, l_dedup = set(), [], []
    for h, l in zip(handles, labels_leg):
        if l not in seen:
            seen.add(l)
            h_dedup.append(h)
            l_dedup.append(l)

    ax.legend(
        h_dedup, l_dedup,
        frameon=True,
        loc="lower right",
        # bbox_to_anchor=(0.97, 1.0),
        ncols=len(h_dedup),          # all entries on one horizontal row
        framealpha=0.92, edgecolor=_GRID, facecolor=_BG,
        handletextpad=0.35, labelspacing=0.25, columnspacing=0.8,
        borderpad=0.45,
        handlelength=1.0,
    )

    # ── Annotation (small, below x-axis) ─────────────────────────────────────
    ci_pct = int(ci_q * 100)
    # base_annot = (
    #     f"Error bars: {ci_pct}% CI (Welch-style SE propagation); "
    #     "NLL normalised by dataset size"
    # )
    # if extra_bottom_annot_text:
    #     base_annot += f"; {extra_bottom_annot_text}."
    # else:
    #     base_annot += "."

    # # Annotation is intentionally smaller than body text — use legend fontsize.
    # annot_size = mplt.rcParams.get("legend.fontsize", 6)
    # ax.annotate(
    #     base_annot,
    #     xy=(0, 0), xycoords="axes fraction",
    #     xytext=(0, -0.18), textcoords="axes fraction",
    #     fontsize=annot_size, color="#888888",
    #     ha="left", va="top", annotation_clip=False,
    # )

    if out:
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved to {out}")
    else:
        plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot NLL difference (N=1 minus N) with 95% confidence intervals "
            "across datasets. Baselines that fall far outside the data range "
            "are shown in a separate broken-axis panel."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset_dirs", nargs="+", required=True, metavar="DIR",
    )
    parser.add_argument(
        "--dataset_labels", nargs="+", required=True, metavar="LABEL",
    )
    parser.add_argument(
        "--metric", nargs="+", required=True, metavar="METRIC",
    )
    parser.add_argument(
        "--ensemble_position", nargs="+", type=int, required=True, metavar="POS",
    )
    parser.add_argument(
        "--ensemble_base", type=int, default=0, metavar="POS",
    )
    parser.add_argument(
        "--ensemble_base_dir", nargs="+", default=None, metavar="DIR",
    )
    parser.add_argument(
        "--baselines", nargs="*", default=None, metavar="LABEL=VALUE",
    )
    parser.add_argument(
        "--dataset_sizes", nargs="+", required=True, metavar="NAME=N",
        help=(
            "Number of examples per dataset for NLL normalisation. "
            'Pass "name=count" per dataset.'
        ),
    )
    parser.add_argument(
        "--bundle", default="neurips2024", metavar="BUNDLE",
        help=(
            "tueplots bundle name for venue-accurate figure sizing and font "
            "sizes. Examples: neurips2024, icml2024, iclr2024, aistats2023, "
            "tmlr2023, jmlr2001. Default: neurips2024."
        ),
    )
    parser.add_argument(
        "--usetex", action="store_true", default=False,
        help=(
            "Render text with LaTeX (requires a working LaTeX installation). "
            "Produces exact font matching for the chosen venue bundle."
        ),
    )
    parser.add_argument(
        "--out", default=None,
        help="Output file (png / pdf / svg). Omit to display interactively.",
    )
    parser.add_argument(
        "--bottom_annot", default="",
        help="Extra text appended to the annotation below the x-axis.",
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
            f"--metric must receive either 1 value or {n_ds} values, "
            f"got {len(args.metric)}."
        )

    for entry in args.dataset_sizes:
        if "=" not in entry:
            parser.error(f"--dataset_sizes entries must be 'name=count', got: {entry!r}")
        name, _, count_str = entry.partition("=")
        try:
            DATASET_MAP[name.strip()] = int(count_str.strip())
        except ValueError:
            parser.error(f"Count in --dataset_sizes must be an integer, got: {entry!r}")

    baselines = parse_baselines(args.baselines) if args.baselines else []

    if args.ensemble_base_dir is None:
        base_dirs = [None] * n_ds
    elif len(args.ensemble_base_dir) == 1:
        raw_bd = args.ensemble_base_dir[0]
        base_dirs = [None if raw_bd == "." else raw_bd] * n_ds
    elif len(args.ensemble_base_dir) == n_ds:
        base_dirs = [None if d == "." else d for d in args.ensemble_base_dir]
    else:
        parser.error(
            f"--ensemble_base_dir must receive 1 or {n_ds} values, "
            f"got {len(args.ensemble_base_dir)}."
        )

    plot(
        dataset_dirs=args.dataset_dirs,
        dataset_labels=args.dataset_labels,
        metrics=metrics,
        positions=args.ensemble_position,
        base_position=args.ensemble_base,
        base_dirs=base_dirs,
        baselines=baselines,
        out=args.out,
        extra_bottom_annot_text=args.bottom_annot,
        bundle=args.bundle,
        usetex=args.usetex,
    )


if __name__ == "__main__":
    main()
