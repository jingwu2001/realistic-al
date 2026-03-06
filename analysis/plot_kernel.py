"""plot_kernel.py
~~~~~~~~~~~~~~~~
Plotting script with explicit control over which experiment folder types to include.

Two folder conventions exist:
  WITH kernel :  ..._acq-vendi_norm-minmax_kernel-cosine_gamma-None_ep-200
  WITHOUT kernel:  ..._acq-vendi_ep-200

Use --kernel to include only those WITH a kernel spec (legend: "Vendi-rbf-1.0").
Use --no-kernel to include only those WITHOUT a kernel spec (legend: "Vendi").
Both flags may be passed together to include both types.

Other options mirror plot_simple.py:
  --prefix, --title, --no-std, --results-path, --save-path,
  --skip-before, --skip-after, --query-methods
"""

import sys
import argparse
import datetime
import colorsys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Fixed palette for non-Vendi / non-Bandit methods.
# Vendi and Bandit receive dynamic themed palettes (see build_palette()).
# ---------------------------------------------------------------------------
PALETTE = {
    "BALD":     "#4C72B0",  # slate blue
    "Core-Set": "#55A868",  # medium green
    "Entropy":  "#8172B2",  # muted purple
    "BADGE":    "#64B5CD",  # steel blue
    "Random":   "#6B6B6B",  # neutral grey
}

# All recognised acquisition methods (key = folder token after 'acq-').
ALL_QUERYMETHODS = {
    "bald":          "BALD",
    "kcentergreedy": "Core-Set",
    "entropy":       "Entropy",
    "random":        "Random",
    "badge":         "BADGE",
    "vendi":         "Vendi",
    "bandit":        "Bandit",
}

SKIP_DIR = ["active-cifar10_high"]

sns.set_style("whitegrid")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_acq(folder: str) -> str | None:
    """Return the acquisition key from a folder name, or None."""
    for part in folder.split("_"):
        if part.startswith("acq-"):
            return part[4:]
    return None


def get_kernel_params(folder: str) -> dict:
    """Return {'kernel': str|None, 'gamma': float|None} parsed from a folder name."""
    kernel = None
    gamma = None
    for part in folder.split("_"):
        if part.startswith("kernel-"):
            kernel = part[7:]
        if part.startswith("gamma-"):
            raw = part[6:]
            gamma = None if raw == "None" else float(raw)
    return {"kernel": kernel, "gamma": gamma}


def has_kernel(folder: str) -> bool:
    return get_kernel_params(folder)["kernel"] is not None


def method_label(folder: str, acq_key: str, use_kernel_in_label: bool) -> str:
    """Build the legend label for an experiment folder.

    If use_kernel_in_label is True the kernel/gamma are appended, e.g.
    'Vendi-rbf-1.0'.  Otherwise just the base name is used, e.g. 'Vendi'.
    """
    base = ALL_QUERYMETHODS[acq_key]
    if use_kernel_in_label:
        kp = get_kernel_params(folder)
        if kp["kernel"]:
            base += f"-{kp['kernel']}"
        if kp["gamma"] is not None:
            base += f"-{kp['gamma']}"
    return base


def get_aubc(performance: np.ndarray) -> float:
    """Area Under the Budget Curve (trapezoid, normalised to [0,1])."""
    if len(performance) <= 1:
        return 0.0
    dx = 1.0 / (len(performance) - 1)
    return float(np.trapz(performance, dx=dx))


def hue_sweep_palette(n: int, hue_min: float, hue_max: float,
                      lightness: float = 0.40, saturation: float = 0.80):
    """Return *n* colors sweeping hue linearly in HLS space."""
    if n <= 0:
        return []
    hues = np.linspace(hue_min, hue_max, n) % 1.0
    return [colorsys.hls_to_rgb(h, lightness, saturation) for h in hues]


def build_palette(unique_methods):
    """Build a {method_name: color} dict with themed palettes.

    Vendi-rbf-*    → cool teal/cyan theme   (hue 0.47–0.60)
    Vendi-cosine   → purple/violet theme    (hue 0.72–0.82)
    Bandit-rbf-*   → warm orange/amber theme (hue 0.06–0.14)
    Bandit-cosine  → red/rose theme         (hue 0.93–0.98)
    Others         → fixed PALETTE or husl fallback
    """
    vendi_rbf_methods    = sorted(m for m in unique_methods
                                  if m.startswith("Vendi") and "cosine" not in m)
    vendi_cosine_methods = sorted(m for m in unique_methods
                                  if m.startswith("Vendi") and "cosine" in m)
    bandit_rbf_methods   = sorted(m for m in unique_methods
                                  if m.startswith("Bandit") and "cosine" not in m)
    bandit_cosine_methods = sorted(m for m in unique_methods
                                   if m.startswith("Bandit") and "cosine" in m)
    other_methods        = [m for m in unique_methods
                            if not m.startswith("Vendi") and not m.startswith("Bandit")]

    # Vendi-rbf  → cool teal/cyan (hue 0.47–0.60)
    vendi_rbf_colors    = hue_sweep_palette(len(vendi_rbf_methods), 0.47, 0.60)
    # Vendi-cosine → purple/violet (hue 0.72–0.82)
    vendi_cosine_colors = hue_sweep_palette(len(vendi_cosine_methods), 0.72, 0.82)

    # Bandit-rbf  → warm orange/amber (hue 0.06–0.14)
    bandit_rbf_colors    = hue_sweep_palette(len(bandit_rbf_methods), 0.06, 0.14)
    # Bandit-cosine → red/rose (hue 0.93–0.98)
    bandit_cosine_colors = hue_sweep_palette(len(bandit_cosine_methods), 0.93, 0.98)

    other_colors = sns.color_palette("husl", len(other_methods)) if other_methods else []

    palette = {}
    for i, m in enumerate(vendi_rbf_methods):
        palette[m] = vendi_rbf_colors[i]
    for i, m in enumerate(vendi_cosine_methods):
        palette[m] = vendi_cosine_colors[i]
    for i, m in enumerate(bandit_rbf_methods):
        palette[m] = bandit_rbf_colors[i]
    for i, m in enumerate(bandit_cosine_methods):
        palette[m] = bandit_cosine_colors[i]
    c_idx = 0
    for m in other_methods:
        if m in PALETTE:
            palette[m] = PALETTE[m]
        else:
            palette[m] = other_colors[c_idx]
            c_idx += 1
    return palette


def make_plot(plot_df, palette, title, linestyles=None, markers=None,
              xlabel="Active Learning Cycle", ylabel="Test Accuracy", show_std=True):
    """Create and return a matplotlib Figure.

    Plots each method individually so linestyle, color and marker can vary.
    linestyles: optional dict {method_name: linestyle}, e.g. {'Vendi-cosine': ':'}
    markers:    optional dict {method_name: marker},    e.g. {'Vendi-rbf-1.0': '^'}
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for method in plot_df["Method"].unique():
        mdf     = plot_df[plot_df["Method"] == method]
        color   = palette.get(method)
        ls      = (linestyles or {}).get(method, "-")
        mk      = (markers   or {}).get(method, "o")
        grouped = mdf.groupby("Cycle")["Accuracy"]
        means   = grouped.mean()
        stds    = grouped.std().fillna(0)

        ax.plot(means.index, means.values,
                marker=mk, linestyle=ls, color=color, label=method)
        if show_std:
            ax.fill_between(means.index,
                            means - stds, means + stds,
                            alpha=0.15, color=color)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_regime(
    regime_dir: Path,
    skip_before: str | None,
    skip_after: str | None,
    include_kernel: bool,
    include_no_kernel: bool,
    allowed_methods: set[str] | None,
    not_finished: list,
    no_csv_found: list,
    exceptions: list,
) -> tuple[list, list]:
    """Load all valid experiment runs under *regime_dir*.

    Parameters
    ----------
    include_kernel:    include folders that *have* a kernel spec
    include_no_kernel: include folders that *don't* have a kernel spec
    allowed_methods:   if not None, only include methods whose base name is in this set
    """
    dataset_name = regime_dir.parent.name
    regime_name  = regime_dir.name
    all_data: list[pd.DataFrame] = []
    aubc_results: list[dict] = []

    skip_before_dt = (datetime.datetime.strptime(skip_before, "%Y-%m-%d_%H-%M-%S-%f")
                      if skip_before else None)
    skip_after_dt  = (datetime.datetime.strptime(skip_after,  "%Y-%m-%d_%H-%M-%S-%f")
                      if skip_after  else None)

    for exp_dir in regime_dir.iterdir():
        if not exp_dir.is_dir():
            continue

        acq_key = get_acq(exp_dir.name)
        if acq_key is None or acq_key not in ALL_QUERYMETHODS:
            continue

        # --- folder-type filter (vendi / bandit only) ---
        if acq_key in ("vendi", "bandit"):
            folder_has_kernel = has_kernel(exp_dir.name)
            if include_kernel and include_no_kernel:
                # Both flags active: for vendi/bandit prefer kernel variants only.
                if not folder_has_kernel:
                    continue
            elif folder_has_kernel and not include_kernel:
                continue
            elif not folder_has_kernel and not include_no_kernel:
                continue
        else:
            folder_has_kernel = False  # other methods never have kernel in name

        # Build method label (kernel appended when folder has one)
        label = method_label(exp_dir.name, acq_key, use_kernel_in_label=folder_has_kernel)

        # --- query-method filter ---
        # Two modes:
        #   exact override: any entry in allowed_methods starts with "{base_name}-"
        #                   → only include if the full label matches one of those entries
        #   base-name match: no exact overrides for this base
        #                   → include if base_name is in allowed_methods (or no filter at all)
        base_name = ALL_QUERYMETHODS[acq_key]
        if allowed_methods is not None:
            exact_overrides = {m for m in allowed_methods if m.startswith(base_name + "-")}
            if exact_overrides:
                if label not in exact_overrides:
                    continue
            else:
                if base_name not in allowed_methods:
                    continue

        # --- load seeds ---
        seeds_data: list[pd.DataFrame] = []
        for seed_dir in exp_dir.iterdir():
            if not seed_dir.is_dir():
                continue

            # timestamp filter
            try:
                run_dt = datetime.datetime.strptime(seed_dir.name, "%Y-%m-%d_%H-%M-%S-%f")
                if skip_before_dt and run_dt < skip_before_dt:
                    continue
                if skip_after_dt  and run_dt > skip_after_dt:
                    continue
            except ValueError:
                pass  # non-timestamped subdirectory — include it

            metric_file = seed_dir / "test_metrics.csv"
            if not metric_file.exists():
                no_csv_found.append(seed_dir)
                continue

            try:
                df = pd.read_csv(metric_file)
                if len(df) < 10:
                    not_finished.append(seed_dir)
                    continue
                df = df.reset_index(drop=True)
                df.rename(columns={"test/acc": "Accuracy"}, inplace=True)
                df["Method"] = label
                df["Cycle"]  = df.index
                df["Run"]    = seed_dir.name
                seeds_data.append(df)

                aubc_results.append({
                    "Dataset": dataset_name,
                    "Regime":  regime_name,
                    "Method":  label,
                    "Run":     seed_dir.name,
                    "AUBC":    get_aubc(df["Accuracy"].values),
                })
            except Exception as exc:
                print(f"  [ERROR] {seed_dir}: {exc}")
                exceptions.append(seed_dir)

        n_seeds = len(seeds_data)
        if n_seeds < 3:
            print(f"  [WARNING] {label} ({exp_dir.name}): only {n_seeds}/3 seeds have results.")

        all_data.extend(seeds_data)

    return all_data, aubc_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    base_path = Path(args.results_path)
    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    prefix    = args.prefix or "plot"
    suffix    = f"_{args.suffix}" if args.suffix else ""
    title     = args.title  or ""
    show_std  = not args.no_std

    include_kernel    = args.kernel
    include_no_kernel = args.no_kernel
    if not include_kernel and not include_no_kernel:
        # default: include both
        include_kernel = include_no_kernel = True

    # Normalise --query-methods to the display-name case used in ALL_QUERYMETHODS
    # (e.g. "vendi" → "Vendi", "BALD" → "BALD") so the comparison always works.
    _display_names = {v.lower(): v for v in ALL_QUERYMETHODS.values()}
    allowed_methods: set[str] | None = (
        {_display_names.get(m.lower(), m) for m in args.query_methods}
        if args.query_methods else None
    )

    not_finished: list = []
    no_csv_found: list = []
    exceptions:   list = []

    allowed_datasets: set[str] | None = set(args.dataset) if args.dataset else None

    for dataset_dir in sorted(base_path.iterdir()):
        if not dataset_dir.is_dir() or dataset_dir.name in SKIP_DIR:
            print(f"Skipping dataset: {dataset_dir.name}")
            continue
        if allowed_datasets is not None and dataset_dir.name not in allowed_datasets:
            continue

        for regime_dir in sorted(dataset_dir.iterdir()):
            if not regime_dir.is_dir() or regime_dir.name in SKIP_DIR:
                continue

            print(f"\n{'='*60}")
            print(f"  {dataset_dir.name} / {regime_dir.name}")
            print(f"{'='*60}")

            all_data, aubc_results = load_regime(
                regime_dir,
                skip_before=args.skip_before,
                skip_after=args.skip_after,
                include_kernel=include_kernel,
                include_no_kernel=include_no_kernel,
                allowed_methods=allowed_methods,
                not_finished=not_finished,
                no_csv_found=no_csv_found,
                exceptions=exceptions,
            )

            if not all_data:
                print("  No data found — skipping.")
                continue

            plot_df = pd.concat(all_data, ignore_index=True)
            unique_methods = plot_df["Method"].unique().tolist()
            palette = build_palette(unique_methods)

            # Dotted lines for cosine-kernel methods, solid for everything else.
            linestyles = {m: ":" if "cosine" in m else "-" for m in unique_methods}

            # Markers: one shape per base method family so variants are visually grouped.
            # All Vendi-* share a triangle-up, all Bandit-* share a square, etc.
            BASE_MARKERS = {
                "Vendi":    "^",   # triangle up
                "Bandit":   "s",   # square
                "BALD":     "o",   # circle
                "Random":   "X",   # thick x
                "Core-Set": "D",   # diamond
                "BADGE":    "P",   # thick plus
                "Entropy":  "v",   # triangle down
            }
            markers = {}
            for m in unique_methods:
                base = next((b for b in BASE_MARKERS if m.startswith(b)), None)
                markers[m] = BASE_MARKERS[base] if base else "o"

            plot_title = f"{dataset_dir.name} - {regime_dir.name}"
            if title:
                plot_title += f" - {title}"

            # --- Full plot (all methods) ---
            fig = make_plot(plot_df, palette, plot_title,
                            linestyles=linestyles, markers=markers, show_std=show_std)
            fname = f"{prefix}_{dataset_dir.name}_{regime_dir.name}{suffix}.png"
            fig.savefig(save_path / fname, dpi=300)
            plt.close(fig)
            print(f"  Saved: {save_path / fname}")

            # --- Selected plot (Random, BALD, Vendi*, Bandit*) ---
            base_selected = {"Random", "BALD", "Vendi", "Bandit"}
            sel_methods = [m for m in unique_methods
                           if any(m.startswith(b) for b in base_selected)]
            sel_df = plot_df[plot_df["Method"].isin(sel_methods)]

            if not sel_df.empty:
                fig_sel = make_plot(sel_df, palette,
                                    plot_title + " (Selected)",
                                    linestyles=linestyles, markers=markers, show_std=show_std)
                sel_fname = fname.replace(".png", "_selected.png")
                fig_sel.savefig(save_path / sel_fname, dpi=300)
                plt.close(fig_sel)
                print(f"  Saved: {save_path / sel_fname}")

            # --- AUBC summary ---
            if aubc_results:
                aubc_df = pd.DataFrame(aubc_results)
                summary = (aubc_df.groupby("Method")["AUBC"]
                           .agg(["mean", "std"])
                           .reset_index()
                           .sort_values("mean", ascending=False))
                print("  AUBC Summary:")
                print(summary.to_string(index=False))
                aubc_fname = f"{prefix}_{dataset_dir.name}_{regime_dir.name}{suffix}_aubc.csv"
                summary.to_csv(save_path / aubc_fname, index=False)

    # --- Summary of issues ---
    if no_csv_found:
        print("\n[Missing test_metrics.csv]")
        for d in no_csv_found:
            print(f"  {d}")
    if not_finished:
        print("\n[Incomplete runs (< 10 rows)]")
        for d in not_finished:
            print(f"  {d}")
    if exceptions:
        print("\n[Exceptions during loading]")
        for d in exceptions:
            print(f"  {d}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot active-learning results with kernel/no-kernel folder selection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- output ---
    parser.add_argument("--prefix", type=str, default="plot",
                        help="Prefix for saved plot/CSV filenames.")
    parser.add_argument("--suffix", type=str, default=None,
                        help="Suffix appended to saved plot/CSV filenames before the extension, "
                             "e.g. --suffix kernel gives prefix_dataset_regime_kernel.png.")
    parser.add_argument("--title", type=str, default="",
                        help="Extra string appended to each plot title.")
    parser.add_argument("--no-std", action="store_true",
                        help="Disable shaded standard-deviation band.")
    parser.add_argument("--save-path", type=str, default="./plots_simple",
                        help="Directory to write plots and AUBC CSVs into.")

    # --- input ---
    parser.add_argument("--results-path", type=str,
                        default="/home/jing/Desktop/realistic-al/experiments/activelearning",
                        help="Root of the experiment results tree.")
    parser.add_argument("--skip-before", type=str, default=None,
                        help="Ignore runs whose timestamp is BEFORE this value "
                             "(format: YYYY-MM-DD_HH-MM-SS-ffffff).")
    parser.add_argument("--skip-after", type=str, default=None,
                        help="Ignore runs whose timestamp is AFTER this value.")

    # --- folder-type selection ---
    parser.add_argument("--kernel", action="store_true",
                        help="Include experiment folders that contain a kernel spec "
                             "(e.g. kernel-rbf_gamma-1.0). Legend label: 'Vendi-rbf-1.0'.")
    parser.add_argument("--no-kernel", action="store_true",
                        help="Include experiment folders that do NOT contain a kernel spec. "
                             "Legend label: 'Vendi'.")

    # --- dataset filter ---
    parser.add_argument("--dataset", nargs="+", default=None,
                        metavar="DATASET",
                        help="Only plot these dataset folder(s), e.g. --dataset cifar10_imb. "
                             "If omitted all datasets are plotted.")

    # --- method filter ---
    parser.add_argument("--query-methods", nargs="+", default=None,
                        metavar="METHOD",
                        help="Whitelist of base method names to include, e.g. "
                             "--query-methods BALD Vendi Bandit Random. "
                             "If omitted all methods are included.")

    args = parser.parse_args()
    main(args)
