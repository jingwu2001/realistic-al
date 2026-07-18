import argparse
import json

import datetime

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Maps acq-* folder key → display name. Folders with unknown keys are skipped.
QUERYMETHODS = {
    "bald": "BALD",
    "kcentergreedy": "Core-Set",
    "entropy": "Entropy",
    "random": "Random",
    "badge": "BADGE",
    "vendi": "Vendi",
    "bandit": "Bandit",
    # "batchbald": "BatchBALD",
    # "vendi-l2": "Vendi-L2",
    # "vendi-minmax": "Vendi-MinMax",
}

# Fixed colors for base method names (no kernel/gamma suffix).
PALETTE = {
    "BALD":     "#d62728",  # red
    "Core-Set": "#1f77b4",  # blue
    "Entropy":  "#2ca02c",  # green
    "BADGE":    "#9467bd",  # purple
    "Random":   "#6B6B6B",  # grey
    "Vendi":    "#3939C6",  # deep blue
    "Bandit":   "#e79e00",  # amber
}

# HLS hue ranges for auto-coloring kernel/gamma variants by family.
# Methods whose base name isn't listed here fall back to seaborn "husl".
FAMILY_HUE = {
    "Vendi":  (0.47, 0.60),  # teal → sky-blue (cool)
    "Bandit": (0.98, 0.10),  # crimson → amber  (warm, wraps past 1.0)
}

# Marker and dash-pattern per (kernel, gamma) combination.
# Variants within a family share the family's base color; marker+dash distinguish them.
KERNEL_MARKERS = {
    (None,     None):          "o",   # no kernel (standard)
    ("cosine", None):          "X",   # cosine kernel
    ("rbf",    "median"):      "v",   # median heuristic
    ("rbf",    0.0009765625):  "^",   # rbf γ=1/1024
    ("rbf",    0.1):           "D",   # rbf γ=0.1
    ("rbf",    1.0):           "s",   # rbf γ=1.0
    ("rbf",    10.0):          "P",   # rbf γ=10
}

KERNEL_DASHES = {
    (None,     None):          (1, 0),        # solid
    ("cosine", None):          (6, 2),        # long dash
    ("rbf",    "median"):      (3, 1, 1, 1),  # compact dash-dot
    ("rbf",    0.0009765625):  (2, 2),        # dotted
    ("rbf",    0.1):           (4, 2, 1, 2),  # dash-dot
    ("rbf",    1.0):           (6, 2, 1, 2),  # long dash-dot
    ("rbf",    10.0):          (4, 1, 1, 1),  # dense dash-dot
}

SKIP_DIR    = ["cifar10", "active-cifar10_high", "ecg5000", "p12_transformer"]
SKIP_REGIME = []

# Controls method naming and palette strategy.
# "method" → base name only (Vendi)
# "kernel" → base + kernel type + gamma (Vendi-rbf-0.1)
granularity = "kernel"

EVAL_METRICS = ["precision_cls1", "recall_cls1", "f1_cls1", "auroc", "auprc"]
EVAL_METRIC_LABELS = {
    "precision_cls1": "Precision",
    "recall_cls1":    "Recall",
    "f1_cls1":        "F1",
    "auroc":          "AUROC",
    "auprc":          "AUPRC",
}

sns.set_style("whitegrid")


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_exp_dir(exp_dir: Path) -> dict:
    """Parse folder name fields into a structured dict.

    Example folder:
      basic_model-resnet_drop-0.5_aug-cifar_randaugmentMC_acq-vendi_norm-minmax_kernel-rbf_gamma-0.1_ep-200__wloss-True
    Returns: {exp_dir, acq, kernel, gamma, drop}
    """
    acq = kernel = drop = None
    gamma = None
    for part in exp_dir.name.split("_"):
        if part.startswith("acq-"):
            acq = part[4:]          # e.g. "vendi", "bald", "kcentergreedy"
        elif part.startswith("kernel-"):
            kernel = part[7:]       # e.g. "rbf", "cosine"
        elif part.startswith("gamma-"):
            val = part[6:]
            if val == "None":
                gamma = None
            else:
                try:
                    gamma = float(val)
                except ValueError:
                    gamma = val
        elif part.startswith("drop-"):
            drop = part[5:]         # e.g. "0.5", "0"
    return {"exp_dir": exp_dir, "acq": acq, "kernel": kernel, "gamma": gamma, "drop": drop}


def get_method_name(acq: str, kernel, gamma, gran: str) -> str:
    """Build the display label from parsed fields according to granularity."""
    base = QUERYMETHODS.get(acq, acq)
    if gran == "method":
        return base
    # "kernel" granularity: differentiate by full kernel config (type + gamma)
    if kernel and kernel is not None:
        base += f"-{kernel}"
        if gamma is not None:
            base += f"-{gamma}"
    return base


# ---------------------------------------------------------------------------
# Palette building
# ---------------------------------------------------------------------------


def build_palette(methods, distinct=False) -> dict:
    """Assign colors to exactly the methods given.

    distinct=False (default): all variants within a family share the family's
      base PALETTE color; line style and marker carry the within-family distinction.
    distinct=True: every method gets its own color from a high-contrast husl
      palette, ignoring family grouping. Use this for kernel/gamma sweep plots.
    """
    methods = list(methods)
    if distinct:
        colors = sns.color_palette("husl", len(methods))
        return {m: c for m, c in zip(methods, colors)}

    palette = {}
    unknown_families: list[str] = []
    for m in methods:
        if m in PALETTE:
            palette[m] = PALETTE[m]
        else:
            family = m.split("-")[0]
            if family in PALETTE:
                palette[m] = PALETTE[family]  # same color as base family member
            elif family not in unknown_families:
                unknown_families.append(family)
    if unknown_families:
        fallback = list(sns.color_palette("husl", len(unknown_families)))
        family_color = dict(zip(unknown_families, fallback))
        for m in methods:
            if m not in palette:
                palette[m] = family_color[m.split("-")[0]]
    return palette


def get_marker(kernel, gamma) -> str:
    return KERNEL_MARKERS.get((kernel, gamma), "o")


def get_dash(kernel, gamma):
    return KERNEL_DASHES.get((kernel, gamma), (1, 0))


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def _gamma_matches(gamma, allowed: list[str]) -> bool:
    if str(gamma) in allowed:
        return True
    try:
        gamma_float = float(gamma)
    except (TypeError, ValueError):
        return False
    for candidate in allowed:
        try:
            if np.isclose(gamma_float, float(candidate), rtol=0.0, atol=1e-4):
                return True
        except ValueError:
            continue
    return False


def should_include(exp: dict, dir_filter, allowed_gammas, acq_gammas, active_methods, kernel_methods=None, acq_exclude_paths=None) -> bool:
    """Return True if this experiment directory should be loaded.

    Gamma filtering (both only apply to dirs that have a kernel specified):
      allowed_gammas: global list of allowed gamma values (all methods).
      acq_gammas:     per-method override dict {acq_key: [gamma_str, ...]};
                      when an acq has an entry here it ignores allowed_gammas.
    Dirs with no kernel are always kept regardless of gamma filters.
    acq_exclude_paths: per-method path exclusion {acq_key: [path_substring, ...]};
                       excludes dirs whose full path contains any listed substring.
    """
    if dir_filter and dir_filter not in exp["exp_dir"].name:
        return False
    if exp["acq"] not in active_methods:
        return False
    if acq_exclude_paths and exp["acq"] in acq_exclude_paths:
        exp_path_str = str(exp["exp_dir"])
        if any(ep in exp_path_str for ep in acq_exclude_paths[exp["acq"]]):
            return False
    if exp["kernel"] is not None:
        if kernel_methods is not None and exp["acq"] not in kernel_methods:
            return False
        # Determine which gamma list applies for this method.
        if acq_gammas and exp["acq"] in acq_gammas:
            gammas = acq_gammas[exp["acq"]]
        elif allowed_gammas is not None:
            gammas = allowed_gammas
        else:
            gammas = None
        if gammas is not None:
            if exp["gamma"] is None or not _gamma_matches(exp["gamma"], gammas):
                return False
    return True


def _discover_exp_dirs(regime_dir: Path) -> list[Path]:
    """Find experiment directories, including dirs nested under commit-* containers."""
    exp_dirs = []
    for d in regime_dir.rglob("*"):
        if not d.is_dir():
            continue
        parsed = parse_exp_dir(d)
        if parsed["acq"] in QUERYMETHODS:
            exp_dirs.append(d)
    return exp_dirs


def _dedupe_unspecified_vendi_gamma(exps: list[dict]) -> list[dict]:
    """Drop no-kernel Vendi dirs when the explicit RBF gamma=1.0 dir is present."""
    has_explicit_vendi_1 = any(
        exp["acq"] == "vendi" and exp["kernel"] == "rbf" and exp["gamma"] == 1.0
        for exp in exps
    )
    if not has_explicit_vendi_1:
        return exps
    return [
        exp for exp in exps
        if not (exp["acq"] == "vendi" and exp["kernel"] is None and exp["gamma"] is None)
    ]


# ---------------------------------------------------------------------------
# Seed loading
# ---------------------------------------------------------------------------

def _select_seed_dirs(exp_dir: Path, skip_before, skip_after) -> list:
    """Return up to the last 3 seed run dirs within the optional time window."""
    t_min = datetime.datetime.strptime(skip_before, "%Y-%m-%d_%H-%M-%S-%f") if skip_before else None
    t_max = datetime.datetime.strptime(skip_after,  "%Y-%m-%d_%H-%M-%S-%f") if skip_after  else None
    all_dirs = sorted([d for d in exp_dir.iterdir() if d.is_dir()], key=lambda x: x.name)
    filtered = []
    for d in all_dirs:
        try:
            t = datetime.datetime.strptime(d.name, "%Y-%m-%d_%H-%M-%S-%f")
            if t_min and t < t_min:
                continue
            if t_max and t > t_max:
                continue
        except ValueError:
            pass  # non-timestamp dirs (e.g. .hydra) are kept as-is
        filtered.append(d)
    return filtered[-3:]


def _load_seed_df(seed_dir: Path, not_finished: list, no_df_found: list):
    """Load and return a cleaned DataFrame for one seed, or None on failure."""
    metric_file = seed_dir / "test_metrics.csv"
    if not metric_file.exists():
        no_df_found.append(seed_dir)
        return None
    df = pd.read_csv(metric_file)
    if len(df) < 8:
        not_finished.append(seed_dir)
        return None
    df = df.reset_index(drop=True)
    df.rename(columns={"test/acc": "Accuracy", "test/bacc": "Balanced Accuracy",
                        "test/auroc": "auroc", "test/auprc": "auprc"}, inplace=True)
    df["Cycle"] = df.index
    df["Run"]   = seed_dir.name

    # eval_metrics.csv has correct binary AUROC/AUPRC — overrides inflated multiclass values.
    eval_file = seed_dir / "eval_metrics.csv"
    if eval_file.exists():
        eval_df = pd.read_csv(eval_file).rename(columns={"loop": "Cycle"})
        available = [c for c in EVAL_METRICS if c in eval_df.columns]
        if available:
            df = df.drop(columns=[c for c in available if c in df.columns], errors="ignore")
            df = df.merge(eval_df[["Cycle"] + available], on="Cycle", how="left")
    return df


# ---------------------------------------------------------------------------
# Regime loading  (parse → filter → name → load seeds → aggregate)
# ---------------------------------------------------------------------------

def _load_regime(regime_dir, skip_before, skip_after,
                  not_finished, no_df_found, met_exceptions,
                  dir_filter=None, query_methods=None, allowed_gammas=None, acq_gammas=None, kernel_methods=None, acq_exclude_paths=None):
    """Load all experiment runs under one regime directory.

    Returns (all_data_list, aubc_results_list, runs_used_dict, method_markers_dict).
    """
    dataset_name = regime_dir.parent.name
    regime_name  = regime_dir.name
    active_methods = query_methods if query_methods else QUERYMETHODS

    all_data       = []
    aubc_results   = []
    runs_used      = {}
    method_markers = {}  # method_name → matplotlib marker string
    method_dashes  = {}  # method_name → matplotlib dash tuple

    # 1. Parse every subdirectory into structured metadata.
    exps = [parse_exp_dir(d) for d in _discover_exp_dirs(regime_dir)]

    # 2. Filter by dir_filter, allowed_gammas, and known acq methods.
    exps = [e for e in exps if should_include(e, dir_filter, allowed_gammas, acq_gammas, active_methods, kernel_methods, acq_exclude_paths)]
    exps = _dedupe_unspecified_vendi_gamma(exps)

    # 3. For each surviving experiment, load its seed runs.
    for exp in exps:
        method_name = get_method_name(exp["acq"], exp["kernel"], exp["gamma"], granularity)
        method_markers[method_name] = get_marker(exp["kernel"], exp["gamma"])
        method_dashes[method_name]  = get_dash(exp["kernel"], exp["gamma"])
        seed_dirs   = _select_seed_dirs(exp["exp_dir"], skip_before, skip_after)
        runs_used[method_name] = [str(d) for d in seed_dirs]

        seeds_data = []
        for seed_dir in seed_dirs:
            try:
                df = _load_seed_df(seed_dir, not_finished, no_df_found)
                if df is None:
                    continue
                df["Method"] = method_name
                seeds_data.append(df)
                aubc_results.append({
                    "Dataset": dataset_name,
                    "Regime":  regime_name,
                    "Method":  method_name,
                    "Run":     seed_dir.name,
                    "AUBC":    get_aubc(df["Accuracy"].values),
                })
            except Exception:
                met_exceptions.append(seed_dir)

        all_data.extend(seeds_data)
        if len(seeds_data) < 3:
            print(f"  Warning: {method_name} ({exp['exp_dir']}) has fewer than 3 seeds ({len(seeds_data)}).")

    return all_data, aubc_results, runs_used, method_markers, method_dashes


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def get_aubc(performance: np.ndarray) -> float:
    """Area Under the Budget Curve (trapz, x normalised to [0,1])."""
    if len(performance) <= 1:
        return 0.0
    return np.trapezoid(performance, dx=1 / (len(performance) - 1)).item()


# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------

def _save_lineplot(plot_df, y, ylabel, title, palette, markers, dashes, show_std, path):
    method_order = list(plot_df["Method"].unique())
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=plot_df, x="Cycle", y=y,
                 hue="Method", hue_order=method_order,
                 style="Method", style_order=method_order,
                 palette=palette,
                 markers=[markers[m] for m in method_order],
                 dashes=[dashes[m] for m in method_order],
                 ci="sd" if show_std else None)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Active Learning Cycle")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(prefix, title, show_std=True, dir_filter=None, query_methods=None, allowed_gammas=None, acq_gammas=None, kernel_methods=None, distinct_colors=False, acq_exclude_paths=None):
    base_path  = Path(args.results_path)
    save_path  = Path(args.save_path) / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {save_path}")
    if args.note:
        (save_path / "note.txt").write_text(args.note + "\n")

    not_finished   = []
    no_df_found    = []
    met_exceptions = []
    processed      = []

    for dataset_dir in base_path.iterdir():
        if not dataset_dir.is_dir() or dataset_dir.name != args.dataset:
            print(f"Skipping: {dataset_dir}")
            continue

        for regime_dir in sorted(dataset_dir.iterdir(), key=lambda x: x.name):
            if not regime_dir.is_dir() or regime_dir.name in SKIP_REGIME or regime_dir.name in SKIP_DIR:
                continue
            if args.regime and regime_dir.name != args.regime:
                continue
            print(f"  {dataset_dir.name} / {regime_dir.name}")

            all_data, aubc_results, runs_used, method_markers, method_dashes = _load_regime(
                regime_dir, args.skip_before, args.skip_after,
                not_finished, no_df_found, met_exceptions,
                dir_filter=dir_filter, query_methods=query_methods, allowed_gammas=allowed_gammas, acq_gammas=acq_gammas, kernel_methods=kernel_methods, acq_exclude_paths=acq_exclude_paths,
            )

            with open(save_path / f"{prefix}_{dataset_dir.name}_{regime_dir.name}_runs.json", "w") as f:
                json.dump(runs_used, f, indent=2)

            print(f"  Loaded {len(all_data)} seed DataFrames.")
            if not all_data:
                print("  No data found, skipping.")
                continue

            plot_df = pd.concat(all_data, ignore_index=True)
            unique_methods = list(plot_df["Method"].unique())
            palette = build_palette(unique_methods, distinct=distinct_colors)
            markers = {m: method_markers.get(m, "o")      for m in unique_methods}
            dashes  = {m: method_dashes.get(m,  (1, 0))   for m in unique_methods}
            tag     = f"{dataset_dir.name} - {regime_dir.name}"
            pfx     = f"{prefix}_{dataset_dir.name}_{regime_dir.name}"

            _save_lineplot(plot_df, "Accuracy",         "Test Accuracy",          f"{tag} - Accuracy - {title}",          palette, markers, dashes, show_std, save_path / f"{pfx}_acc.png")
            _save_lineplot(plot_df, "Balanced Accuracy", "Test Balanced Accuracy", f"{tag} - Balanced Accuracy - {title}", palette, markers, dashes, show_std, save_path / f"{pfx}_bacc.png")

            for col in EVAL_METRICS:
                if col not in plot_df.columns:
                    continue
                label = EVAL_METRIC_LABELS.get(col, col)
                _save_lineplot(plot_df, col, label, f"{tag} - {label} - {title}", palette, markers, dashes, show_std, save_path / f"{pfx}_{col}.png")

            if aubc_results:
                aubc_df      = pd.DataFrame(aubc_results)
                aubc_summary = aubc_df.groupby("Method")["AUBC"].agg(["mean", "std"]).reset_index()
                aubc_summary = aubc_summary.sort_values("mean", ascending=False)
                print("  AUBC Summary:")
                print(aubc_summary.to_string(index=False))
                aubc_summary.to_csv(save_path / f"{pfx}_aubc.csv", index=False)

            processed.append(regime_dir)

    print("\nProcessed:", processed)
    if no_df_found:
        print("test_metrics.csv not found:")
        for d in no_df_found: print(" ", d)
    if not_finished:
        print("test_metrics.csv has < 8 rows:")
        for d in not_finished: print(" ", d)
    if met_exceptions:
        print("Exceptions encountered:")
        for d in met_exceptions: print(" ", d)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_acq_gammas(raw: list | None) -> dict | None:
    """Parse ['bandit:0.0009765625', 'vendi:0.1'] → {'bandit': ['0.0009765625'], 'vendi': ['0.1']}."""
    if not raw:
        return None
    result = {}
    for s in raw:
        method, gamma = s.split(":", 1)
        result.setdefault(method, []).append(gamma)
    return result


def _parse_acq_exclude_paths(raw: list | None) -> dict | None:
    """Parse ['bandit:commit-1762dfa9'] → {'bandit': ['commit-1762dfa9']}."""
    if not raw:
        return None
    result = {}
    for s in raw:
        method, path_sub = s.split(":", 1)
        result.setdefault(method, []).append(path_sub)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot active learning experiment results.")
    parser.add_argument("--prefix",         type=str,        required=True,  help="Filename prefix for saved outputs.")
    parser.add_argument("--title",          type=str,        default="",     help="Plot title suffix.")
    parser.add_argument("--no-std",         action="store_true",             help="Disable shaded std band.")
    parser.add_argument("--distinct-colors", action="store_true",            help="Give each method its own color (use for kernel/gamma sweep plots).")
    parser.add_argument("--dataset",        type=str,        default="ecg5000", help="Dataset folder name (e.g. cifar10_imb).")
    parser.add_argument("--regime",         type=str,        default=None,   help="Regime folder name to plot (e.g. active-p12_med_bal). Plots all regimes if omitted.")
    parser.add_argument("--query-methods",  nargs="+",       default=None,   help="Allowlist of acq keys, e.g. --query-methods vendi bandit.")
    parser.add_argument("--dir-filter",     type=str,        default=None,   help="Substring that experiment folder names must contain, e.g. drop-0.5.")
    parser.add_argument("--allowed-gammas", nargs="+",       default=None,   help="Gamma values to allow for all methods (dirs with no kernel are always kept).")
    parser.add_argument("--acq-gammas",        nargs="+",       default=None,   help="Per-method gamma allowlists: 'acq:gamma' pairs, e.g. --acq-gammas bandit:0.0009765625. Overrides --allowed-gammas for that method.")
    parser.add_argument("--acq-exclude-paths", nargs="+",       default=None,   help="Per-method path exclusion: 'acq:path_substring' pairs, e.g. --acq-exclude-paths bandit:commit-1762dfa9.")
    parser.add_argument("--kernel-methods", nargs="+",       default=None,   help="Allow kernel/gamma variants only for these acq keys. Non-kernel dirs are still kept.")
    parser.add_argument("--skip-before",    type=str,        default=None,   help="Exclude runs before this timestamp (YYYY-MM-DD_HH-MM-SS-ffffff).")
    parser.add_argument("--skip-after",     type=str,        default=None,   help="Exclude runs after this timestamp (YYYY-MM-DD_HH-MM-SS-ffffff).")
    parser.add_argument("--note",           type=str,        default=None,   help="Free-text note saved as note.txt in the output folder.")
    parser.add_argument("--results-path",   type=str,        default="/home/jing/Desktop/realistic-al/experiments/activelearning")
    parser.add_argument("--save-path",      type=str,        default="/home/jing/Desktop/realistic-al/analysis/plots_simple")
    args = parser.parse_args()

    active_query_methods = None
    if args.query_methods:
        active_query_methods = {k: v for k, v in QUERYMETHODS.items() if k in args.query_methods}

    main(args.prefix, args.title, show_std=not args.no_std,
         dir_filter=args.dir_filter, query_methods=active_query_methods,
         allowed_gammas=args.allowed_gammas,
         acq_gammas=_parse_acq_gammas(args.acq_gammas),
         kernel_methods=set(args.kernel_methods) if args.kernel_methods else None,
         distinct_colors=args.distinct_colors,
         acq_exclude_paths=_parse_acq_exclude_paths(args.acq_exclude_paths))
