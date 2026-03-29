import sys
import argparse
import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import datetime
import colorsys

# --- Configuration ---

# Mapping from experiment folder name key to display name
# Ensure keys match 'acq-{key}' in folder names
QUERYMETHODS = {
    "bald": "BALD",
    "kcentergreedy": "Core-Set",
    "entropy": "Entropy",
    "random": "Random",
    "badge": "BADGE",
    # "batchbald": "BatchBALD",
    # "vendi-l2": "Vendi-L2",
    # "vendi-minmax": "Vendi-MinMax",
    "vendi": "Vendi",
    "bandit": "Bandit",
}

PALETTE = {
    "BALD": "#d62728",      # distinct red
    "Core-Set": "#1f77b4",  # distinct blue
    "Entropy": "#2ca02c",   # distinct green
    "BADGE": "#9467bd",     # distinct purple
    "Random": "#6B6B6B",    # neutral grey
    "Vendi": "#3939C6",     # distinct orange ###
    "Bandit": "#e79e00",    # distinct brown
}

SKIP_DIR = [
    'cifar10',
    'active-cifar10_high'
]

SKIP_REGIME = [
    # 'active-ecg5000_high',
]

granularity = "kernel"

# SKIP_CRITERION = lambda x: 'drop-0' in x

sns.set_style("whitegrid")


def get_kernel_params(x: str) -> dict:
    """Extracts kernel parameters from experiment folder name."""
    kernel = None
    gamma = None
    l = x.split('_')
    for i in l:
        if i.startswith('kernel-'):
            kernel = i.split('-')[1]
        if i.startswith('gamma-'):
            gamma = i.split('-')[1]
            gamma = None if gamma == 'None' else float(gamma)
    return {'kernel': kernel, 'gamma': gamma}

def get_acq(x: str) -> dict:
    """Extracts acquisition parameters from experiment folder name."""
    acq = None
    l = x.split('_')
    for i in l:
        if i.startswith('acq-'):
            acq = i.split('-')[1]
    return {'acq': acq}
    

def get_aubc(performance: np.ndarray) -> float:
    """Computes the Area Under the Budget Curve.
    Integral goes from 0 to 1 over the number of cycles.
    """
    if len(performance) <= 1:
        return 0.0
    dx = 1 / (len(performance) - 1)
    return np.trapz(performance, dx=dx).item()



def _load_regime(regime_dir, skip_before, skip_after, label_suffix, not_finished, no_df_found, met_exceptions):
    """Load all experiment runs under a single regime directory.

    Returns (all_data_list, aubc_results_list, dataset_name, regime_name).
    """
    dataset_name = regime_dir.parent.name
    regime_name = regime_dir.name
    all_data = []
    aubc_results = []
    runs_used = {}

    for exp_dir in regime_dir.iterdir():
        if not exp_dir.is_dir():
            continue

        method_key = get_acq(exp_dir.name).get('acq')
        if not method_key or method_key not in QUERYMETHODS:
            continue

        method_name = QUERYMETHODS[method_key]

        if granularity == "kernel":
            kernel_params = get_kernel_params(exp_dir.name)
            if kernel_params['kernel']:
                method_name += f"-{kernel_params['kernel']}"
            if kernel_params['gamma'] is not None:
                method_name += f"-{kernel_params['gamma']}"

        if label_suffix:
            method_name += f" {label_suffix}"

        seeds_data = []
        skip_before_time = datetime.datetime.strptime(skip_before, "%Y-%m-%d_%H-%M-%S-%f") if skip_before else None
        skip_after_time = datetime.datetime.strptime(skip_after, "%Y-%m-%d_%H-%M-%S-%f") if skip_after else None

        all_seed_dirs = sorted([d for d in exp_dir.iterdir() if d.is_dir()], key=lambda x: x.name)
        filtered_seed_dirs = []
        for d in all_seed_dirs:
            try:
                run_time = datetime.datetime.strptime(d.name, "%Y-%m-%d_%H-%M-%S-%f")
                if (skip_before_time and run_time < skip_before_time) or (skip_after_time and run_time > skip_after_time):
                    continue
            except ValueError:
                pass
            filtered_seed_dirs.append(d)

        selected_seed_dirs = filtered_seed_dirs[-3:]
        runs_used[method_name] = [str(d) for d in selected_seed_dirs]

        for seed_dir in selected_seed_dirs:

            metric_file = seed_dir / "test_metrics.csv"
            if not metric_file.exists():
                no_df_found.append(seed_dir)
                continue

            try:
                df = pd.read_csv(metric_file)
                if len(df) < 8:
                    not_finished.append(seed_dir)
                    continue
                df = df.reset_index(drop=True)
                df.rename(columns={"test/acc": "Accuracy", "test/bacc": "Balanced Accuracy"}, inplace=True)
                df["Method"] = method_name
                df["Cycle"] = df.index
                df["Run"] = seed_dir.name
                seeds_data.append(df)

                acc_values = df["Accuracy"].values
                aubc = get_aubc(acc_values)
                aubc_results.append({
                    "Dataset": dataset_name,
                    "Regime": regime_name,
                    "Method": method_name,
                    "Run": seed_dir.name,
                    "AUBC": aubc,
                })
            except Exception:
                met_exceptions.append(seed_dir)
                continue

        if seeds_data:
            all_data.extend(seeds_data)
        if len(seeds_data) < 3:
            print(f"Warning: {method_name} has less than 3 seeds.")

    return all_data, aubc_results, runs_used


def main(prefix, title, show_std=True):
    processed = []
    base_path = Path(args.results_path)
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = Path(args.save_path) / run_timestamp
    save_path.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {save_path}")

    if args.note:
        (save_path / "note.txt").write_text(args.note + "\n")
    not_finished = []
    no_df_found = []
    met_exceptions = []
    # --- Iterate over Datasets (e.g. cifar10_imb) ---
    for dataset_dir in base_path.iterdir():
        if not dataset_dir.is_dir() or dataset_dir.name != "ecg5000":
            print(f"Skipping: {dataset_dir}")
            continue

        # --- Iterate over Label Regimes (e.g. active-cifar10_med) ---
        for regime_dir in sorted(dataset_dir.iterdir(), key=lambda x: x.name):
            if not regime_dir.is_dir() or regime_dir.name in SKIP_REGIME or regime_dir.name in SKIP_DIR:
                continue
            print(f"  {dataset_dir.name} / {regime_dir.name}")
            all_data = []
            aubc_results = []

            d, a, runs_used = _load_regime(regime_dir, args.skip_before, args.skip_after, "",
                                           not_finished, no_df_found, met_exceptions)
            all_data.extend(d)
            aubc_results.extend(a)

            runs_json_path = save_path / f"{prefix}_{dataset_dir.name}_{regime_dir.name}_runs.json"
            with open(runs_json_path, "w") as f:
                json.dump(runs_used, f, indent=2)

            print(f"  ALL_DATA: {len(all_data)} runs")
            if not all_data:
                print("  No data found, skipping.")
                continue
            # Combine all for plotting
            plot_df = pd.concat(all_data, ignore_index=True)

            # --- Dynamic Palette ---
            unique_methods = plot_df["Method"].unique()
            current_palette = {}

            vendi_methods = sorted([m for m in unique_methods if m.startswith("Vendi")])
            bandit_methods = sorted([m for m in unique_methods if m.startswith("Bandit")])
            other_methods = [m for m in unique_methods if m not in vendi_methods and m not in bandit_methods]

            # Sweep hue across a range at fixed lightness+saturation.
            # Vendi  → cool theme: teal → cyan → sky-blue  (hue 0.47 → 0.60)
            # Bandit → warm theme: crimson → orange → amber (hue 0.97 → 0.10, wraps)
            def hue_sweep_palette(n, hue_min, hue_max, lightness=0.40, saturation=0.80):
                """Return n distinct colors by linearly sweeping hue in HLS space."""
                if n <= 0:
                    return []
                hues = np.linspace(hue_min, hue_max, n) % 1.0
                return [colorsys.hls_to_rgb(h, lightness, saturation) for h in hues]

            current_palette = PALETTE

            # # Vendi: teal/cyan/sky-blue family  (cool)
            # vendi_colors = hue_sweep_palette(len(vendi_methods), 0.47, 0.60) if vendi_methods else []
            # # Bandit: crimson/orange/amber family  (warm)
            # n_bandit = len(bandit_methods)
            # if n_bandit == 1:
            #     bandit_colors = [colorsys.hls_to_rgb(0.04, 0.45, 0.85)]  # vivid orange-red
            # elif n_bandit > 1:
            #     bandit_colors = hue_sweep_palette(n_bandit, 0.98, 0.10)
            # else:
            #     bandit_colors = []
            # other_colors = sns.color_palette("husl", len(other_methods)) if other_methods else []

            # for i, m in enumerate(vendi_methods):
            #     current_palette[m] = vendi_colors[i]

            # for i, m in enumerate(bandit_methods):
            #     current_palette[m] = bandit_colors[i]

            # c_idx = 0
            # for m in other_methods:
            #     if m in PALETTE:
            #         current_palette[m] = PALETTE[m]
            #     else:
            #         current_palette[m] = other_colors[c_idx]
            #         c_idx += 1

            # --- Plotting Accuracy ---
            plt.figure(figsize=(10, 6))
            sns.lineplot(
                data=plot_df,
                x="Cycle",
                y="Accuracy",
                hue="Method",
                palette=current_palette,
                ci="sd" if show_std else None,  # Standard Deviation
                marker="o"
            )
            plt.title(f"{dataset_dir.name} - {regime_dir.name} - Accuracy - {title}")
            plt.ylabel("Test Accuracy")
            plt.xlabel("Active Learning Cycle")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()

            filename = f"{prefix}_{dataset_dir.name}_{regime_dir.name}_acc.png"
            plt.savefig(save_path / filename, dpi=300)
            plt.close()
            print(f"  Saved plot to {save_path / filename}")

            # --- Plotting Balanced Accuracy ---
            plt.figure(figsize=(10, 6))
            sns.lineplot(
                data=plot_df,
                x="Cycle",
                y="Balanced Accuracy",
                hue="Method",
                palette=current_palette,
                ci="sd" if show_std else None,  # Standard Deviation
                marker="o"
            )
            plt.title(f"{dataset_dir.name} - {regime_dir.name} - Balanced Accuracy - {title}")
            plt.ylabel("Test Balanced Accuracy")
            plt.xlabel("Active Learning Cycle")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()

            filename_bacc = f"{prefix}_{dataset_dir.name}_{regime_dir.name}_bacc.png"
            plt.savefig(save_path / filename_bacc, dpi=300)
            plt.close()
            print(f"  Saved bacc plot to {save_path / filename_bacc}")

            # --- Save AUBC ---
            if aubc_results:
                aubc_df = pd.DataFrame(aubc_results)
                aubc_summary = aubc_df.groupby("Method")["AUBC"].agg(["mean", "std"]).reset_index()
                aubc_summary = aubc_summary.sort_values('mean', ascending=False)
                print("  AUBC Summary:")
                print(aubc_summary)

                aubc_filename = f"{prefix}_{dataset_dir.name}_{regime_dir.name}_aubc.csv"
                aubc_summary.to_csv(save_path / aubc_filename, index=False)

            processed.append(regime_dir)

    print("processed:")
    print(processed)

    if no_df_found:
        print("test_metrics.csv not found for:")
        for directory in no_df_found:
            print(directory)
    
    if not_finished:
        print("test_metrics.csv length < 10:")
        for directory in not_finished:
            print(directory)

    if met_exceptions:
        print("Got exceptions for:")
        for directory in met_exceptions:
            print(directory)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot simple analysis results.")
    parser.add_argument("--prefix", type=str, help="Prefix for the saved files.")
    parser.add_argument("--title", type=str, help="Title for the plots.")
    parser.add_argument("--no-std", action="store_true", help="Turn off standard deviation in plots.")
    parser.add_argument("--base-path", type=str, help="Base path for the experiments.")
    parser.add_argument("--query-methods", nargs="+", default=None,
                        help="Subset of query methods to plot, e.g. --query-methods BALD Vendi Bandit")
    parser.add_argument("--skip-before", type=str, default=None,
                        help="Only include runs at or after this timestamp (YYYY-MM-DD_HH-MM-SS-ffffff). Overrides SKIP_BEFORE.")
    parser.add_argument("--skip-after", type=str, default=None,
                        help="Only include runs at or before this timestamp (YYYY-MM-DD_HH-MM-SS-ffffff). Overrides SKIP_AFTER.")

    parser.add_argument("--note", type=str, default=None,
                        help="Optional note saved as note.txt in the output folder.")
    parser.add_argument("--results-path", type=str, default="/home/jing/Desktop/realistic-al/experiments/activelearning",
                        help="Base path for the experiments.")
    parser.add_argument("--save-path", type=str, default="/home/jing/Desktop/realistic-al/analysis/plots_simple",
                        help="Path to save the plots.")

    args = parser.parse_args()

    if args.skip_before is not None:
        SKIP_BEFORE = args.skip_before
    if args.skip_after is not None:
        SKIP_AFTER = args.skip_after

    main(args.prefix, args.title, show_std=not args.no_std)

