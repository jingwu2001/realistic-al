# Analysis Guide

This guide explains how to use your experiment results to draw plots and calculate the Area Under the Budget Curve (AUBC).

The main script for analysis is `plot.py`. This script loads your experiment results, generates performance plots, and calculates the AUBC for each method.

## 1. Prerequisites

Ensure you have the necessary Python packages installed (these are standard in the project environment):
- `matplotlib`
- `seaborn`
- `pandas`
- `numpy`

## 2. Setting up `plot.py`

You need to modify `analysis/plot.py` to match your local setup and include all the methods you want to analyze.

### 2.1. Set the Results Path

Locate the `RESULTSPATH` variable in `plot.py` (around line 25). Change it to the absolute path where your `activelearning` folder is located.

For example, if your results are in `/home/jing/Desktop/realistic-al/experiments/activelearning`, set:

```python
RESULTSPATH = "/home/jing/Desktop/realistic-al/experiments/activelearning"
```

The script expects the following directory structure under `RESULTSPATH`:
`DATASET_NAME/LABEL_REGIME/EXPERIMENT_NAME/SEED/...`

### 2.2. Add New Query Methods

You want to include the following methods: `badge`, `entropy`, `kcentergreedy`, `random`, `vendi`, `bald`, `bandit`.

Update the `QUERYMETHODS` dictionary in `plot.py` (around line 66) to include `vendi` and `bandit`:

```python
QUERYMETHODS = {
    "bald": "BALD",
    "kcentergreedy": "Core-Set",
    "entropy": "Entropy",
    "random": "Random",
    "badge": "BADGE",
    "batchbald": "BatchBALD",
    "vendi": "Vendi",      # Add this
    "bandit": "Bandit",    # Add this
}
```

*Note: The keys (e.g., "bandit") must match the method name found in your experiment folder names (e.g., `...acq-bandit...`).*

### 2.3. Update Color Palette

Update the `PALETTE` dictionary (around line 75) to assign colors to the new methods:

```python
PALETTE = {
    "BALD": "tab:blue",
    "Core-Set": "tab:green",
    "Entropy": "tab:orange",
    "BADGE": "tab:purple",
    "Random": "tab:red",
    "BatchBALD": "tab:cyan",
    "Vendi": "tab:brown",  # Add a color for Vendi
    "Bandit": "tab:pink",  # Add a color for Bandit
}
```

## 3. Generating Plots and AUBC

Once `plot.py` is configured, run it from the `analysis` directory:

```bash
python plot.py
```

### What happens when you run this?

1.  **Data Loading**: The script scans `RESULTSPATH` for experiment folders matching the datasets defined in `DATASETS`. It consolidates metrics from `stored.npz` or `test_metrics.csv` into a DataFrame.
2.  **AUBC Calculation**:
    *   The script automatically calculates the Area Under the Budget Curve (AUBC) for each experiment.
    *   It looks at the test accuracy (`test/acc` or similar) over the learning cycles.
    *   Code location: `get_aubc` function called within the loop (lines 518-528).
3.  **Output**:
    *   **Plots**: Saved in the `plots/` directory (created automatically).
    *   **AUBC Data**: Saved as `plots/aubc.csv`. You can open this CSV to see the calculated AUBC values for all methods.

## 4. Customzing the Plots

### Filtering Experiments
You can control which experiments are plotted by modifying the `plot_label_settings` and `plot_value_settings_dict` in the `__main__` block of `plot.py`.

### Baselines
If you haven't run the "full" training (training on the 100% dataset) or don't have those results available, you might want to disable the upper bound line in the plots.
In `plot_settings` (around line 470), set `"upper_bound": [False]`.

```python
plot_settings = {"sharey": [True, False], "upper_bound": [False]}
```

## 5. Troubleshooting

-   **"No Experiments Performed"**: Use the `list_dir` tool or check manually to ensure your `RESULTSPATH` is correct and matches the structure expected by `dataframe.py`.
-   **Method not showing**: Ensure the experiment folder name contains `acq-{method_name}` matching the key in `QUERYMETHODS`.