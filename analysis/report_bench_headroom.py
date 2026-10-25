"""
Learning-curve headroom of the random-query runs on the mimic3-benchmarks input
(launchers/sweep_exp_mimic3_bench_{pheno,ihm}_lstm_random.py).

For every (data, cohort, budget) group: the per-round test metric for each seed,
the seed mean +- sd, the gain from the first to the last round, the gain over
the last three rounds (is the curve still rising?), and AUBC.

Phenotyping: test/macro_auroc (+ macro_auprc); mortality: test/auroc (+ auprc).

Usage:
  python analysis/report_bench_headroom.py [--out analysis/cache/bench_headroom.md]
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path("experiments/activelearning")
DATA = {  # data config -> (metric, secondary metric)
    "mimic3_ihm_bench": ("test/auroc", "test/auprc"),
    "mimic3_pheno_bench": ("test/macro_auroc", "test/macro_auprc"),
    "mimic3_pheno_bench_full": ("test/macro_auroc", "test/macro_auprc"),
}
RUN_RE = re.compile(r"model-lstm_demo-False_acq-random_sso-(True|False)_start-(\d+)_step-(\d+)_ep-50")


def collect() -> pd.DataFrame:
    rows = []
    for data, (metric, metric2) in DATA.items():
        for run in sorted((ROOT / data / "active-mimic3_med").glob("*/model-lstm*/*/")):
            m = RUN_RE.fullmatch(run.parent.name)
            if not m or not (run / "test_metrics.csv").exists():
                continue
            seed = yaml.safe_load(open(run / "loop-0" / "hparams.yaml"))["trainer"]["seed"]
            tm = pd.read_csv(run / "test_metrics.csv", index_col=0)
            start, step = int(m.group(2)), int(m.group(3))
            for rnd, r in tm.iterrows():
                rows.append(dict(data=data, single_stay=m.group(1) == "True", start=start,
                                 seed=seed, round=int(rnd), labels=start + step * int(rnd),
                                 metric=r[metric], metric2=r[metric2],
                                 finished=(run / "aubc.json").exists(), run=str(run)))
    return pd.DataFrame(rows)


def report(df: pd.DataFrame) -> str:
    out = []
    for (data, sso, start), g in df.groupby(["data", "single_stay", "start"]):
        metric, metric2 = DATA[data]
        cohort = "single-stay" if sso else "all stays"
        curve = g.pivot_table(index="seed", columns="labels", values="metric")
        curve2 = g.pivot_table(index="seed", columns="labels", values="metric2")
        n_done = g.groupby("seed").finished.first().sum()
        out.append(f"\n### {data} | {cohort} | start {start}  ({n_done}/{curve.shape[0]} seeds finished)\n")
        out.append(f"{metric} per round (labels):\n")
        tab = curve.copy()
        tab.loc["mean"], tab.loc["sd"] = curve.mean(), curve.std()
        out.append(tab.round(3).to_string())
        first, last = curve.iloc[:, 0], curve.iloc[:, -1]
        tail = curve.iloc[:, -1] - curve.iloc[:, max(0, curve.shape[1] - 4)]
        # trapezoid AUBC over the labelled-set axis, normalised to the budget span
        x = np.array(curve.columns, dtype=float)
        aubc = curve.apply(lambda r: np.trapz(r.values, x) / (x[-1] - x[0]) if len(x) > 1 else np.nan, axis=1)
        out.append(
            f"\n\ngain first->last: {(last - first).mean():+.3f} +- {(last - first).std():.3f} | "
            f"gain over last 3 rounds: {tail.mean():+.3f} +- {tail.std():.3f} | "
            f"AUBC {aubc.mean():.3f} +- {aubc.std():.3f} | "
            f"{metric2} last round {curve2.iloc[:, -1].mean():.3f} +- {curve2.iloc[:, -1].std():.3f}\n"
        )
    return "\n".join(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("analysis/cache/bench_headroom.md"))
    args = p.parse_args()
    df = collect()
    if df.empty:
        raise SystemExit("no runs found")
    text = report(df)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(text)
