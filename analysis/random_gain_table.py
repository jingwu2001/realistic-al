"""
One row per random-query setting (model x variables x window x demographics x
cohort x budget): first/last-round test metric, the gain between them, and the
gain over the last three rounds, from every finished seed. Ranks the settings by
learning-curve headroom.

Usage: python analysis/random_gain_table.py [out.csv]   (default analysis/cache/random_gain_table.csv)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from report_bench_headroom import collect

INPUT = {  # data config -> (variables, window)
    "mimic3_pheno": (129, "24h"), "mimic3_pheno_h48": (129, "48h"), "mimic3_pheno_h168": (129, "168h"), "mimic3_pheno_full": (129, "full"),
    "mimic3_pheno_bench_h24": (17, "24h"), "mimic3_pheno_bench": (17, "48h"),
    "mimic3_pheno_bench_h168": (17, "168h"), "mimic3_pheno_bench_full": (17, "full"),
    "mimic3_ihm_bench": (17, "48h"),
}
df = collect()
df = df[df.model.isin(["lstm", "sand"]) & df.finished]
rows = []
for (data, model, demo, wave, sso, start), g in df.groupby(["data", "model", "demo", "wave", "single_stay", "start"]):
    curve = g.pivot_table(index="seed", columns="labels", values="metric")
    if curve.shape[1] < 10:
        continue
    first, last = curve.iloc[:, 0], curve.iloc[:, -1]
    gain, tail = last - first, curve.iloc[:, -1] - curve.iloc[:, -4]
    v, w = INPUT[data]
    rows.append(dict(
        task="IHM" if data == "mimic3_ihm_bench" else "pheno25", model=model, vars=v, window=w,
        demo=64 if (demo and model == "sand") else 0, cohort="single" if sso else "all",
        budget=f"{start}->{start * 10}", seeds=len(curve),
        first=first.mean(), last=last.mean(), gain=gain.mean(), gain_sd=gain.std(),
        last3=tail.mean(), last3_sd=tail.std(),
    ))
t = pd.DataFrame(rows).sort_values(["task", "budget", "gain"], ascending=[True, True, False])
pd.set_option("display.width", 200)
print(t.round(3).to_string(index=False))
t.to_csv(sys.argv[1] if len(sys.argv) > 1 else "analysis/cache/random_gain_table.csv", index=False)
