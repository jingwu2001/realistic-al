"""Generate docs/sweep_run_tables.md from the actual launcher grids.

Run from the repo root:
    python launchers/gen_sweep_run_tables.py
"""
import importlib
import os
import sys
from itertools import product

_here = os.path.dirname(os.path.abspath(__file__))
_repo = os.path.dirname(_here)
sys.path.insert(0, _here)

EMB = {"vendi": "feat", "gvendi": "grad"}


def parse_rows(mod, single_seed=True, groups_first=True):
    """Replicate the launcher's run ordering. groups_first=True mirrors
    QuerySweepLauncher (joint groups are the slowest axes); groups_first=False
    mirrors the base ExperimentLauncher, where each joint group effectively
    sits at its first member key's dict position (kernel rows innermost).
    Returns full run dicts in launch order (seeds forced to one)."""
    config = {k: (v if isinstance(v, list) else [v]) for k, v in mod.config_dict.items()}
    hparam = {k: (v if isinstance(v, list) else [v]) for k, v in mod.hparam_dict.items()}
    if single_seed:
        hparam["trainer.seed"] = hparam["trainer.seed"][:1]
    joint = {**config, **hparam}
    groups = getattr(mod, "joint_iteration", None) or []
    if groups and not isinstance(groups[0], (list, tuple)):
        groups = [groups]

    grouped = {k: tuple(g) for g in groups for k in g}
    group_axes = {tuple(g): (tuple(g), list(zip(*[joint[k] for k in g]))) for g in groups}
    axes, emitted = [], set()
    if groups_first:
        axes = list(group_axes.values())
        emitted = set(group_axes)
        for k, v in joint.items():
            if k not in grouped:
                axes.append(((k,), [(x,) for x in v]))
    else:
        for k, v in joint.items():
            if k in grouped:
                g = grouped[k]
                if g not in emitted:
                    axes.append(group_axes[g])
                    emitted.add(g)
            else:
                axes.append(((k,), [(x,) for x in v]))

    rows, seen = [], set()
    for combo in product(*[vals for _, vals in axes]):
        d = {}
        for (keys, _), vals in zip(axes, combo):
            d.update(dict(zip(keys, vals)))
        key = tuple(sorted((k, str(v)) for k, v in d.items()))
        if key not in seen:
            seen.add(key)
            rows.append(d)
    return rows


def fmt_row(i, d, setting_cols):
    q = d["query"]
    if q in EMB:
        emb = EMB[q]
        kernel = d["query.vendi.kernel"]
        norm = d["query.vendi.normalization"]
        gamma = d["query.vendi.gamma"] if kernel == "rbf" else "-"
    else:
        emb, kernel, norm, gamma = "-", "-", "-", "-"
    cells = [str(i)] + [str(d.get(c, "-")) for c in setting_cols] + [q, emb, kernel, norm, str(gamma)]
    return "| " + " | ".join(cells) + " |"


def table(mod_names, setting_cols, setting_headers):
    lines = []
    header = ["#"] + setting_headers + ["query", "emb", "kernel", "norm", "gamma"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    i = 0
    for mod_name, groups_first in mod_names:
        mod = importlib.import_module(mod_name)
        for d in parse_rows(mod, groups_first=groups_first):
            i += 1
            lines.append(fmt_row(i, d, setting_cols))
    return i, "\n".join(lines)


out = []
out.append("""# Sweep Run Tables (one seed)

One row per run, in **launch order for a single seed** (`trainer.seed=12345`).
With the default 3 seeds every row runs 3x: in the `sweep_exp_*_query` launchers
the 3 seeds occupy consecutive launcher indices (row r -> launches 3r-2..3r);
in the MIMIC launchers seeds nest as outer blocks instead — when slicing with
`--num_start/--num_end`, confirm indices with `--debug` first. Generated
2026-07-16 from the launcher grids (verified against `--debug` output) —
regenerate with `python launchers/gen_sweep_run_tables.py` after editing a
launcher.

Column notes: **emb** `feat` = vendi (feature embeddings), `grad` = gvendi
(`vendi.use_grad=true`, BADGE-style last-layer gradients). **gamma** applies to
rbf only (`dim` = 1/(2D), `median` = 1/(2·median²)). Baselines show `-` for the
kernel columns (inert placeholders).
""")

sections = []

n, t = table([("sweep_exp_cifar10_query", True)], ["data", "active"], ["data", "active"])
sections.append(("CIFAR-10 (+imb)", "`sweep_exp_cifar10_query.py`", n))
out.append(f"## CIFAR-10 / CIFAR-10-imb — {n} runs\n\nvendi rows: ~1.2 h (low) / ~4.8 h (med) / ~0.75 h (imb-low) / ~1.5-2 h (imb-med); baselines ~35-50 min (batchbald: est. 1-1.5 h low / 3-6 h med).\n\n" + t)

n, t = table([("sweep_exp_cifar100_query", True)], ["active"], ["active"])
sections.append(("CIFAR-100", "`sweep_exp_cifar100_query.py`", n))
out.append(f"## CIFAR-100 — {n} runs\n\nvendi rows: long query phases at L>=500 (~7 min/round full pool); batchbald-med/high are the heaviest rows (hours per round).\n\n" + t)

n, t = table([("sweep_exp_p12_transformer_query", True)], ["data"], ["eval split"])
sections.append(("P12", "`sweep_exp_p12_transformer_query.py`", n))
out.append(f"## P12 (transformer) — {n} runs\n\nCheap: ~7.6k pool, binary task; vendi rows a few minutes of query per round.\n\n" + t)

n, t = table([("sweep_exp_mimic3_sand_basic", False), ("sweep_exp_mimic3_sand_kernels", False)], ["data"], ["eval split"])
sections.append(("MIMIC-III", "`sweep_exp_mimic3_sand_basic.py` + `sweep_exp_mimic3_sand_kernels.py`", n))
out.append(f"## MIMIC-III (SAND) — {n} runs\n\nTwo launchers: baselines (`_basic`, rows 1-14) then vendi grid (`_kernels`, rows 15-70); run numbers restart per launcher for `--num_start/--num_end` (basic: 1-14, kernels: 1-56 at one seed). batchbald: est. hours/run (float64 greedy over ~30k pool).\n\n" + t)

summary = ["| dataset | launcher(s) | runs/seed |", "|---|---|---|"]
summary += [f"| {name} | {files} | {n} |" for name, files, n in sections]
summary.append(f"| **total** | | **{sum(n for _, _, n in sections)}** |")
out.insert(1, "\n".join(summary))

with open(os.path.join(_repo, "docs", "sweep_run_tables.md"), "w") as f:
    f.write("\n\n".join(out) + "\n")
print("written docs/sweep_run_tables.md")
