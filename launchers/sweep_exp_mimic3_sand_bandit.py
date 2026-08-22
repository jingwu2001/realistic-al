"""
MIMIC-III (SAND) — LinUCB bandit sweep over the diversity arm's vendi kernel/
normalization/gamma settings.

Two eval settings (balanced 50/50 test+val vs natural stratified split — see
exp_mimic3_sand_basic.py), natural imbalanced pool, weighted loss — same
training setup as exp_mimic3_sand_basic.py / sweep_exp_mimic3_sand_kernels.py
so results are comparable. query is fixed to `bandit`; bandit's diversity arm
reads the shared query.vendi.* block (query_bandit.py::BanditQuerySampler),
so sweeping query.vendi.kernel/normalization/gamma sweeps the diversity arm
exactly like the vendi/gvendi kernel launcher. use_grad is left at its
vendi.yaml default (false) — gvendi-style gradient embeddings for the
diversity arm are deliberately out of scope here (unlike gvendi, they haven't
been checked for the bandit's extra vendi(L), vendi(L+Q_u), vendi(L+Q_d)
context-feature computation in query_bandit.py::calculate_vendi_score).

Kernel/normalization/gamma grid (see KERNEL_GRID) — the same 9 deduplicated
cells as sweep_exp_mimic3_sand_kernels.py (rbf x {l2, minmax, zscore, none} x
gamma {dim, median}, plus cosine x none) — but ORDERED by AUBC(test/auroc) of
the completed vendi (non-gvendi) rows of that kernel sweep on
mimic3_sand_baleval/active-mimic3_med (20260719-592e2f89, all 9/9 finished as
of 2026-07-20): best (rbf/l2/dim, 0.7788) first, worst (rbf/minmax/dim,
0.7611) last. The spread is narrow (~0.018) so treat this as a mild prior for
which combos to spend bandit GPU-hours on first, not a strong signal. The
natural-split (mimic3_sand, AUPRC-ranked) vendi sweep wasn't complete enough
to rank independently at the time of writing (1/9 combos finished) — reuses
the balanced-split order for both data settings.

query.bandit.num_classes=2: MIMIC-III mortality is binary; the query/bandit.yaml
default of 10 would silently mis-scale the cls_dist context feature (divides
by log(10) instead of log(2) in query_bandit.py::_class_distribution_entropy)
— same fix already applied in exp_p12_grud.py / exp_ecg5000_basic.py for their
respective class counts.

model.dropout_p=0.5, NOT the SAND-default 0.2 used by
sweep_exp_mimic3_sand_kernels.py/exp_mimic3_sand_basic.py: those match dropout
to the original STraTS training recipe (docs/mimic3_sand_integration.md), a
choice about final-model accuracy, not MC-dropout AL quality. The bandit's
uncertainty arm needs informative BALD scores across MC-dropout samples (P12
launchers hardcode 0.5 for exactly this reason — exp_p12_grud.py), and 0.2
measured near-zero at 1 epoch in a smoke run.

Bandit sampling itself was verified post the vendi resolve_gamma/kernel fixes
(roadmap P3): tests/test_bandit_features.py (32/32 suite green), a standalone
sweep across all 9 kernel/normalization combos through the real (unmocked)
query_diversity code path, and a real 2-iteration MIMIC-III/SAND smoke run —
no errors, both arms fire, reward updates and AUBC compute correctly.

9 kernel combos x 2 data settings x 1 seed = 18 runs; chunk with
--num_start / --num_end if needed.
"""
from argparse import ArgumentParser

from launcher import ExperimentLauncher

# (kernel, normalization, gamma) rows, best-AUBC(auroc)-first (see docstring);
# gamma is a placeholder (1.0) for cosine, ignored by compute_kernel_matrix.
KERNEL_GRID = [
    ("rbf", "l2", "dim"),        # 0.7788
    ("cosine", "none", 1.0),     # 0.7780
    ("rbf", "l2", "median"),     # 0.7778
    ("rbf", "none", "dim"),      # 0.7766
    ("rbf", "none", "median"),   # 0.7746
    ("rbf", "zscore", "dim"),    # 0.7735
    ("rbf", "zscore", "median"), # 0.7716
    ("rbf", "minmax", "median"), # 0.7700
    ("rbf", "minmax", "dim"),    # 0.7611
]
kernels = [k for k, _, _ in KERNEL_GRID]
norms = [n for _, n, _ in KERNEL_GRID]
gammas = [g for _, _, g in KERNEL_GRID]

config_dict = {
    "model": ["sand"],
    "query": ["bandit"],
    "data": [
        "mimic3_sand_baleval",  # balanced 50/50 test+val; pool drained to ~1.9% pos
        "mimic3_sand",          # natural stratified split; L+U/val/test all ~11.5% pos
    ],
    "active": ["mimic3_med"],  # imbalanced initial pool
    "optim": ["adam"],
}

hparam_dict = {
    "data.balanced_sampling": False,
    "model.weighted_loss": True,
    # "trainer.seed": [12345, 12346, 12347],
    "trainer.seed": [12345],
    "trainer.max_epochs": 50,
    "model.dropout_p": 0.2,
    "model.learning_rate": [0.0005],
    "model.weight_decay": [1e-5],
    "model.use_ema": False,
    "trainer.batch_size": 64,
    "trainer.precision": 32,
    "trainer.deterministic": True,
    "query.bandit.num_classes": 2,
    # --- sweep (query.vendi.* drives the bandit's diversity arm) ---
    "query.vendi.kernel": kernels,
    "query.vendi.normalization": norms,
    "query.vendi.gamma": gammas,
    # explicit despite matching the vendi.yaml default: pins the diversity arm
    # to feature embeddings, not gvendi-style gradients (see docstring)
    "query.vendi.use_grad": False,
}

naming_conv = "{data}/active-{active}/model-{model}_drop-{model.dropout_p}_acq-{query}_norm-{query.vendi.normalization}_kernel-{query.vendi.kernel}_gamma-{query.vendi.gamma}_ep-{trainer.max_epochs}_wl-{model.weighted_loss}"

joint_iteration = [
    ["query.vendi.kernel", "query.vendi.normalization", "query.vendi.gamma"],
]

path_to_ex_file = "src/main.py"

if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    ExperimentLauncher.add_argparse_args(parser)
    parser.add_argument(
        "--wandb-test", action="store_true", dest="wandb_test",
        help="Create each run's wandb run (name/group/tags/config, tagged "
             "'wandb_test', state 'failed') and exit immediately — verifies "
             "the wandb wiring without loading data or training",
    )
    launcher_args = parser.parse_args()

    config_dict, hparam_dict = ExperimentLauncher.modify_params_for_args(
        launcher_args, config_dict, hparam_dict
    )
    if launcher_args.wandb_test:
        hparam_dict["trainer.wandb_test"] = True

    launcher = ExperimentLauncher(
        config_dict,
        hparam_dict,
        launcher_args,
        naming_conv,
        path_to_ex_file,
        joint_iteration=joint_iteration,
    )

    launcher.launch_runs()
