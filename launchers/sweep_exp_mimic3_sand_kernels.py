"""
MIMIC-III (SAND) — vendi kernel/normalization/use_grad sweep.

Two eval settings (balanced 50/50 test+val vs natural stratified split — see
exp_mimic3_sand_basic.py), natural imbalanced pool, weighted loss — same
training setup as exp_mimic3_sand_basic.py so results are comparable.

Grid: use_grad true/false (query=gvendi / query=vendi, both read the shared
query.vendi.* block) x the deduplicated kernel/normalization grid
(see KERNEL_GRID):
  rbf x {l2, minmax, zscore, none} x gamma {dim, median}
  + cosine x none                                       (9 combos per method)
cosine keeps only the `none` cell (it is invariant to per-sample scaling and
its feature-wise norms were pruned 2026-07-16); the linear kernel was dropped
entirely (l2+linear == cosine). rbf sweeps the two bandwidth heuristics
("dim" = 1/(2 D), "median" = 1/(2 median^2) — see resolve_gamma).

2 queries x 9 kernel combos x 2 data settings x 3 seeds = 108 runs; chunk
with --num_start / --num_end if needed.
"""
from argparse import ArgumentParser

from launcher import ExperimentLauncher

# aligned (kernel, normalization, gamma) rows, zipped via joint_iteration;
# gamma is ignored by cosine (placeholder 1.0)
KERNEL_GRID = (
    [("cosine", "none", 1.0)]
    +
    [
        ("rbf", norm, gamma)
        for norm in ("l2", "minmax", "zscore", "none")
        for gamma in ("dim", "median")
    ]
)
kernels = [k for k, _, _ in KERNEL_GRID]
norms = [n for _, n, _ in KERNEL_GRID]
gammas = [g for _, _, g in KERNEL_GRID]

config_dict = {
    "model": ["sand"],
    "query": [
        "gvendi",
        "vendi",
    ],
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
    # --- sweep (query.vendi.* drives both vendi and gvendi) ---
    "query.vendi.kernel": kernels,
    "query.vendi.normalization": norms,
    "query.vendi.gamma": gammas,
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
