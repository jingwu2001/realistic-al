"""
MIMIC-III (SAND) — vendi kernel/normalization/use_grad sweep, resumed.

Identical grid and training setup to sweep_exp_mimic3_sand_kernels.py; the only
difference is that the (data, query, kernel, normalization, gamma, seed) cells
listed in DONE below are skipped.

The 2026-07-30 attempt of that sweep (commit 89844322) completed 3/36 runs
before the shared GPU filled up with another user's jobs; the remaining 33
runs died in the query step with CUDA OOM (or the cuSOLVER
`cusolverDnCreate` failure that OOM surfaces as). Nothing is wrong with those
configs, so this launcher simply re-runs everything except the 3 that finished.

Note the 36 finished runs of the same grid from 2026-07-19..21 do NOT count as
done: they predate the patient-level train/val/test split (68f26d92,
2026-07-28) and are not comparable to the current results.
"""
from argparse import ArgumentParser
from typing import Any, Dict

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

# (data, query, kernel, normalization, gamma, seed) cells that already have a
# finished run under the current split — completed 2026-07-30, commit 89844322:
#   experiments/activelearning/mimic3_sand_baleval/active-mimic3_med/20260730-89844322/
#   wandb: k8u4zmno, zvgc779o, mag63jft
DONE = {
    ("mimic3_sand_baleval", "gvendi", "cosine", "none", 1.0, 12345),
    ("mimic3_sand_baleval", "gvendi", "rbf", "none", "dim", 12345),
    ("mimic3_sand_baleval", "gvendi", "rbf", "none", "median", 12345),
}


class ResumeExperimentLauncher(ExperimentLauncher):
    def skip_config(self, config_settings: Dict[str, Any]) -> bool:
        if super().skip_config(config_settings):
            return True
        cell = (
            config_settings.get("data"),
            config_settings.get("query"),
            config_settings.get("query.vendi.kernel"),
            config_settings.get("query.vendi.normalization"),
            config_settings.get("query.vendi.gamma"),
            config_settings.get("trainer.seed"),
        )
        return cell in DONE


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

    launcher = ResumeExperimentLauncher(
        config_dict,
        hparam_dict,
        launcher_args,
        naming_conv,
        path_to_ex_file,
        joint_iteration=joint_iteration,
    )

    launcher.launch_runs()
