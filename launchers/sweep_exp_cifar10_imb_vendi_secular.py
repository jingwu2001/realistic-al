"""
CIFAR-10 (long-tail imb) vendi q=1 — secular fast-path vs exact eigendecomposition.

A short two-round active-learning run whose ONLY swept axis is
``query.vendi.secular``:

    secular=True  -> _get_vendi dispatches to vendi_from_features_secular
                     (vendi-approx C arrowhead-secular solver, CPU)
    secular=False -> the production vendi_from_features (dense torch eigvalsh)

Everything else is fixed and the run is fully deterministic (same seed, same
kernel/normalization/gamma), so the two settings should acquire the SAME batch
each round — the secular path is a speed optimisation that reproduces the exact
q=1 Vendi scores to ~machine precision. This launcher exists to confirm that
inside a real AL loop (not just the offline demo in
vendi-approx/src/demo_secular_acq.py) and to compare eig_time_s.

Both runs log to a SEPARATE wandb project, ``vendi-approx`` (via
++trainer.wandb_project), to keep this methodological check out of the main
realistic-al project.

Query settings (secular constraints): q=1 and a unit-diagonal kernel (rbf here);
gamma="dim" (1/2D) is scale-adaptive so the kernel does not saturate under minmax.

Grid = query.vendi.secular {True, False} x 1 (data, active) x 1 seed = 2 runs,
each doing active.num_iter=2 acquisition rounds.
"""
from argparse import ArgumentParser

from launcher import ExperimentLauncher

# --- config-group selections ------------------------------------------------
config_dict = {
    "model": "resnet",
    "query": ["vendi"],          # q=1 Vendi -> _get_vendi, which honors secular
    "data": ["cifar10_imb"],     # long-tail CIFAR-10
    "active": ["cifar10_med"],   # 250 start / 250 acq_size; num_iter overridden below
    "optim": ["sgd_cosine"],
}

# --- hydra ++overrides ------------------------------------------------------
hparam_dict = {
    # log this comparison to a different wandb project
    "trainer.wandb_project": "vendi-approx",
    # two acquisition rounds only (cifar10_med defaults to 10)
    "active.num_iter": 2,

    # data / training (matches the cifar10_imb vendi sweep settings)
    "data.val_size": [1250],
    "model.weighted_loss": [True],
    "trainer.seed": [12345],
    "trainer.max_epochs": 200,
    "model.dropout_p": [0.5],
    "model.learning_rate": [0.1],
    "model.use_ema": False,
    "data.transform_train": ["cifar_randaugmentMC"],
    "trainer.precision": 16,
    "trainer.batch_size": 1024,
    "trainer.deterministic": True,

    # --- the axis of interest ---
    "query.vendi.secular": [False],
    # secular requires q=1 (config default) + a unit-diagonal kernel; pin the
    # kernel/normalization/gamma so both runs are identical apart from `secular`.
    "query.vendi.kernel": ["rbf"],
    "query.vendi.normalization": ["minmax"],
    "query.vendi.gamma": [1.0],
}

naming_conv = (
    "{data}/active-{active}/"
    "basic_model-{model}_acq-{query}_secular-{query.vendi.secular}"
    "_norm-{query.vendi.normalization}_kernel-{query.vendi.kernel}"
    "_gamma-{query.vendi.gamma}_ep-{trainer.max_epochs}"
)

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
    )

    launcher.launch_runs()
