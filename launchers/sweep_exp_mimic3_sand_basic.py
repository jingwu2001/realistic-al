"""
MIMIC-III (SAND) — baseline query methods.

Two eval settings, both with the natural imbalanced pool:
  mimic3_sand_baleval — balanced 50/50 test+val (drains the pool to ~1.9% pos)
  mimic3_sand         — natural stratified split, nothing discarded:
                        L+U pool, val, and test all ~11.5% positive
Weighted loss compensates for the pool imbalance during training (STraTS
convention for this task).

Kernel/normalization/use_grad sweeps for vendi live in
exp_mimic3_sand_kernels.py.
"""
from argparse import ArgumentParser

from launcher import ExperimentLauncher

config_dict = {
    "model": ["sand"],
    "query": [
        "random",
        "entropy",
        "kcentergreedy",
        "bald",
        "badge",
        "variationratios",
        # batchbald: heavy but feasible on the binary task (100 greedy steps
        # over the ~30k pool in float64, est. hours per run) — prune the rows
        # with --num_start/--num_end or ++active.m if it proves too slow
        # "batchbald",
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
    "trainer.seed": [12345, 12346, 12347],
    "trainer.max_epochs": 50,
    "model.dropout_p": 0.2,
    "model.learning_rate": [0.0005],
    "model.weight_decay": [1e-5],
    "model.use_ema": False,
    "trainer.batch_size": 64,
    "trainer.precision": 32,
    "trainer.deterministic": True,
}

naming_conv = "{data}/active-{active}/model-{model}_drop-{model.dropout_p}_acq-{query}_ep-{trainer.max_epochs}_wl-{model.weighted_loss}"

joint_iteration = None

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
