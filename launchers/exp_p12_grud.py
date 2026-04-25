"""
Launcher for active learning experiments on the P12 dataset with GRU-D.

Usage:
    cd launchers
    python exp_p12_grud.py              # dry-run (prints commands)
    python exp_p12_grud.py --run local  # run locally

Prerequisites:
    - Run  `bash Raindrop/prepare_for_al.sh`  to produce datasets/p12/*.npy
    - Set  DATA_ROOT  and  EXPERIMENT_ROOT  environment variables
"""

import argparse

from launcher import ExperimentLauncher

config_dict = {
    "model":  ["gru_d"],
    "query":  [
        "bandit",
        "random",
        "entropy",
        "bald",
        "badge",
        "kcentergreedy",
        "vendi",
    ],
    "data":   ["p12"],
    "active": ["p12_low", "p12_med"],
    "optim":  ["adam"],
}

# Use dropout_p=0.5 for all query methods so training stochasticity is matched
# across acquisition strategies.
_QUERY_ORDER = config_dict["query"]
_DROPOUT_BY_QUERY = {
    "bandit": 0.5, "bald": 0.5,
    "random": 0.5, "entropy": 0.5, "badge": 0.5, "kcentergreedy": 0.5, "vendi": 0.5,
}

hparam_dict = {
    "trainer.seed":          [12345, 12346, 12347],
    "trainer.max_epochs":    50,
    "model.dropout_p":       [_DROPOUT_BY_QUERY[q] for q in _QUERY_ORDER],
    "model.learning_rate":   [0.001],
    "trainer.precision":     32,
    "trainer.batch_size":    64,
    "trainer.deterministic": True,
    "model.weighted_loss":   False,
    "query.vendi.kernel":    "rbf",
    "query.vendi.gamma":     1.0,
    "query.vendi.normalization": "minmax",
    "query.bandit.num_classes": 2,
}

naming_conv = (
    "{data}/active-{active}/basic_model-{model}_drop-{model.dropout_p}_acq-{query}_ep-{trainer.max_epochs}"
)

joint_iteration = [
    ["model.dropout_p", "query"],
]

path_to_ex_file = "src/main.py"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    ExperimentLauncher.add_argparse_args(parser)
    launcher_args = parser.parse_args()

    config_dict_, hparam_dict_ = ExperimentLauncher.modify_params_for_args(
        launcher_args, config_dict, hparam_dict
    )

    launcher = ExperimentLauncher(
        config_dict_,
        hparam_dict_,
        launcher_args,
        naming_conv,
        path_to_ex_file,
        joint_iteration=joint_iteration,
    )

    launcher.launch_runs()
