import argparse

from launcher import ExperimentLauncher

config_dict = {
    "model": ["inception_time"],
    "query": [
        "bandit",
        "vendi",
        # "random",
        # "entropy",
        # "kcentergreedy",
        # "bald",
        # "badge",
    ],
    "data": ["ecg5000"],
    "active": [
        "ecg5000_low",
        # "ecg5000_med",
        "ecg5000_10",
        # "ecg5000_high", # not enough samples
    ],
    "optim": ["adam"],
}

hparam_dict = {
    "data.val_size": [100] * len(config_dict['active']),
    "trainer.seed": [12345, 12346, 12347],
    "trainer.max_epochs": 17,
    "model.dropout_p": [0.5] * len(config_dict['query']),
    "model.learning_rate": [0.0001442986898059786],
    "trainer.precision": 32,
    "trainer.batch_size": 8,
    "trainer.deterministic": True,
    "query.vendi.kernel": "rbf",
    "query.vendi.gamma": 1.0,
    "query.vendi.normalization": "minmax",
}
naming_conv = (
    "{data}/active-{active}/basic_model-{model}_acq-{query}_ep-{trainer.max_epochs}"
)

joint_iteration = [
    # No complex joint variations yet
    ["model.dropout_p", "query"],
    ["data.val_size", "active"]
]

path_to_ex_file = "src/main.py"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    ExperimentLauncher.add_argparse_args(parser)
    launcher_args = parser.parse_args()

    config_dict, hparam_dict = ExperimentLauncher.modify_params_for_args(
        launcher_args, config_dict, hparam_dict
    )

    launcher = ExperimentLauncher(
        config_dict,
        hparam_dict,
        launcher_args,
        naming_conv,
        path_to_ex_file,
        joint_iteration=joint_iteration,
    )

    launcher.launch_runs()
