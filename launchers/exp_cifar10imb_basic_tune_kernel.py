from argparse import ArgumentParser

from launcher import ExperimentLauncher

config_dict = {
    "model": "resnet",
    "query": [
        "bandit",
        "vendi",
        # "random",
        # "entropy",
        # "kcentergreedy",
        # "bald",
        # "badge",
    ],
    "data": ["cifar10_imb"],
    "active": [
        "cifar10_low",
        "cifar10_med",
        # "cifar10_high",
    ],  # did not run! "standard_250", "cifar10_low_data"
    "optim": ["sgd_cosine"],
    # "query.vendi.kernel": ["rbf", "rbf", "rbf", "cosine"],
    # "query.vendi.gamma": [0.1, 1.0, 10.0, None],
    "query.vendi.kernel": ["rbf"],
    "query.vendi.gamma": [1/1024],
    "query.vendi.normalization": ["minmax"],
}

hparam_dict = {
    "model.weighted_loss": True,
    # "data.val_size": [50 * 5, 250 * 5, None],
    "data.val_size": [50 * 5, 250 * 5],
    "trainer.seed": [12345, 12346, 12347],
    "trainer.max_epochs": 200,
    "model.weight_decay": 5e-3,
    "model.dropout_p": [0.5] * len(config_dict['query']),
    "model.learning_rate": [0.1],
    "model.use_ema": False,
    "data.transform_train": [
        "cifar_randaugmentMC",
    ],
    "trainer.batch_size": 1024,
    "trainer.precision": 16,
    "trainer.deterministic": True,
}

naming_conv = "{data}/active-{active}/basic_model-{model}_drop-{model.dropout_p}_aug-{data.transform_train}_acq-{query}_norm-{query.vendi.normalization}_kernel-{query.vendi.kernel}_gamma-{query.vendi.gamma}_ep-{trainer.max_epochs}__wloss-{model.weighted_loss}"

joint_iteration = [
    ["model.dropout_p", "query"],
    # ["query.vendi.kernel", "query.vendi.gamma"],
    ["data.val_size", "active"]
]

path_to_ex_file = "src/main.py"

if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
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
