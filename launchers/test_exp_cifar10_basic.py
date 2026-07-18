from argparse import ArgumentParser

from launcher import ExperimentLauncher

config_dict = {
    "model": "resnet",
    "query": [
        "vendi",
        "gvendi",
        "random",
        "entropy",
        "kcentergreedy",
        "bald",
        "badge",
        "batchbald",
        "variationratios",
    ],
    "data": ["cifar10"],
    "active": [
        "cifar10_low",
    ],
    "optim": ["sgd_cosine"],
}

hparam_dict = {
    "data.val_size": [250],
    "trainer.seed": [12345],
    "trainer.max_epochs": 50,
    "model.dropout_p": [0.5] * len(config_dict['query']),
    "model.learning_rate": [0.1],
    "model.use_ema": False,
    "data.transform_train": [
        "cifar_randaugmentMC",
    ],
    "trainer.precision": 16,
    "trainer.batch_size": 1024,
    "trainer.deterministic": True,
    "active.num_iter": 3,
}
naming_conv = (
    "{data}/active-{active}/basic_model-{model}_drop-{model.dropout_p}_aug-{data.transform_train}_acq-{query}_ep-{trainer.max_epochs}"
)

joint_iteration = [
    ["active", "data.val_size"],
    ["query", "model.dropout_p"],
    # ["query.vendi.kernel", "query.vendi.gamma"],
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
