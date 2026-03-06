from argparse import ArgumentParser
from launcher import ExperimentLauncher

config_dict = {
    "model": "resnet",
    "query": ["random"],
    "data": ["cifar10"],
    "active": ["cifar10_low"],
    "optim": ["sgd_cosine"],
}

hparam_dict = {
    "data.val_size": [100],
    "trainer.seed": [12345],
    "trainer.max_epochs": 1,
    "model.dropout_p": [0],
    "model.learning_rate": [0.1],
    "model.use_ema": False,
    "trainer.precision": 16,
    "trainer.batch_size": 32,
    "trainer.deterministic": True,
    "active.acq_size": 2,
    "active.num_iter": 1,
}

naming_conv = (
    "{data}/active-{active}/smoke_test_{model}_acq-{query}"
)

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
    )

    launcher.launch_runs()
