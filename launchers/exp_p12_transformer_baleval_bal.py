"""
P12 Transformer — balanced test/val, BALANCED training pool (weighted sampling).

Test and val sets are 50% positive + 50% negative.
Training dataloader uses WeightedRandomSampler to present balanced batches
from the naturally imbalanced pool.  Initial labeled set is also balanced
(p12_med_bal: 50 survived + 50 died).
"""
from argparse import ArgumentParser

from launcher import ExperimentLauncher

config_dict = {
    "model": ["transformer2"],
    "query": [
        "bald",
        "random",
        "entropy",
        "badge",
        "kcentergreedy",
        "vendi",
        "bandit",
    ],
    "data": ["p12_transformer_baleval"],
    "active": ["p12_med_bal"],  # balanced initial pool: 50 survived + 50 died
    "optim": ["adam"],
}

hparam_dict = {
    "data.balanced_sampling": True,
    "model.weighted_loss": True,
    "trainer.seed": [12345, 12346, 12347],
    "trainer.max_epochs": 50,
    "model.dropout_p": [0.5] * len(config_dict["query"]),
    "model.learning_rate": [0.001],
    "model.weight_decay": [1e-5],
    "model.use_ema": False,
    "trainer.batch_size": 64,
    "trainer.precision": 32,
    "trainer.deterministic": True,
}

naming_conv = "{data}/active-{active}/model-{model}_drop-{model.dropout_p}_acq-{query}_ep-{trainer.max_epochs}_bs-{data.balanced_sampling}_wl-{model.weighted_loss}"

joint_iteration = [
    ["model.dropout_p", "query"],
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
