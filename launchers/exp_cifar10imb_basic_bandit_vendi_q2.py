"""
CIFAR-10 imbalanced — LinUCB bandit whose diversity arm runs vendi-q2
(query.bandit.use_vendi_q2, closed-form Renyi q=2 Vendi; q is pinned to 2 in
main.py). Same training setup as exp_cifar10imb_basic_tune_kernel.py so results
are comparable to the vendi-q2 kernel sweep.

Diversity arm: feature embeddings (use_grad=False), cosine kernel, min-max
normalization. gamma is left at its vendi.yaml default — ignored by cosine.
model.dropout_p=0.5 feeds the uncertainty (BALD) arm's MC-dropout samples;
query.bandit.num_classes keeps its bandit.yaml default of 10 (CIFAR-10).

2 active settings x 3 seeds = 6 runs.
"""
import subprocess

from argparse import ArgumentParser

from launcher import ExperimentLauncher

_commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).strip().decode()

config_dict = {
    "model": "resnet",
    "query": [
        "bandit"
    ],
    "data": ["cifar10_imb"],
    "active": [
        # "cifar10_low",
        "cifar10_med",
    ],
    "optim": ["sgd_cosine"],
    "query.bandit.use_vendi_q2": True,
    "query.vendi.kernel": "cosine",
    "query.vendi.normalization": "minmax",
    "query.vendi.use_grad": False,
}

hparam_dict = {
    "model.weighted_loss": True,
    "data.val_size": [250 * 5],
    "trainer.seed": [12345],
    "trainer.max_epochs": 200,
    "model.weight_decay": 5e-3,
    "model.dropout_p": 0.5,
    "model.learning_rate": [0.1],
    "model.use_ema": False,
    "data.transform_train": [
        "cifar_randaugmentMC",
    ],
    "trainer.batch_size": 1024,
    "trainer.precision": 16,
    "trainer.deterministic": True,
    "trainer.num_workers": 8
}

# {query} renders as "bandit-vq2" when use_vendi_q2 is set (launcher.generate_name)
naming_conv = (
    "{data}/active-{active}/"
    f"commit-{_commit}/"
    "basic_model-{model}_drop-{model.dropout_p}_aug-{data.transform_train}_acq-{query}_norm-{query.vendi.normalization}_kernel-{query.vendi.kernel}_ep-{trainer.max_epochs}__wloss-{model.weighted_loss}"
)

joint_iteration = [
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
