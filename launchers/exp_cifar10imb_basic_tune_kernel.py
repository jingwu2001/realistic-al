import subprocess

from argparse import ArgumentParser

from launcher import ExperimentLauncher

_commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).strip().decode()

kernels_gammas = [('rbf',    1.0),
                  ('rbf',    'median'),
                  ('rbf',    'dim'),
                  ('cosine', None)]
normalizations = ["l2", "minmax", "zscore"]

config_dict = {
    "model": "resnet",
    "query": [
        "vendi-q2"
    ],
    "data": ["cifar10_imb"],
    "active": [
        "cifar10_low",
        "cifar10_med",
    ],
    "optim": ["sgd_cosine"],
    # kernel & gamma vary together as (kernel, gamma) pairs (see joint_iteration):
    # each rbf gets its bandwidth, cosine gets a null gamma (ignored for cosine).
    "query.vendi.kernel": [kernel for kernel, _ in kernels_gammas],
    "query.vendi.gamma": [gamma for _, gamma in kernels_gammas],
    "query.vendi.normalization": normalizations,
    "query.vendi.use_grad": [True, False],
}

hparam_dict = {
    "model.weighted_loss": True,
    "data.val_size": [50 * 5, 250 * 5],
    "trainer.seed": [12345, 12346, 12347],
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

naming_conv = (
    "{data}/active-{active}/"
    f"commit-{_commit}/"
    "basic_model-{model}_drop-{model.dropout_p}_aug-{data.transform_train}_acq-{query}_norm-{query.vendi.normalization}_kernel-{query.vendi.kernel}_gamma-{query.vendi.gamma}_ep-{trainer.max_epochs}__wloss-{model.weighted_loss}"
)

joint_iteration = [
    ["query.vendi.kernel", "query.vendi.gamma"],
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
