"""
4-class ECG5000 fill-in: the (config, seed) pairs still missing after the
2026-10-14 OOM and the 2026-10-15 disk-full crash, for configs that already
have SOME of their 3 / 6 seeds. Each entry repeats the overrides of the launch
that originally ran its config, so the new seeds pair with the finished ones.

  feature vendi-q2, med, balanced init, rbf-median-none: seed 12350
      (exp_ecg5000_basic_tune_kernel_balanced.py; 12345-12349 done)
  feature bandit with the q=1 diversity arm, low, balanced init,
  rbf-median-minmax: seeds 12347-12350
      (2026-10-14 bandit launch, killed by the OOM; 12345/12346 done). That
      launch set neither query.bandit.use_vendi_q2 nor query.bandit.num_classes,
      so both stay at their yaml defaults (false, 10) here too.

Not included: med vendi-q2 rbf-dim-zscore / cosine-zscore / cosine-minmax
(no balanced seed 12348-12350 finished; rerun by
exp_ecg5000_basic_tune_kernel_balanced.py), and low grad bandit-vq2
rbf-median-l2 (all 6 seeds died in the OOM; the grad sweep in
exp_ecg5000_basic_tune_kernel.py now runs med only).

Every varying key is iterated jointly over `missing_runs`, one entry per run.
"""
from argparse import ArgumentParser

from launcher import ExperimentLauncher

# (query, active, kernel, gamma, normalization, seed)
missing_runs = [
    ("vendi-q2", "ecg5000_med", "rbf", "median", "none", 12350),
    ("bandit", "ecg5000_low", "rbf", "median", "minmax", 12347),
    ("bandit", "ecg5000_low", "rbf", "median", "minmax", 12348),
    ("bandit", "ecg5000_low", "rbf", "median", "minmax", 12349),
    ("bandit", "ecg5000_low", "rbf", "median", "minmax", 12350),
]

config_dict = {
    "model": ["inception_time"],
    "query": [q for q, _, _, _, _, _ in missing_runs],
    "data": ["ecg5000_4c"],
    "active": [a for _, a, _, _, _, _ in missing_runs],
    "optim": ["adam"],
}

hparam_dict = {
    "query.vendi.kernel": [k for _, _, k, _, _, _ in missing_runs],
    "query.vendi.gamma": [g for _, _, _, g, _, _ in missing_runs],
    "query.vendi.normalization": [n for _, _, _, _, n, _ in missing_runs],
    "query.vendi.use_grad": False,
    "data.val_size": 500,
    "trainer.max_epochs": 17,
    "model.dropout_p": 0.5,
    "model.learning_rate": 0.0001442986898059786,
    "model.weighted_loss": True,
    "trainer.precision": 32,
    "trainer.batch_size": 8,
    "trainer.deterministic": True,
    "active.balanced": [True],
    "trainer.seed": [s for _, _, _, _, _, s in missing_runs],
}

# "_vkernel/_vgamma/_vnorm", not "_kernel/_gamma/_norm": launcher.generate_name
# strips the latter for every query except vendi/bandit (vendi-q2 included),
# which would put all vendi-q2 configs in one folder
naming_conv = "{data}/active-{active}/basic_model-{model}_bal-{active.balanced}_acq-{query}_grad-{query.vendi.use_grad}_vkernel-{query.vendi.kernel}_vgamma-{query.vendi.gamma}_vnorm-{query.vendi.normalization}_wloss-{model.weighted_loss}_ep-{trainer.max_epochs}"

joint_iteration = [
    [
        "query",
        "active",
        "query.vendi.kernel",
        "query.vendi.gamma",
        "query.vendi.normalization",
        "trainer.seed",
    ],
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
