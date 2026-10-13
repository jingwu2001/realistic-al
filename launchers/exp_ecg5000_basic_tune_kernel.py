"""
4-class ECG5000 (class 4 "Unclassified Beat" removed, data=ecg5000_4c): all
baselines + a vendi-q2 kernel x normalization sweep restricted to
NON-SATURATED kernels, with and without a class-balanced initial labelled set.
Recipe identical to exp_ecg5000_basic.py (InceptionTime, Adam lr 1.44e-4,
dropout 0.5, batch 8, 17 epochs, weighted loss).

Grid: (5 baselines + 8 vendi-q2 configs) x {ecg5000_low (25 -> 250 labels),
ecg5000_med (50 -> 500 labels)} x active.balanced {True, False} x 3 seeds
= 156 runs.

active.balanced (initial labelled set only; queries are unaffected):
  True  -> 5 per class (20 labels), the rest of the budget drawn randomly
           (realistic-al label strategy for imbalanced data)
  False -> the whole budget drawn randomly from the pool

Which vendi-q2 configs are kept (feature embedding, use_grad=false): measured
2026-10-13 with query_diversity.kernel_diagnostics on the InceptionTime features
of the March ECG5000 checkpoints (3 budgets x 3 runs x rounds 0/4/8). A config
is kept if, in EVERY round, < 30% of labelled-pair kernel values are > 0.99
(all-ones collapse) and < 30% are < 0.01 (identity collapse), no candidates tie
at the acquisition cutoff, and (rbf) the median kernel argument is in [0.1, 5].
  kept:    rbf median x {l2, minmax, zscore, none}, rbf dim + zscore,
           rbf 1.0 + l2, cosine x {zscore, minmax}
  dropped: rbf 1.0 + {minmax, zscore, none}  (-> identity; minmax is the old
           ECG5000 config), rbf dim + {l2, none} (-> all-ones), rbf dim +
           minmax (median arg 0.04), cosine + none (up to 41% of pairs > 0.99).
           cosine + l2 is the same kernel as cosine + none.

query and the vendi block are iterated jointly (one entry per method); the
baselines get null vendi values, which they never read.
"""
from argparse import ArgumentParser

from launcher import ExperimentLauncher

baselines = ["random", "entropy", "bald", "badge", "kcentergreedy"]

# (kernel, gamma, normalization); gamma is ignored by cosine
vq2_configs = [
    ("rbf", "median", "l2"),
    ("rbf", "median", "minmax"),
    ("rbf", "median", "zscore"),
    ("rbf", "median", "none"),
    ("rbf", "dim", "zscore"),
    ("rbf", 1.0, "l2"),
    ("cosine", None, "zscore"),
    ("cosine", None, "minmax"),
]

config_dict = {
    "model": ["inception_time"],
    "query": baselines + ["vendi-q2"] * len(vq2_configs),
    "data": ["ecg5000_4c"],
    "active": [
        "ecg5000_low",
        "ecg5000_med",
    ],
    "optim": ["adam"],
}

hparam_dict = {
    "query.vendi.kernel": [None] * len(baselines) + [k for k, _, _ in vq2_configs],
    "query.vendi.gamma": [None] * len(baselines) + [g for _, g, _ in vq2_configs],
    "query.vendi.normalization": [None] * len(baselines) + [n for _, _, n in vq2_configs],
    "query.vendi.use_grad": False,
    "data.val_size": 500,
    "trainer.max_epochs": 17,
    "model.dropout_p": 0.5,
    "model.learning_rate": 0.0001442986898059786,
    "model.weighted_loss": True,
    "trainer.precision": 32,
    "trainer.batch_size": 8,
    "trainer.deterministic": True,
    "active.balanced": [True, False],
    # placed LAST so seed is the innermost (fastest-varying) axis
    "trainer.seed": [12345, 12346, 12347],
}

# "_vkernel/_vgamma/_vnorm", not "_kernel/_gamma/_norm": launcher.generate_name
# strips the latter for every query except vendi/bandit (vendi-q2 included),
# which would put all vendi-q2 configs in one folder
naming_conv = "{data}/active-{active}/basic_model-{model}_bal-{active.balanced}_acq-{query}_vkernel-{query.vendi.kernel}_vgamma-{query.vendi.gamma}_vnorm-{query.vendi.normalization}_wloss-{model.weighted_loss}_ep-{trainer.max_epochs}"

joint_iteration = [
    ["query", "query.vendi.kernel", "query.vendi.gamma", "query.vendi.normalization"],
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
