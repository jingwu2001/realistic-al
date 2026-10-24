"""
Dropout check for the 25-condition TS-only LSTM, at the AL starting budgets:
does the benchmark's LSTM input dropout help, and should the MC head use 0.5
like CIFAR / ECG5000? One training per run (query=random, active.num_iter=1)
on a random starting set, selected on val/macro_auroc as in the AL sweeps.
Recipe otherwise identical to sweep_exp_mimic3_pheno_lstm_nodemo_baselines.py,
so the (0.3, 0.3) cell repeats round 0 of its random runs (same seeds).

Grid: input dropout {0, 0.3} x head dropout {0.3, 0.5}
x start {100, 500} x 3 seeds = 24 runs.

Logged to the separate wandb project realistic-al-tuning, so these runs never
pool with the AL groups in realistic-al.
"""
from argparse import ArgumentParser

from launcher import ExperimentLauncher

config_dict = {
    "model": ["lstm"],
    "query": ["random"],
    "data": ["mimic3_pheno"],
    "active": ["mimic3_med"],
    "optim": ["adam"],
}

# Run order is itertools.product over config_dict then hparam_dict values, with
# the LAST multi-value key varying fastest (innermost loop):
#     for input dropout: for head dropout: for start: for seed
hparam_dict = {
    "data.multitask": True,                  # all 25 conditions, 25 sigmoid outputs
    "data.single_stay_only": True,
    "data.balanced_sampling": False,
    "active.balanced": False,
    "model.use_static": False,
    "model.weighted_loss": True,             # per-label BCE pos_weight = #neg/#pos
    "trainer.max_epochs": 50,
    "model.dropout_p": 0.3,                  # benchmark: --dropout 0.3
    "model.input_dropout_p": [0.0, 0.3],
    "model.head_dropout_p": [0.3, 0.5],
    "model.learning_rate": 0.001,            # benchmark: Adam lr 1e-3
    "model.weight_decay": 0.0,               # benchmark: --l2 0
    "model.use_ema": False,
    "trainer.batch_size": 8,                 # benchmark: --batch_size 8
    "trainer.gradient_clip_val": 0,          # benchmark: no clipping
    "optim.lr_scheduler.name": "null",       # benchmark: constant lr
    "trainer.precision": 32,
    "trainer.deterministic": True,
    "trainer.wandb_project": "realistic-al-tuning",
    "active.num_iter": 1,                    # one training on the starting set
    "active.num_labelled": [100, 500],
    "active.acq_size": [100, 500],
    # placed LAST so seed is the innermost (fastest-varying) axis
    "trainer.seed": [12345, 12346, 12347],
}

naming_conv = "{data}/active-{active}/tune_model-{model}_demo-{model.use_static}_indrop-{model.input_dropout_p}_headdrop-{model.head_dropout_p}_sso-{data.single_stay_only}_start-{active.num_labelled}_ep-{trainer.max_epochs}"

# num_labelled and acq_size step together as (start, step) pairs
joint_iteration = [["active.num_labelled", "active.acq_size"]]

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
