"""
25-condition multi-label phenotyping (data.multitask=true) with the benchmark
LSTM, TIME SERIES ONLY (model.use_static=false: no Age/Gender, features
256-d): all baselines, random initial labelled set only. Recipe identical to
sweep_exp_mimic3_pheno_single_task_lstm_nodemo_baselines.py (benchmark LSTM:
Adam lr 1e-3, batch 8, dropout 0.3, no clipping; 50 epochs, single-stay
cohort, per-label weighted BCE). The vendi-q2 / bandit counterpart is
sweep_exp_mimic3_pheno_lstm_nodemo_tune_vendi.py.

Grid: 5 baselines x (start, step) in {(100, 100), (500, 500)} x 3 seeds
= 30 runs.

active.balanced=False: the whole budget drawn randomly from the pool.

wandb data name: mimic3_pheno_lstm_nodemo.
"""
from argparse import ArgumentParser

from launcher import ExperimentLauncher

config_dict = {
    "model": ["lstm"],
    # slowest first: with --jobs the queue starts runs in this order
    "query": [
        "badge",
        "kcentergreedy",
        "bald",
        "entropy",
        "random",
    ],
    "data": ["mimic3_pheno"],
    "active": ["mimic3_med"],
    "optim": ["adam"],
}

# Run order is itertools.product over config_dict then hparam_dict values, with
# the LAST multi-value key varying fastest (innermost loop):
#     for query: for label(=start/step): for seed
hparam_dict = {
    "data.multitask": True,                  # all 25 conditions, 25 sigmoid outputs
    "data.single_stay_only": True,
    "data.balanced_sampling": False,
    "active.balanced": [False],
    "model.use_static": False,
    "model.weighted_loss": True,             # per-label BCE pos_weight = #neg/#pos
    "trainer.max_epochs": 50,
    "model.dropout_p": 0.3,                  # benchmark: --dropout 0.3
    "model.learning_rate": 0.001,            # benchmark: Adam lr 1e-3
    "model.weight_decay": 0.0,               # benchmark: --l2 0
    "model.use_ema": False,
    "trainer.batch_size": 8,                 # benchmark: --batch_size 8
    "trainer.gradient_clip_val": 0,          # benchmark: no clipping
    "optim.lr_scheduler.name": "null",       # benchmark: constant lr
    "trainer.precision": 32,
    "trainer.deterministic": True,
    "active.num_iter": 10,                   # 10 acquisition rounds
    # (start, step) budgets, iterated jointly via joint_iteration below:
    "active.num_labelled": [100, 500],
    "active.acq_size": [100, 500],
    # placed LAST so seed is the innermost (fastest-varying) axis
    "trainer.seed": [12345, 12346, 12347],
}

# no task token: every single-task LSTM folder carries task-8 / task-11
naming_conv = "{data}/active-{active}/model-{model}_demo-{model.use_static}_bal-{active.balanced}_acq-{query}_sso-{data.single_stay_only}_start-{active.num_labelled}_step-{active.acq_size}_ep-{trainer.max_epochs}"

# num_labelled and acq_size step together as (start, step) pairs
joint_iteration = [["active.num_labelled", "active.acq_size"]]

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
