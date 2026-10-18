"""
Single-task phenotyping with the benchmark LSTM (demo2: Age/Gender narrowed to
2 tanh dims, features 258-d): all baselines, with and without a class-balanced
initial labelled set.
  task 8   = Congestive heart failure; nonhypertensive
  task 11  = Diabetes mellitus without complication
Recipe identical to sweep_exp_mimic3_pheno_single_task_lstm_demo2.py (benchmark
LSTM: Adam lr 1e-3, batch 8, dropout 0.3, no clipping; 50 epochs, single-stay
cohort, weighted BCE). The 25-condition counterpart is
sweep_exp_mimic3_pheno_lstm_balanced_baselines.py.

Grid: 5 baselines x tasks {8, 11} x (start, step) in {(100, 100), (500, 500)}
x active.balanced {True, False} x 3 seeds = 120 runs.

active.balanced (initial labelled set only; queries are unaffected; see
run_training.label_active_dm):
  True  -> num_labelled // 2 positives + num_labelled // 2 negatives
           (pool prevalence: t8 24%, t11 19%).
  False -> the whole budget drawn randomly from the pool (as all earlier runs).
           These cells repeat the 2026-10-09 demo2 baselines
           (20261009-fc59d4043): same seeds, same LSTM code -> same runs.

wandb data name: mimic3_pheno_t08_lstm_demo2 / mimic3_pheno_t11_lstm_demo2,
plus "_bal" for active.balanced=True.
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
#     for query: for task: for balanced: for label(=start/step): for seed
hparam_dict = {
    "data.multitask": False,                 # single condition, scalar output
    "data.task_id": [8, 11],                 # phenotype.npy column (0-based)
    "data.single_stay_only": True,
    "data.balanced_sampling": False,
    "active.balanced": [False],
    "model.use_static": False,
    "model.weighted_loss": True,             # BCE pos_weight = #neg/#pos
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

# task id and balanced in the path, else those cells collide on disk
naming_conv = "{data}/active-{active}/model-{model}_demo-{model.use_static}_bal-{active.balanced}_acq-{query}_task-{data.task_id}_sso-{data.single_stay_only}_start-{active.num_labelled}_step-{active.acq_size}_ep-{trainer.max_epochs}"

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
