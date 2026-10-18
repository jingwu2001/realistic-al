"""
Single-task phenotyping with the Harutyunyan et al. 2017 benchmark LSTM
(model=lstm, models/networks/bayesian_lstm.py): the second architecture next to
SAND, to check whether the query-method ranking is model-agnostic.

Same AL protocol and grid as the SAND single-task sweeps
(sweep_exp_mimic3_pheno_single_task.py and its 2026-09-29 predecessor): 7 methods
x tasks {8, 11} x both cohorts x (start, step) in {(100, 100), (500, 500)}
x 3 seeds = 168 runs. Only the model and its training recipe change; the recipe
follows the benchmark's phenotyping LSTM (README: --dim 256 --depth 1
--dropout 0.3 --batch_size 8; Adam lr 1e-3, no clipping, no L2) instead of the
STraTS SAND recipe (lr 5e-4, batch 16, clip 0.3, dropout 0.2).
  task 8  = Congestive heart failure; nonhypertensive
  task 11 = Diabetes mellitus without complication
wandb data name: mimic3_pheno_t08_lstm / mimic3_pheno_t11_lstm (256-d demo
embedding, the 2026-10-04 sweep); with the narrowed default model.d_static_out=2
(2026-10-08) it is mimic3_pheno_t08_lstm_demo2 / mimic3_pheno_t11_lstm_demo2.
"""
from argparse import ArgumentParser

from launcher import ExperimentLauncher

config_dict = {
    "model": ["lstm"],
    # slowest first: with --jobs the queue starts runs in this order, so the
    # plain-vendi runs (~2 min/acquisition vs ~20 s for vendi-q2) don't pile
    # up at the tail
    "query": [
        # "badge",
        # "kcentergreedy",
        # "bald",
        # "entropy",
        # "random",
        # "vendi-q2",
        # "vendi",
        "bandit"
    ],
    "data": ["mimic3_pheno"],
    "active": ["mimic3_med"],  # random initial labelling (num_iter=10)
    "optim": ["adam"],
}

# Run order is itertools.product over config_dict then hparam_dict values, with
# the LAST multi-value key varying fastest (innermost loop):
#     for query: for task: for cohort: for label(=start/step): for seed
hparam_dict = {
    "data.multitask": False,                 # single condition, scalar output
    "data.task_id": [8],                 # phenotype.npy column (0-based)
    "data.single_stay_only": [True],  # sweep both cohorts
    "data.balanced_sampling": False,
    "model.weighted_loss": True,             # BCE pos_weight = #neg/#pos (as for SAND)
    "trainer.max_epochs": 50,                # AL protocol, same as SAND
    "model.dropout_p": 0.3,                  # benchmark: --dropout 0.3
    "model.learning_rate": [0.001],          # benchmark: Adam lr 1e-3
    "model.weight_decay": 0.0,               # benchmark: --l2 0
    "model.use_ema": False,
    "trainer.batch_size": 8,                 # benchmark: --batch_size 8
    "trainer.gradient_clip_val": 0,          # benchmark: no clipping
    "optim.lr_scheduler.name": "null",       # benchmark: constant lr
    "trainer.precision": 32,
    "trainer.deterministic": True,
    "active.num_iter": 10,                   # 10 acquisition rounds
    # (start, step) budgets, iterated jointly via joint_iteration below:
    "active.num_labelled": [500],
    "active.acq_size": [500],
    # fixed vendi block, same as the SAND single-task runs (read by vendi and
    # vendi-q2; inert for the other methods).
    "query.vendi.kernel": "rbf",
    "query.vendi.gamma": "median",             # 1/(2 D)
    "query.vendi.normalization": "l2",
    "query.vendi.batch_size": 1,
    "query.vendi.use_grad": False,
    # placed LAST so seed is the innermost (fastest-varying) axis
    "trainer.seed": [12345, 12346, 12347],
}

# task id in the path, else the two tasks collide on disk
naming_conv = "{data}/active-{active}/model-{model}_acq-{query}_task-{data.task_id}_sso-{data.single_stay_only}_start-{active.num_labelled}_step-{active.acq_size}_ep-{trainer.max_epochs}"

# num_labelled and acq_size step together as (start, step) pairs rather than
# forming a 3x3 cartesian product. task_id / single_stay_only are cartesian axes.
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
