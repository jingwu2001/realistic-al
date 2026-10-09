"""
Single-task phenotyping with the benchmark LSTM and a NARROWED demographics
embedding: Age/Gender go through Linear(2 -> 2*dim) + Tanh + Linear(2*dim -> 2)
+ Tanh (model.d_static_out=2) before the concat, instead of the 256-wide
linear-output embedding of sweep_exp_mimic3_pheno_single_task_lstm.py. Features
are 258-d (h 256 + demo 2), all in (-1, 1); kcenter/vendi/BADGE use all 258.

Question: does the query-method ranking change once Age/Gender stop dominating
the feature distances? With the 256-d embedding the demo half was ~96% of the
pairwise get_features distance, so kcenter/vendi/BADGE mostly diversified over
demographics (the time-series-only control is
sweep_exp_mimic3_pheno_single_task_nodemo.py --model lstm).

Model state at launch (2026-10-09): the demo output ends in a Tanh, and
ConsistentMCDropout covers all 258 features, demo dims included.

Same AL protocol, recipe and grid as the 256-d LSTM sweep it is compared
against: 7 methods x tasks {8, 11} x single-stay cohort x (start, step) in
{(100, 100), (500, 500)} x 3 seeds = 84 runs. Same seeds -> same initial
labelled sets, so each run pairs with its 256-d and no-demo counterparts.
  task 8  = Congestive heart failure; nonhypertensive
  task 11 = Diabetes mellitus without complication
wandb data name: mimic3_pheno_t08_lstm_demo2 / mimic3_pheno_t11_lstm_demo2.
"""
from argparse import ArgumentParser

from launcher import ExperimentLauncher

config_dict = {
    "model": ["lstm"],
    # same order as the 256-d LSTM sweep: plain vendi (by far the slowest
    # query) last, so the baselines report back first
    "query": [
        "badge",
        "kcentergreedy",
        "bald",
        "entropy",
        "random",
        "vendi-q2",
        # "vendi",
    ],
    "data": ["mimic3_pheno"],
    "active": ["mimic3_med"],  # random initial labelling (num_iter=10)
    "optim": ["adam"],
}

# Run order is itertools.product over config_dict then hparam_dict values, with
# the LAST multi-value key varying fastest (innermost loop):
#     for query: for task: for label(=start/step): for seed
hparam_dict = {
    "model.use_static": True,
    "model.d_static_out": 2,                 # <- narrowed demo embedding (was 256)
    "data.multitask": False,                 # single condition, scalar output
    "data.task_id": [8, 11],                 # phenotype.npy column (0-based)
    "data.single_stay_only": True,
    "data.balanced_sampling": False,
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
    # fixed vendi block, same as the 256-d LSTM runs (read by vendi and
    # vendi-q2; inert for the other methods)
    "query.vendi.kernel": "rbf",
    "query.vendi.gamma": "median",           # median heuristic 1/(2 median^2)
    "query.vendi.normalization": "l2",
    "query.vendi.batch_size": 1,
    "query.vendi.use_grad": False,
    # placed LAST so seed is the innermost (fastest-varying) axis
    "trainer.seed": [12345, 12346, 12347],
}

# demo width in the path, so these never share a folder name with the 256-d runs
naming_conv = "{data}/active-{active}/model-{model}_demodim-{model.d_static_out}_acq-{query}_task-{data.task_id}_sso-{data.single_stay_only}_start-{active.num_labelled}_step-{active.acq_size}_ep-{trainer.max_epochs}"

# num_labelled and acq_size step together as (start, step) pairs rather than
# forming a cartesian product
joint_iteration = [["active.num_labelled", "active.acq_size"]]

path_to_ex_file = "src/main.py"

if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    ExperimentLauncher.add_argparse_args(parser)
    parser.add_argument(
        "--tasks", type=int, nargs="+", default=None,
        help="phenotype task ids (0-based phenotype.npy columns); default 8 11",
    )
    parser.add_argument(
        "--wandb-test", action="store_true", dest="wandb_test",
        help="Create each run's wandb run (name/group/tags/config, tagged "
             "'wandb_test', state 'failed') and exit immediately — verifies "
             "the wandb wiring without loading data or training",
    )
    launcher_args = parser.parse_args()

    if launcher_args.tasks:
        hparam_dict["data.task_id"] = launcher_args.tasks

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
