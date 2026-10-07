"""
Single-task phenotyping with TIME-SERIES INPUT ONLY: the static demographics
(Age, Gender; demo.npy) are dropped via model.use_static=false, so the network
sees only the (24, 387) hourly tensor (129 charted variables x values/mask/delta;
Weight and Height stay in, as charted time-series variables).

Question: does removing the demographics help the AL methods separate? With the
demo embedding in, kcenter/vendi distances on the LSTM features are ~96% Age/Gender
(SAND: 3-4%), so diversity methods mostly diversify over demographics.

Default model = SAND (faster end to end than the LSTM under the same GPU load:
~1.8 vs ~2.2 s/epoch at 1000 labels, 6.8 vs 8.9 s per MC query pass, 2026-10-07;
the LSTM's batch 8 recipe doubles the optimizer steps). `--model lstm` switches to
the benchmark LSTM with its own recipe.

Same AL protocol as the demo runs it is compared against
(sweep_exp_mimic3_pheno_single_task.py / _lstm.py): 7 methods x tasks x
single-stay cohort x (start, step) in {(100, 100), (500, 500)} x 1 seed (12345)
= 28 runs for the default tasks {8, 11}. Seed 12345 has the same initial labelled
set as the demo runs, so each run pairs with its demo counterpart.
  task 8  = Congestive heart failure; nonhypertensive
  task 11 = Diabetes mellitus without complication
  (`--tasks 17 15 4 10` for the multitask-shortlisted conditions)
wandb data name: mimic3_pheno_t08_nodemo (SAND) / mimic3_pheno_t08_lstm_nodemo.
"""
from argparse import ArgumentParser

from launcher import ExperimentLauncher

# Per-model training recipes, identical to the demo-input sweeps so that only
# use_static changes.
RECIPES = {
    # STraTS SAND recipe (sweep_exp_mimic3_pheno_single_task.py)
    "sand": {
        "model.dropout_p": 0.2,
        "model.learning_rate": 0.0005,
        "model.weight_decay": 0.0,
        "trainer.batch_size": 16,
        "trainer.gradient_clip_val": 0.3,
    },
    # Harutyunyan benchmark LSTM recipe (sweep_exp_mimic3_pheno_single_task_lstm.py)
    "lstm": {
        "model.dropout_p": 0.3,
        "model.learning_rate": 0.001,
        "model.weight_decay": 0.0,
        "trainer.batch_size": 8,
        "trainer.gradient_clip_val": 0,
    },
}

config_dict = {
    "model": ["sand"],
    # cheapest first: this is a single-seed probe run sequentially, so the
    # baselines report back early and plain vendi (q=1, the slowest by far:
    # hours of query time per run at acq 500) comes last
    "query": [
        "random",
        "badge",
        "entropy",
        "bald",
        "kcentergreedy",
        "vendi-q2",
        "vendi",
    ],
    "data": ["mimic3_pheno"],
    "active": ["mimic3_med"],  # random initial labelling (num_iter=10)
    "optim": ["adam"],
}

# Run order is itertools.product over config_dict then hparam_dict values, with
# the LAST multi-value key varying fastest (innermost loop):
#     for query: for task: for label(=start/step)
hparam_dict = {
    "model.use_static": False,               # <- time series only, no Age/Gender
    "data.multitask": False,                 # single condition, scalar output
    "data.task_id": [8, 11],                 # phenotype.npy column (0-based)
    "data.single_stay_only": True,
    "data.balanced_sampling": False,
    "model.weighted_loss": True,             # BCE pos_weight = #neg/#pos
    "trainer.max_epochs": 50,
    **RECIPES["sand"],
    "model.use_ema": False,
    "optim.lr_scheduler.name": "null",       # constant lr
    "trainer.precision": 32,
    "trainer.deterministic": True,
    "active.num_iter": 10,                   # 10 acquisition rounds
    # (start, step) budgets, iterated jointly via joint_iteration below:
    "active.num_labelled": [100, 500],
    "active.acq_size": [100, 500],
    # fixed vendi block, same as the demo single-task runs (read by vendi and
    # vendi-q2; inert for the other methods)
    "query.vendi.kernel": "rbf",
    "query.vendi.gamma": "median",           # median heuristic 1/(2 median^2)
    "query.vendi.normalization": "l2",
    "query.vendi.batch_size": 1,
    "query.vendi.use_grad": False,
    "trainer.seed": [12345],                 # single-seed probe
}

# demo-{use_static} in the path, else these collide on disk with the demo runs
naming_conv = "{data}/active-{active}/model-{model}_demo-{model.use_static}_acq-{query}_task-{data.task_id}_sso-{data.single_stay_only}_start-{active.num_labelled}_step-{active.acq_size}_ep-{trainer.max_epochs}"

# num_labelled and acq_size step together as (start, step) pairs rather than
# forming a cartesian product
joint_iteration = [["active.num_labelled", "active.acq_size"]]

path_to_ex_file = "src/main.py"

if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    ExperimentLauncher.add_argparse_args(parser)
    parser.add_argument(
        "--model", choices=sorted(RECIPES), default="sand",
        help="network (and its training recipe); default sand (the faster one)",
    )
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

    config_dict["model"] = [launcher_args.model]
    hparam_dict.update(RECIPES[launcher_args.model])
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
