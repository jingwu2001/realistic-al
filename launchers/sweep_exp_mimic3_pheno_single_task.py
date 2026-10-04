"""
Single-task phenotyping: each condition trained on its own (data.multitask=False,
one sigmoid output), same recipe as sweep_exp_mimic3_pheno_basic.py.
  task 8  = Congestive heart failure; nonhypertensive
  task 11 = Diabetes mellitus without complication
wandb data name: mimic3_pheno_t08 / mimic3_pheno_t11.
"""
from argparse import ArgumentParser

from launcher import ExperimentLauncher

config_dict = {
    "model": ["sand"],
    # slowest first: with --jobs the queue starts runs in this order, so the
    # plain-vendi runs (~2 min/acquisition vs ~20 s for vendi-q2) don't pile
    # up at the tail
    "query": [
        "vendi",
        "vendi-q2",
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
    "data.task_id": [8, 11],                 # phenotype.npy column (0-based)
    "data.single_stay_only": [False, True],  # sweep both cohorts
    "data.balanced_sampling": False,
    "model.weighted_loss": True,             # BCE pos_weight = #neg/#pos
    "trainer.max_epochs": 50,
    "model.dropout_p": 0.2,
    "model.learning_rate": [0.0005],
    "model.weight_decay": 0.0,             # mimic sand: AdamW default wd=0
    "model.use_ema": False,
    "trainer.batch_size": 16,              # mimic sand: train_batch_size 16
    "trainer.gradient_clip_val": 0.3,      # mimic sand: clip_grad_norm 0.3
    "optim.lr_scheduler.name": "null",     # mimic sand: constant lr (no scheduler)
    "trainer.precision": 32,
    "trainer.deterministic": True,
    "active.num_iter": 10,                  # 10 acquisition rounds
    # (start, step) budgets, iterated jointly via joint_iteration below:
    "active.num_labelled": [100, 500],
    "active.acq_size": [100, 500],
    # fixed vendi block (only vendi-q2 reads it; inert for the plain methods).
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
