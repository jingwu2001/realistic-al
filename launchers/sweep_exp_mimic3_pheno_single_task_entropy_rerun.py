"""
Rerun of the entropy runs from sweep_exp_mimic3_pheno_single_task.py (launched
2026-09-29, dir 20260929-491df56c) that died with "OSError: [Errno 28] No space
left on device" when the disk filled up on 2026-09-30 ~13:26:
  task 11 (Diabetes mellitus without complication), single_stay_only=True,
  both (start, step) budgets x 3 seeds = 6 runs.
All other entropy cells (task 8 both cohorts, task 11 sso=False) finished 3/3.
Same recipe as the parent sweep; only query/task_id/cohort are pinned.
"""
from argparse import ArgumentParser

from launcher import ExperimentLauncher

config_dict = {
    "model": ["sand"],
    "query": ["entropy"],
    "data": ["mimic3_pheno"],
    "active": ["mimic3_med"],  # random initial labelling (num_iter=10)
    "optim": ["adam"],
}

hparam_dict = {
    "data.multitask": False,                 # single condition, scalar output
    "data.task_id": 11,                      # phenotype.npy column (0-based)
    "data.single_stay_only": True,           # only the failed cohort
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
    # fixed vendi block, kept for config parity with the parent sweep (inert for entropy)
    "query.vendi.kernel": "rbf",
    "query.vendi.gamma": "median",             # 1/(2 D)
    "query.vendi.normalization": "l2",
    "query.vendi.batch_size": 1,
    "query.vendi.use_grad": True,
    # placed LAST so seed is the innermost (fastest-varying) axis
    "trainer.seed": [12345, 12346, 12347],
}

# same convention as the parent sweep (task id in the path)
naming_conv = "{data}/active-{active}/model-{model}_acq-{query}_task-{data.task_id}_sso-{data.single_stay_only}_start-{active.num_labelled}_step-{active.acq_size}_ep-{trainer.max_epochs}"

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
