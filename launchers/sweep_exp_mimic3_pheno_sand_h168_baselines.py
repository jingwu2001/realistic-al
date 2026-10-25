"""
25-condition phenotyping, SAND (real attention) on the STraTS input at 168 h
(data=mimic3_pheno_h168: 129 variables, first week of the stay) with
demographics (64-d): the AL baselines. This is the setting with the most
headroom in the 2026-10-25 random grid (random gain +.106 at 100 -> 1k, +.096
at 500 -> 5k, best final macro AUROC .807).

random is NOT run here: its 6 runs come from
sweep_exp_mimic3_pheno_sand_window_random.py (folder 20261025-22f904d09). The
hparam_dict below is copied from that launcher (model.use_static fixed to True),
so wandb names/groups and the per-seed initial labelled sets match. Keep it
identical.

Grid: {badge, kcentergreedy, bald, entropy} x (start, step) in {(100, 100),
(500, 500)} x 3 seeds = 24 runs.

wandb data name: mimic3_pheno_h168_attnfix.
"""
from argparse import ArgumentParser

from launcher import ExperimentLauncher

config_dict = {
    "model": ["sand"],
    # slowest first: with --jobs the queue starts runs in this order
    "query": ["badge", "kcentergreedy", "bald", "entropy"],
    "data": ["mimic3_pheno_h168"],
    "active": ["mimic3_med"],
    "optim": ["adam"],
}

hparam_dict = {
    "data.multitask": True,                  # all 25 conditions, 25 sigmoid outputs
    "data.single_stay_only": True,
    "data.balanced_sampling": False,
    "active.balanced": [False],
    "model.use_static": True,                # SAND's 64-d Age+Gender embedding (the leading setting)
    "model.weighted_loss": True,             # per-label BCE pos_weight
    "trainer.max_epochs": 50,
    "model.dropout_p": 0.2,
    "model.learning_rate": 0.0005,
    "model.weight_decay": 0.0,               # mimic sand: AdamW default wd=0
    "model.use_ema": False,
    "trainer.batch_size": 16,                # mimic sand: train_batch_size 16
    "trainer.gradient_clip_val": 0.3,        # mimic sand: clip_grad_norm 0.3
    "optim.lr_scheduler.name": "null",       # mimic sand: constant lr (no scheduler)
    "trainer.precision": 32,
    "trainer.deterministic": True,
    "active.num_iter": 10,                   # 10 acquisition rounds
    "active.num_labelled": [100, 500],
    "active.acq_size": [100, 500],
    "trainer.seed": [12345, 12346, 12347],
}

naming_conv = "{data}/active-{active}/model-{model}_attnfix_demo-{model.use_static}_acq-{query}_sso-{data.single_stay_only}_start-{active.num_labelled}_step-{active.acq_size}_ep-{trainer.max_epochs}"

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
