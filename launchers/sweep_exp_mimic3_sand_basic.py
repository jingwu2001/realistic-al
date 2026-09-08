"""
MIMIC-III (SAND) — random+baseline learning curves, both cohorts.

Seven acquisition methods (random, badge, kcentergreedy, bald, entropy,
vendi-q2, vendi-q2-sequential), 10 acquisition rounds (num_iter=10 => models
trained at num_labelled + {0..9}*acq_size), swept jointly over three
(start, step) budgets:
  (100, 100)   -> 100 .. 1000
  (500, 500)   -> 500 .. 5000
  (1000, 1000) -> 1000 .. 10000

Both cohorts are swept via ``data.single_stay_only`` in {False, True}:
  False -> all stays        (44,812; STraTS fixed split, ~11.5% positive)
  True  -> single-stay only  (26,609 patients w/ one ICU stay, ~12.2% positive)
The test cohort is fixed (STraTS's split); the seed only moves val/pool/init.

Natural imbalanced split only (data=mimic3_sand, BAL EVAL OFF) — comparable to
the STraTS paper. Other settings mimic the STraTS SAND recipe
(~/Desktop/STraTS/sand.md, Part 4): model=sand, lr 5e-4, dropout 0.2, weighted
loss, batch_size 16, gradient_clip_val 0.3, optim=adam + weight_decay=0 (== AdamW
wd=0), lr_scheduler=null (constant lr), 50 epochs. The vendi block is pinned
(rbf / gamma="dim" / zscore); only vendi-q2[-sequential] read it, inert for the
rest.

7 methods x 3 budgets x 2 cohorts x 3 seeds = 126 runs. --debug prints them.
"""
from argparse import ArgumentParser

from launcher import ExperimentLauncher

config_dict = {
    "model": ["sand"],
    "query": [
        "random",
        "badge",
        "kcentergreedy",
        "bald",
        "entropy",
        "vendi-q2",
        "vendi-q2-sequential",
    ],
    "data": ["mimic3_sand"],   # natural stratified split; L+U/val/test all ~11.5% pos
    "active": ["mimic3_med"],  # imbalanced initial pool (num_iter=10)
    "optim": ["adam"],
}

hparam_dict = {
    "data.single_stay_only": [False, True],  # sweep both cohorts
    "data.balanced_sampling": False,
    "model.weighted_loss": True,
    "trainer.seed": [12345, 12346, 12347],
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
    "active.num_labelled": [100, 500, 1000],
    "active.acq_size": [100, 500, 1000],
    # fixed vendi block (only vendi-q2 / vendi-q2-sequential read it; inert for
    # the plain methods). batch_size consumed only by the sequential path.
    "query.vendi.kernel": "rbf",
    "query.vendi.gamma": "dim",             # 1/(2 D)
    "query.vendi.normalization": "zscore",
    "query.vendi.batch_size": 1,
}

naming_conv = "{data}/active-{active}/model-{model}_acq-{query}_sso-{data.single_stay_only}_start-{active.num_labelled}_step-{active.acq_size}_ep-{trainer.max_epochs}"

# num_labelled and acq_size step together as (start, step) pairs rather than
# forming a 3x3 cartesian product. single_stay_only is a normal cartesian axis.
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
