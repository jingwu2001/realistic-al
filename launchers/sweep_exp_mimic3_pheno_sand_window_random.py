"""
25-condition phenotyping with SAND (real attention), random query only: the
learning-curve headroom of each input x window x demographics combination.

Inputs (data configs):
  17 variables (mimic3-benchmarks, 76 channels): mimic3_pheno_bench_h24,
      mimic3_pheno_bench (48 h), mimic3_pheno_bench_h168
  129 variables (STraTS, 387 channels): mimic3_pheno (24 h), mimic3_pheno_h48,
      mimic3_pheno_h168
Demographics: model.use_static False (none) / True (SAND's 64-d embedding,
d_static_out null).

Recipe identical to sweep_exp_mimic3_pheno_basic.py (STraTS SAND: batch 16,
lr 5e-4, clip 0.3, dropout 0.2, constant lr; 50 epochs, per-label weighted BCE),
single-stay cohort, random initial set.

Grid: 6 inputs x 2 demographics x (start, step) in {(100, 100), (500, 500)}
x 3 seeds = 72 runs.

Reusing these runs as the random baseline of a later full sweep: keep the data
config, this hparam_dict (incl. model.use_static), active config, budgets and
seeds identical and drop "random" from the query list. The wandb name/group
({data}[_nodemo]_attnfix/mimic3_med/random) and the per-seed initial labelled set
then match; filter budgets on config.num_labelled / acq_size.
"""
from argparse import ArgumentParser

from launcher import ExperimentLauncher

config_dict = {
    "model": ["sand"],
    "query": ["random"],
    # longest windows first: the slowest runs start first with --jobs
    "data": [
        "mimic3_pheno_h168",
        "mimic3_pheno_bench_h168",
        "mimic3_pheno_h48",
        "mimic3_pheno_bench",
        "mimic3_pheno",
        "mimic3_pheno_bench_h24",
    ],
    "active": ["mimic3_med"],
    "optim": ["adam"],
}

hparam_dict = {
    "data.multitask": True,                  # all 25 conditions, 25 sigmoid outputs
    "data.single_stay_only": True,
    "data.balanced_sampling": False,
    "active.balanced": [False],
    "model.use_static": [False, True],       # no demographics / SAND's 64-d Age+Gender embedding
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
