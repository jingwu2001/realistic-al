"""
25-condition phenotyping on the mimic3-benchmarks input (Harutyunyan et al.:
17 variables, 76 channels, its patient split) with the benchmark LSTM, random
query only: how much the learning curve rises over the AL budget for each
window (headroom). Recipe identical to
sweep_exp_mimic3_pheno_lstm_nodemo_baselines.py (benchmark LSTM: Adam lr 1e-3,
batch 8, dropout 0.3, no clipping; 50 epochs, per-label weighted BCE, no
Age/Gender).

Grid: data {mimic3_pheno_bench (first 48 h), mimic3_pheno_bench_full (whole
stay)} x single_stay_only {True, False} x (start, step) in {(100, 100),
(500, 500)} x 3 seeds = 24 runs.

wandb data names: mimic3_pheno_bench_lstm_nodemo, mimic3_pheno_bench_full_lstm_nodemo.
"""
from argparse import ArgumentParser

from launcher import ExperimentLauncher

config_dict = {
    "model": ["lstm"],
    "query": ["random"],
    # whole stay first: the slowest runs start first with --jobs
    "data": ["mimic3_pheno_bench_full", "mimic3_pheno_bench"],
    "active": ["mimic3_med"],
    "optim": ["adam"],
}

hparam_dict = {
    "data.multitask": True,                  # all 25 conditions, 25 sigmoid outputs
    "data.single_stay_only": [True, False],
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
    "active.num_labelled": [100, 500],
    "active.acq_size": [100, 500],
    "trainer.seed": [12345, 12346, 12347],
}

naming_conv = "{data}/active-{active}/model-{model}_demo-{model.use_static}_acq-{query}_sso-{data.single_stay_only}_start-{active.num_labelled}_step-{active.acq_size}_ep-{trainer.max_epochs}"

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
