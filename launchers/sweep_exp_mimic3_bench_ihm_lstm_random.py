"""
In-hospital mortality on the mimic3-benchmarks input (Harutyunyan et al.: the
benchmark's own task = first 48 h of ICU stays lasting >= 48 h, 17 variables, 76
channels, its patient split) with the benchmark LSTM, random query only: how
much the learning curve rises over the AL budget (headroom). Same LSTM recipe
as sweep_exp_mimic3_bench_pheno_lstm_random.py (weighted 2-class loss); all
stays, the benchmark's cohort.

Grid: (start, step) in {(100, 100), (500, 500)} x 3 seeds = 6 runs.

wandb data name: mimic3_ihm_bench_lstm_nodemo.
"""
from argparse import ArgumentParser

from launcher import ExperimentLauncher

config_dict = {
    "model": ["lstm"],
    "query": ["random"],
    "data": ["mimic3_ihm_bench"],
    "active": ["mimic3_med"],
    "optim": ["adam"],
}

hparam_dict = {
    "data.single_stay_only": False,          # the benchmark's cohort: all stays
    "data.balanced_sampling": False,
    "active.balanced": [False],
    "model.use_static": False,
    "model.weighted_loss": True,             # class-weighted loss (~13% deaths)
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
