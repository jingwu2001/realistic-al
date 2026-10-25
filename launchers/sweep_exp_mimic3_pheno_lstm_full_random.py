"""
25-condition phenotyping with the benchmark LSTM on the STraTS input over the
WHOLE stay (data=mimic3_pheno_full: 129 variables, unpadded stays up to ~4,150 h),
random query only: its learning-curve headroom next to the 24 h STraTS input
(sweep_exp_mimic3_pheno_lstm_nodemo_baselines.py, folder 20261021-54f0b2b6c) and
the SAND window grid (sweep_exp_mimic3_pheno_sand_window_random.py). SAND can't
take whole stays (fixed T), so this cell is LSTM-only.

hparam_dict copied from sweep_exp_mimic3_pheno_lstm_nodemo_baselines.py (benchmark
LSTM recipe, time series only, single-stay), so the runs are directly comparable
and reusable as the random baseline of a later full sweep on this input.

Grid: random x (start, step) in {(100, 100), (500, 500)} x 3 seeds = 6 runs.

wandb data name: mimic3_pheno_full_lstm_nodemo.
"""
from argparse import ArgumentParser

from launcher import ExperimentLauncher

config_dict = {
    "model": ["lstm"],
    "query": ["random"],
    "data": ["mimic3_pheno_full"],
    "active": ["mimic3_med"],
    "optim": ["adam"],
}

hparam_dict = {
    "data.multitask": True,                  # all 25 conditions, 25 sigmoid outputs
    "data.single_stay_only": True,
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
    # (start, step) budgets, iterated jointly via joint_iteration below:
    "active.num_labelled": [100, 500],
    "active.acq_size": [100, 500],
    # placed LAST so seed is the innermost (fastest-varying) axis
    "trainer.seed": [12345, 12346, 12347],
}

naming_conv = "{data}/active-{active}/model-{model}_demo-{model.use_static}_bal-{active.balanced}_acq-{query}_sso-{data.single_stay_only}_start-{active.num_labelled}_step-{active.acq_size}_ep-{trainer.max_epochs}"

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
