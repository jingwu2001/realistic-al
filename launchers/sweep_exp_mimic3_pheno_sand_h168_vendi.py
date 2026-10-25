"""
25-condition phenotyping, SAND (real attention) on the STraTS input at 168 h
(data=mimic3_pheno_h168: 129 variables, first week of the stay) with
demographics (64-d): vendi-q2 and the bandit (diversity arm = vendi-q2) over the
MIMIC-10 kernel configs, as sweep_exp_mimic3_pheno_lstm_nodemo_tune_vendi.py
did for the LSTM. Same recipe as sweep_exp_mimic3_pheno_sand_h168_baselines.py;
random comes from sweep_exp_mimic3_pheno_sand_window_random.py (folder
20261025-22f904d09).

Grid: {vendi-q2, bandit} x 10 kernel configs (7 feature + 3 gradient)
x (start, step) in {(100, 100), (500, 500)} x 3 seeds = 120 runs.

Kernels (MIMIC-10):
  feature (use_grad=false): rbf median x {l2, minmax, zscore, none},
      rbf dim + zscore, cosine x {zscore, none}
  gradient (use_grad=true): rbf median + l2, cosine x {zscore, none}
On SAND features with demographics, feat rbf-median-minmax gives the Age/Gender
block ~50% of the kernel exponent (analysis/probe_sand_demo_kernel.py); every
other config < 5%. The 25-label gradient embedding is 25 x 832 = 20,800-d, so
check the logged al/query/kernel_frac_near_{one,zero} of the first runs.

wandb data name: mimic3_pheno_h168_attnfix.
"""
from argparse import ArgumentParser

from launcher import ExperimentLauncher

# (kernel, gamma, normalization, use_grad)
vq2_configs = [
    ("rbf", "median", "l2", False),
    ("rbf", "median", "minmax", False),
    ("rbf", "median", "zscore", False),
    ("rbf", "median", "none", False),
    ("rbf", "dim", "zscore", False),
    ("cosine", None, "zscore", False),
    ("cosine", None, "none", False),
    # gradient embeddings: only the magnitude-free kernels stay healthy
    ("rbf", "median", "l2", True),
    ("cosine", None, "zscore", True),
    ("cosine", None, "none", True),
]

config_dict = {
    "model": ["sand"],
    "query": ["vendi-q2", "bandit"],
    "data": ["mimic3_pheno_h168"],
    "active": ["mimic3_med"],
    "optim": ["adam"],
}

hparam_dict = {
    "query.vendi.kernel": [k for k, _, _, _ in vq2_configs],
    "query.vendi.gamma": [g for _, g, _, _ in vq2_configs],
    "query.vendi.normalization": [n for _, _, n, _ in vq2_configs],
    "query.vendi.use_grad": [u for _, _, _, u in vq2_configs],
    "query.vendi.batch_size": 1,
    "query.bandit.use_vendi_q2": True,       # diversity arm = vendi-q2
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

naming_conv = "{data}/active-{active}/model-{model}_attnfix_demo-{model.use_static}_acq-{query}_grad-{query.vendi.use_grad}_vkernel-{query.vendi.kernel}_vgamma-{query.vendi.gamma}_vnorm-{query.vendi.normalization}_sso-{data.single_stay_only}_start-{active.num_labelled}_step-{active.acq_size}_ep-{trainer.max_epochs}"

joint_iteration = [
    ["query.vendi.kernel", "query.vendi.gamma", "query.vendi.normalization",
     "query.vendi.use_grad"],
    ["active.num_labelled", "active.acq_size"],
]

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
