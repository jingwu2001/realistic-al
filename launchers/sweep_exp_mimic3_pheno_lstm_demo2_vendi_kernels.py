"""
25-condition multi-label phenotyping (data.multitask=true) with the benchmark
LSTM, demo2 (Age/Gender embedding narrowed to 2 tanh dims, features 258-d):
vendi-q2 over the NON-SATURATED kernel configs, random initial labelled set
only. Recipe identical to sweep_exp_mimic3_pheno_lstm_nodemo_vendi_kernels.py
except model.use_static=true. Baselines:
sweep_exp_mimic3_pheno_lstm_demo2_baselines.py.

Grid: vendi-q2 x 10 kernel configs (7 feature + 3 gradient)
x (start, step) in {(100, 100), (500, 500)} x 3 seeds = 60 runs.

Kernel list = the single-task LSTM one, measured 2026-10-18/19 on single-task
demo2 checkpoints (20261009-fc59d4043; see
sweep_exp_mimic3_pheno_single_task_lstm_nodemo_tune_vendi.py for the
criterion). Never probed on 25 labels: the gradient embedding is
25 x 258 = 6450-d, so check al/query/kernel_frac_near_{one,zero} of the first
finished runs. The 2 demo dims are tanh-bounded but still ~20% of feature l2
distances on single-task demo2 runs; zscore weights them like any other dim.

query is a single value; the four vendi keys are iterated jointly (one entry
per kernel config).
"""
from argparse import ArgumentParser

from launcher import ExperimentLauncher

# (kernel, gamma, normalization, use_grad); gamma is ignored by cosine
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
    "model": ["lstm"],
    "query": ["vendi-q2"],
    "data": ["mimic3_pheno"],
    "active": ["mimic3_med"],
    "optim": ["adam"],
}

# Run order is itertools.product over config_dict then hparam_dict values, with
# the LAST multi-value key varying fastest (innermost loop):
#     for kernel cfg: for start/step: for seed
hparam_dict = {
    "data.multitask": True,                  # all 25 conditions, 25 sigmoid outputs
    "data.single_stay_only": True,
    "data.balanced_sampling": False,
    "active.balanced": [False],
    "query.vendi.kernel": [k for k, _, _, _ in vq2_configs],
    "query.vendi.gamma": [g for _, g, _, _ in vq2_configs],
    "query.vendi.normalization": [n for _, _, n, _ in vq2_configs],
    "query.vendi.use_grad": [u for _, _, _, u in vq2_configs],
    "query.vendi.batch_size": 1,
    "model.use_static": True,
    "model.d_static_out": 2,                 # demo2 (the lstm.yaml default)
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

# "_vkernel/_vgamma/_vnorm", not "_kernel/_gamma/_norm": launcher.generate_name
# strips the latter for every query except vendi/bandit (vendi-q2 included),
# which would put all vendi-q2 configs in one folder. demodim in the path:
# demo-True alone doesn't say how wide the embedding is.
naming_conv = "{data}/active-{active}/model-{model}_demo-{model.use_static}_demodim-{model.d_static_out}_bal-{active.balanced}_acq-{query}_grad-{query.vendi.use_grad}_vkernel-{query.vendi.kernel}_vgamma-{query.vendi.gamma}_vnorm-{query.vendi.normalization}_sso-{data.single_stay_only}_start-{active.num_labelled}_step-{active.acq_size}_ep-{trainer.max_epochs}"

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
