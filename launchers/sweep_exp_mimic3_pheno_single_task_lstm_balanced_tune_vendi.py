"""
Single-task phenotyping with the benchmark LSTM, TIME SERIES ONLY
(model.use_static=false: no Age/Gender, features 256-d): vendi-q2 and the
bandit (diversity arm = vendi-q2) over every NON-SATURATED (kernel, gamma,
normalization, embedding), with and without a class-balanced initial labelled
set, on task 8 (CHF) and task 11 (diabetes w/o complication).
Recipe, tasks and active.balanced exactly as
sweep_exp_mimic3_pheno_single_task_lstm_balanced_baselines.py (the baselines to
compare against). The 25-condition counterpart is
sweep_exp_mimic3_pheno_lstm_balanced_tune_vendi.py.

Grid: {vendi-q2, bandit} x 10 kernel configs (7 feature + 3 gradient) x tasks
{8, 11} x (start, step) in {(100, 100), (500, 500)} x active.balanced
{True, False} x 3 seeds = 480 runs.

Which configs are kept: measured 2026-10-18/19 with
query_diversity.kernel_diagnostics on the random-acquisition LSTM demo2 runs in
20261009-fc59d4043 (t8/t11 x start 100/500 x 3 seeds x rounds 0/4/8 = 36
checkpoints, L = 100 .. 4500). No TS-only LSTM checkpoint exists, so the
TS-only embeddings were approximated by cutting the 2 demo dims from the demo2
features / gradients (for gradients this changed nothing). Same rule as the
ECG5000_4c sweep: kept if, in EVERY checkpoint, < 30% of labelled-pair kernel
values are > 0.99 (all-ones collapse) and < 30% are < 0.01 (identity
collapse), no candidates tie at the acquisition cutoff, and (rbf) the median
kernel argument is in [0.1, 5].
  feature (use_grad=false), 7 kept:
           rbf median x {l2, minmax, zscore, none} (median arg 0.50; up to
           24-26% of pairs < 0.01, borderline), rbf dim + zscore,
           cosine x {zscore, none} (cosine + l2 is the same kernel as + none)
  gradient (use_grad=true; one output -> (p - y_hat) * h), 3 kept:
           rbf median + l2, cosine x {zscore, none}
  dropped (both): rbf 1.0 (removed), rbf dim + {l2, minmax, none}
           (-> all-ones), cosine + minmax (up to 50% / 100% of pairs > 0.99)
  dropped (gradient only): rbf median + {minmax, zscore, none}: when the
           model fits its labelled set (seed 12346, round 0, start 100, both
           tasks) labelled gradients shrink (median norm .018 vs pool up to
           3.6), median gamma ~2000 isolates ~1900 pool rows (every k^2
           underflows) and 3400-3600 candidates tie at the cutoff -> the batch
           is a random pick among them; l2 removes the magnitude (gamma 2.5,
           none isolated). rbf dim + zscore: 93% of pairs > 0.99 at that
           checkpoint.

query is a cartesian axis; the four vendi keys are iterated jointly (one
entry per kernel config).
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
    # bandit first: it runs a BALD pass and the vendi-q2 arm every round
    "query": ["bandit", "vendi-q2"],
    "data": ["mimic3_pheno"],
    "active": ["mimic3_med"],
    "optim": ["adam"],
}

# Run order is itertools.product over config_dict then hparam_dict values, with
# the LAST multi-value key varying fastest (innermost loop):
#     for query: for task: for balanced: for kernel cfg: for start/step: for seed
hparam_dict = {
    "data.multitask": False,                 # single condition, scalar output
    "data.task_id": [8, 11],                 # phenotype.npy column (0-based)
    "data.single_stay_only": True,
    "data.balanced_sampling": False,
    "active.balanced": [True, False],
    "query.vendi.kernel": [k for k, _, _, _ in vq2_configs],
    "query.vendi.gamma": [g for _, g, _, _ in vq2_configs],
    "query.vendi.normalization": [n for _, _, n, _ in vq2_configs],
    "query.vendi.use_grad": [u for _, _, _, u in vq2_configs],
    "query.vendi.batch_size": 1,
    "query.bandit.use_vendi_q2": True,       # diversity arm = vendi-q2
    "model.use_static": False,
    "model.weighted_loss": True,             # BCE pos_weight = #neg/#pos
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
# which would put all vendi-q2 configs in one folder. bandit becomes
# "acq-bandit-vq2" (query.bandit.use_vendi_q2).
naming_conv = "{data}/active-{active}/model-{model}_demo-{model.use_static}_bal-{active.balanced}_acq-{query}_grad-{query.vendi.use_grad}_vkernel-{query.vendi.kernel}_vgamma-{query.vendi.gamma}_vnorm-{query.vendi.normalization}_task-{data.task_id}_sso-{data.single_stay_only}_start-{active.num_labelled}_step-{active.acq_size}_ep-{trainer.max_epochs}"

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
