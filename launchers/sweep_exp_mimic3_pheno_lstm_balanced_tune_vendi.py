"""
25-condition multi-label phenotyping (data.multitask=true) with the benchmark
LSTM (demo2, features 258-d): vendi-q2 and the bandit (diversity arm =
vendi-q2) over every NON-SATURATED (kernel, gamma, normalization), with and
without a class-balanced initial labelled set. Recipe and active.balanced
exactly as sweep_exp_mimic3_pheno_lstm_balanced_baselines.py (the baselines to
compare against). The single-task (t8/t11) counterpart is
sweep_exp_mimic3_pheno_single_task_lstm_balanced_tune_vendi.py.

Grid: {vendi-q2, bandit} x 8 kernel configs x (start, step) in
{(100, 100), (500, 500)} x active.balanced {True, False} x 3 seeds = 192 runs.

Which configs are kept (feature embedding, use_grad=false): the same list as
the single-task launcher, measured 2026-10-18 with
query_diversity.kernel_diagnostics on the LSTM demo2 features of the
random-acquisition SINGLE-TASK runs in 20261009-fc59d4043 (t8/t11 x start
100/500 x 3 seeds x rounds 0/4/8 = 36 checkpoints, L = 100 .. 4500); no
25-label LSTM run existed to probe. Kept if, in EVERY checkpoint, < 30% of
labelled-pair kernel values are > 0.99 (all-ones collapse) and < 30% are < 0.01
(identity collapse), no candidates tie at the acquisition cutoff, and (rbf) the
median kernel argument is in [0.1, 5].
  kept:    rbf median x {l2, minmax, zscore, none} (median arg 0.50),
           rbf dim + zscore (arg 0.24-1.09), rbf 1.0 + l2 (arg 0.20-1.34),
           cosine x {zscore, none} (cosine + l2 is the same kernel as + none)
  dropped: rbf 1.0 + {minmax, zscore, none} (-> identity: up to 85/100/99% of
           pairs < 0.01; 36/18959/3318 candidates tied at the cutoff),
           rbf dim + {l2, minmax, none} (-> all-ones: up to 100/90/34% > 0.99),
           cosine + minmax (up to 32% of pairs > 0.99 at L=900)
Borderline on single-task features: the gamma=median rbfs reached 24-26% of
pairs < 0.01 at L=4500 (cut 30%). The 25-output model learns different
features, so check the logged al/query/kernel_frac_near_{one,zero} of the first
finished runs.

query is a cartesian axis; the three vendi keys are iterated jointly (one
entry per kernel config).
"""
from argparse import ArgumentParser

from launcher import ExperimentLauncher

# (kernel, gamma, normalization); gamma is ignored by cosine
vq2_configs = [
    ("rbf", "median", "l2"),
    ("rbf", "median", "minmax"),
    ("rbf", "median", "zscore"),
    ("rbf", "median", "none"),
    ("rbf", "dim", "zscore"),
    ("rbf", 1.0, "l2"),
    ("cosine", None, "zscore"),
    ("cosine", None, "none"),
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
#     for query: for balanced: for kernel cfg: for start/step: for seed
hparam_dict = {
    "data.multitask": True,                  # all 25 conditions, 25 sigmoid outputs
    "data.single_stay_only": True,
    "data.balanced_sampling": False,
    "active.balanced": [True, False],
    "query.vendi.kernel": [k for k, _, _ in vq2_configs],
    "query.vendi.gamma": [g for _, g, _ in vq2_configs],
    "query.vendi.normalization": [n for _, _, n in vq2_configs],
    "query.vendi.use_grad": False,
    "query.vendi.batch_size": 1,
    "query.bandit.use_vendi_q2": True,       # diversity arm = vendi-q2
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
# which would put all vendi-q2 configs in one folder. bandit becomes
# "acq-bandit-vq2" (query.bandit.use_vendi_q2). No task token: every
# single-task LSTM folder carries task-8 / task-11.
naming_conv = "{data}/active-{active}/model-{model}_demodim-{model.d_static_out}_bal-{active.balanced}_acq-{query}_grad-{query.vendi.use_grad}_vkernel-{query.vendi.kernel}_vgamma-{query.vendi.gamma}_vnorm-{query.vendi.normalization}_sso-{data.single_stay_only}_start-{active.num_labelled}_step-{active.acq_size}_ep-{trainer.max_epochs}"

joint_iteration = [
    ["query.vendi.kernel", "query.vendi.gamma", "query.vendi.normalization"],
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
