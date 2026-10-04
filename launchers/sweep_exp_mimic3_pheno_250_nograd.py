"""
query size = 250
all baselines x 3 seeds
vendi-q2 sweep, grad=False, x 3 seeds
"""
from argparse import ArgumentParser

from launcher import ExperimentLauncher

# Baselines each run ONCE and ignore the query.vendi.* block entirely.
baselines = ["random", "badge", "kcentergreedy", "bald", "entropy"]

# Well-conditioned (normalization, gamma) cells for vendi-q2, aligned row-by-row
# and zipped via joint_iteration below (NOT a cartesian product). See module
# docstring for why only these five keep the RBF kernel between I and J.
vendi_norms  = ["none",   "l2",     "minmax", "zscore", "zscore", "minmax"]
vendi_gammas = ["median", "median", "median", "median", "dim", 1.0]

# One combined query axis, joint-iterated with (normalization, gamma) so that:
#   * each baseline is pinned to a single throwaway cell (zscore, dim) -> 1 run,
#   * vendi-q2 is repeated once per real cell and aligned to it       -> 5 runs.
# The three joint keys must stay equal length (here 5 baselines + 5 cells = 10).
query_list = baselines + ["vendi-q2"] * len(vendi_norms)
norm_list  = ["minmax"] * len(baselines) + vendi_norms
gamma_list = [1.0]    * len(baselines) + vendi_gammas

config_dict = {
    "model": ["sand"],
    "query": query_list,             # 5 baselines + vendi-q2 x 5 kernel cells
    "data": ["mimic3_pheno"],        # 25-condition multi-label phenotyping target
    "active": ["mimic3_med"],        # random initial labelling (balanced=False)
    "optim": ["adam"],
}

hparam_dict = {
    "data.single_stay_only": False,         # pilot: all cohort only
    "data.balanced_sampling": False,
    "model.weighted_loss": True,            # per-label BCE pos_weight
    "trainer.max_epochs": 50,
    "model.dropout_p": 0.2,
    "model.learning_rate": [0.0005],
    "model.weight_decay": 0.0,              # mimic sand: AdamW default wd=0
    "model.use_ema": False,
    "trainer.batch_size": 16,               # mimic sand: train_batch_size 16
    "trainer.gradient_clip_val": 0.3,       # mimic sand: clip_grad_norm 0.3
    "optim.lr_scheduler.name": "null",      # mimic sand: constant lr (no scheduler)
    "trainer.precision": 32,
    "trainer.deterministic": True,
    "active.num_iter": 10,                   # 10 acquisition rounds
    # (start, step) budgets, iterated jointly via joint_iteration below:
    "active.num_labelled": [250],
    "active.acq_size": [250],
    
    "query.vendi.use_grad": False,
    "query.vendi.kernel": "rbf",
    "query.vendi.normalization": norm_list,
    "query.vendi.gamma": gamma_list,
    "query.vendi.batch_size": 1,
    "trainer.seed": [12345, 12346, 12347],                 # single seed for the pilot
}

# query + normalization + gamma must all be in the name, else the vendi cells
# collide to one path (they share query=vendi-q2, distinguished only by the
# kernel cell). Baselines keep a constant _norm-zscore_gamma-dim suffix — inert
# but harmless, and distinct by {query}.
naming_conv = "{data}/active-{active}/model-{model}_acq-{query}_norm-{query.vendi.normalization}_gamma-{query.vendi.gamma}_sso-{data.single_stay_only}_start-{active.num_labelled}_step-{active.acq_size}_ep-{trainer.max_epochs}"

# (num_labelled, acq_size) step together as a (start, step) pair; (query,
# normalization, gamma) are the aligned method/kernel cells (zipped, not
# producted) so baselines run once and vendi-q2 runs once per cell.
joint_iteration = [
    ["active.num_labelled", "active.acq_size"],
    ["query", "query.vendi.normalization", "query.vendi.gamma"],
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
