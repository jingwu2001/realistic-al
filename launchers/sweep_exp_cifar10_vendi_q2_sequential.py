"""
CIFAR-10 vendi-q2-sequential sweep — the closed-form Renyi q=2 Vendi score run
GREEDILY (pick `query.vendi.batch_size` points, fold them into the labeled set,
repeat) over the full embedding-source x kernel x normalization x gamma grid,
crossed with a batch_size axis.

Identical to sweep_exp_cifar10_vendi_q2.py in every other respect (long-tail
cifar10_imb, cifar10_med, weighted loss, val_size 1250, sgd_cosine, 200 epochs,
single seed) — the ONLY changes are:
  * query method `vendi-q2` -> `vendi-q2-sequential` (see config/query and
    query_diversity.NAMES); the sequential dispatch returns selection order.
  * a new INDEPENDENT batch_size axis, ++query.vendi.batch_size in {50, 10, 1}
    (50 = coarse lazy batches, 1 = fully greedy), producted with the 22 query
    rows. batch_size is in the naming_conv so the three settings get distinct
    run directories (otherwise they collide and clobber each other).

The 22-row (kernel, norm, gamma, use_grad) grid and the cosine gamma/l2
handling are unchanged — see sweep_exp_cifar10_vendi_q2.py for the rationale.

22 query rows x 3 batch sizes x 1 (data, active) setting x 1 seed = 66 runs.
"""
from argparse import ArgumentParser
from itertools import product

from launcher import ExperimentLauncher


class QuerySweepLauncher(ExperimentLauncher):
    """Sweep-specific launcher tweaks (as in sweep_exp_cifar10_vendi_q2.py):

    1. parse_product: zip each joint_iteration group into one compound axis
       before the cartesian product, so only genuinely independent axes (seed,
       batch_size, ...) are producted -- the aligned row lists below would
       otherwise blow up the intermediate product.
    2. generate_name: strip the rbf-only gamma part from the cosine cells, on
       the formatted name (the base class's stripping targets the un-validated
       naming convention and never matches -- dead code).
    """

    def parse_product(self) -> list:
        joint_dict = self.merge_dictionaries(self.config_args, self.overwrite_args)
        joint_groups = self.joint_iteration or []
        if joint_groups and not isinstance(joint_groups[0], (list, tuple)):
            joint_groups = [joint_groups]

        axes = []  # (keys tuple, list of aligned value tuples)
        grouped_keys = set()
        for group in joint_groups:
            lengths = {len(joint_dict[key]) for key in group}
            assert len(lengths) == 1, f"joint_iteration group {group} has unequal lengths"
            axes.append((tuple(group), list(zip(*[joint_dict[key] for key in group]))))
            grouped_keys.update(group)
        for key, values in joint_dict.items():
            if key not in grouped_keys:
                axes.append(((key,), [(value,) for value in values]))

        final_product = []
        for combo in product(*[values for _, values in axes]):
            full_dict = {}
            for (keys, _), values in zip(axes, combo):
                full_dict.update(dict(zip(keys, values)))
            if self.skip_config(full_dict):
                continue
            config_dict = {key: full_dict[key] for key in self.config_args}
            param_dict = {key: full_dict[key] for key in self.overwrite_args}
            if (config_dict, param_dict) not in final_product:
                final_product.append((config_dict, param_dict))
        return final_product

    def generate_name(self, config_dict: dict, param_dict: dict) -> str:
        name = super().generate_name(config_dict, param_dict)
        merged = self.merge_dictionaries(config_dict, param_dict)
        kernel = merged.get("query.vendi.kernel")
        gamma = merged.get("query.vendi.gamma")
        if kernel in ("cosine", "linear"):
            name = name.replace(f"_gamma-{gamma}", "")  # gamma is rbf-only
        return name


# --- sweep axes -----------------------------------------------------------
USE_GRAD = [False, True]                        # feat vs BADGE-style gradients
NORMS = ["none", "l2", "minmax", "zscore"]      # all normalize_features options
# RBF bandwidth: the two scale-adaptive heuristics (see module docstring).
# To probe fixed bandwidths, add floats here ONLY alongside a fixed norm.
RBF_GAMMAS = ["dim", "median"]
KERNELS = ["rbf", "cosine"]                      # linear unsupported by q=2
BATCH_SIZES = [50, 10, 1]                        # sequential picks/round (1 = greedy)

# Aligned rows (zipped via joint_iteration). One (kernel, norm, gamma) tuple
# per cell; cosine gets a single inert gamma (1.0, stripped from names) and
# skips l2 (== none for cosine).
use_grads, kernels, norms, gammas = [], [], [], []
for use_grad in USE_GRAD:
    for norm in NORMS:
        for gamma in RBF_GAMMAS:
            use_grads.append(use_grad); kernels.append("rbf")
            norms.append(norm); gammas.append(gamma)
    for norm in NORMS:
        if norm == "l2":            # cosine is L2-invariant: cosine+l2 == cosine+none
            continue
        use_grads.append(use_grad); kernels.append("cosine")
        norms.append(norm); gammas.append(1.0)

queries = ["vendi-q2-sequential"] * len(kernels)

# (data, active) as one zipped axis -- long-tail cifar10_imb, med only (matches
# the July sweep). To re-add settings, extend data/active AND the zipped
# data.val_size / model.weighted_loss lists together.
config_dict = {
    "model": "resnet",
    "query": queries,
    "data": ["cifar10_imb"],
    "active": ["cifar10_med"],
    "optim": ["sgd_cosine"],
}

hparam_dict = {
    "data.val_size": [1250],
    "model.weighted_loss": [True],

    # "trainer.seed": [12345, 12346, 12347],
    "trainer.seed": [12345],
    "trainer.max_epochs": 200,
    "model.dropout_p": [0.5] * len(queries),
    "model.learning_rate": [0.1],
    "model.use_ema": False,
    "data.transform_train": [
        "cifar_randaugmentMC",
    ],
    "trainer.precision": 16,
    "trainer.batch_size": 1024,
    "trainer.deterministic": True,
    # embedding source + kernel/normalization/gamma sweep on the shared
    # query.vendi.* block (use_grad=true = gradient embeddings)
    "query.vendi.use_grad": use_grads,
    "query.vendi.kernel": kernels,
    "query.vendi.normalization": norms,
    "query.vendi.gamma": gammas,
    # independent axis: sequential batch size (producted with the 22 query rows)
    "query.vendi.batch_size": BATCH_SIZES,
}

naming_conv = (
    "{data}/active-{active}/"
    "basic_model-{model}_drop-{model.dropout_p}_aug-{data.transform_train}"
    "_acq-{query}_grad-{query.vendi.use_grad}"
    "_norm-{query.vendi.normalization}_kernel-{query.vendi.kernel}"
    "_gamma-{query.vendi.gamma}_bs-{query.vendi.batch_size}"
    "_ep-{trainer.max_epochs}"
)

joint_iteration = [
    ["data", "active", "data.val_size", "model.weighted_loss"],
    [
        "query",
        "model.dropout_p",
        "query.vendi.use_grad",
        "query.vendi.kernel",
        "query.vendi.normalization",
        "query.vendi.gamma",
    ],
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

    launcher = QuerySweepLauncher(
        config_dict,
        hparam_dict,
        launcher_args,
        naming_conv,
        path_to_ex_file,
        joint_iteration=joint_iteration,
    )

    launcher.launch_runs()
