"""
CIFAR-10 query-method sweep — long-tail cifar10_imb only (med setting,
weighted loss, val_size 1250 per the exp_cifar10imb_basic.py conventions).

Run order: 6 baselines (random, entropy, kcentergreedy, bald, badge,
variationratios; batchbald pruned) -> gvendi and vendi rbf/minmax/gamma-1.0
anchors (legacy config, comparable to historical vendi runs) -> gvendi x9
grid -> vendi x9 grid (see KERNEL_GRID):
  cosine x none
  + rbf x {l2, minmax, zscore, none} x gamma {dim, median}
NOTE: gamma=1.0 saturates the RBF kernel on the C*Z-dim (5120) gvendi
gradient embeddings (verified 2026-07-13) — the gvendi anchor is a reference
point, not an informative cell.

26 query rows x 1 (data, active) setting x 1 seed = 26 runs.
"""
from argparse import ArgumentParser
from itertools import product

from launcher import ExperimentLauncher


class QuerySweepLauncher(ExperimentLauncher):
    """Sweep-specific launcher tweaks, kept local so launcher.py stays untouched:

    1. parse_product: the base class materializes the full cartesian product
       over every value list and only then filters it down to the
       joint_iteration rows. With the aligned 26-entry row lists below, that
       intermediate product is ~35^6 combinations and never finishes. Here
       every joint group is zipped into a single compound axis first, so only
       genuinely independent axes (seed, active, ...) are producted. The
       resulting run list is identical.
    2. generate_name: gvendi carries kernel/normalization settings exactly
       like vendi/bandit, but the base class only knows vendi/bandit — and its
       name-part stripping is broken anyway (see method docstring). Strip the
       inert kernel parts from baseline names and the rbf-only gamma part from
       cosine/linear names here, on the formatted name.
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
        """The base class tries to strip the vendi name parts for non-vendi
        queries by editing the naming convention, but does so after the
        convention's '.'-removal, so those replacements never match (dead
        code); it also doesn't know gvendi. Strip on the formatted name."""
        name = super().generate_name(config_dict, param_dict)
        merged = self.merge_dictionaries(config_dict, param_dict)
        query = merged.get("query")
        kernel = merged.get("query.vendi.kernel")
        norm = merged.get("query.vendi.normalization")
        gamma = merged.get("query.vendi.gamma")
        if query not in ("vendi", "gvendi", "bandit"):
            # baselines: the kernel hyperparameters are inert placeholders
            name = (
                name.replace(f"_norm-{norm}", "")
                .replace(f"_kernel-{kernel}", "")
                .replace(f"_gamma-{gamma}", "")
            )
        elif kernel in ("cosine", "linear"):
            name = name.replace(f"_gamma-{gamma}", "")  # gamma is rbf-only
        return name


# BASELINES = ["random", "entropy", "kcentergreedy", "bald", "badge", "batchbald", "variationratios"]
BASELINES = ["random", "entropy", "kcentergreedy", "bald", "badge", "variationratios"]
# use_grad sweep via the query name: vendi = feature embeddings
# (use_grad=false), gvendi = config alias setting vendi.use_grad=true
# (BADGE-style last-layer gradient embeddings).
VENDI_METHODS = ["gvendi", "vendi"]
# Kernel grid: rbf over all four normalizations x the two bandwidth heuristics
# ("dim" = 1/(2 D), "median" = 1/(2 median^2) of pairwise distances — see
# query_diversity.resolve_gamma), plus a single cosine cell. cosine is
# invariant to per-sample scaling (l2 == none), its feature-wise norms were
# pruned 2026-07-16, and the linear kernel was dropped entirely (l2+linear ==
# cosine). gamma is ignored by cosine (placeholder 1.0, stripped from names).
KERNEL_GRID = (
    [("cosine", "none", 1.0)]
    +
    [
        ("rbf", norm, gamma)
        for norm in ("l2", "minmax", "zscore", "none")
        for gamma in ("dim", "median")
    ]
)

# Aligned rows (zipped via joint_iteration): baselines get placeholder
# kernel/norm values — inert at runtime ('++' override) and stripped from names.
# Row order: baselines -> rbf/minmax/gamma-1.0 anchors (the legacy fixed-gamma
# config, directly comparable to the historical vendi runs) for gvendi then
# vendi -> the full KERNEL_GRID block per method (gvendi then vendi).
queries = list(BASELINES)
kernels = ["rbf"] * len(BASELINES)
norms = ["minmax"] * len(BASELINES)
gammas = [1.0] * len(BASELINES)
for method in VENDI_METHODS:  # legacy-config anchor runs
    queries.append(method)
    kernels.append("rbf")
    norms.append("minmax")
    gammas.append(1.0)
for method in VENDI_METHODS:  # full grid
    for kernel, norm, gamma in KERNEL_GRID:
        queries.append(method)
        kernels.append(kernel)
        norms.append(norm)
        gammas.append(gamma)

# (data, active) as one zipped axis — currently long-tail cifar10_imb
# (exp, rho=0.02) med only. To re-add settings, extend data/active AND the
# zipped data.val_size / model.weighted_loss lists together.
config_dict = {
    "model": "resnet",
    "query": queries,
    # "data": ["cifar10", "cifar10", "cifar10_imb", "cifar10_imb"],
    # "active": ["cifar10_low", "cifar10_med", "cifar10_low", "cifar10_med"],
    # "data": ["cifar10", "cifar10_imb"],
    "data": ["cifar10_imb"],
    "active": ["cifar10_med"],
    
    "optim": ["sgd_cosine"],
}

hparam_dict = {
    # "data.val_size": [250, 2500, 250, 1250],
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
    # kernel/normalization sweep — vendi and gvendi read the shared
    # query.vendi.* block (gvendi = vendi with vendi.use_grad=true)
    "query.vendi.kernel": kernels,
    "query.vendi.normalization": norms,
    "query.vendi.gamma": gammas,
}

naming_conv = (
    "{data}/active-{active}/"
    "basic_model-{model}_drop-{model.dropout_p}_aug-{data.transform_train}"
    "_acq-{query}_norm-{query.vendi.normalization}_kernel-{query.vendi.kernel}"
    "_gamma-{query.vendi.gamma}_ep-{trainer.max_epochs}"
)

joint_iteration = [
    ["data", "active", "data.val_size", "model.weighted_loss"],
    [
        "query",
        "model.dropout_p",
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
