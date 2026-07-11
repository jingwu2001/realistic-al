# Logging Improvement Plan

Clean up the wandb tracking code and make experiment names carry more information. Current state: wandb logic is spread across `src/main.py` (init, per-iteration logging, summary), `src/trainer.py` (WandbLogger wiring), and `_read_loop_metrics`.

## 1. Clean up wandb tracking code

### 1.1 Extract a `utils/wandb_utils.py` module

Move out of `main.py`:

- `init_wandb(cfg) -> wandb_run | None` — the whole block currently at `main.py:47–73` (hydra choices, git commit, session tag, `wandb.init`, `define_metric`). `main()` shrinks to one call.
- `log_al_iteration(wandb_run, i, cfg, train_timer, query_timer, eig_time, log_dir)` — the per-iteration `log_dict` construction. This is currently **duplicated** in two branches (pool-exhausted early-exit at `main.py:255–258` and the normal path at `main.py:278–287`) with slightly different keys; unify into one function so the two paths can't drift.
- `log_label_distributions(wandb_run, datamodule)` — the `init_label_dist/*` summary block (`main.py:187–196`) together with its helper closures `_label_dist`, `_dist_counts`, `_subset_targets`, which don't belong inline in `active_loop`.
- `_read_loop_metrics` — move alongside; also replace its bare `except Exception: pass` with a logged warning so silent metric loss is visible.

### 1.2 Log the full resolved config

`wandb.init(config=...)` currently hand-picks 8 fields. Log the entire resolved Hydra config instead (`OmegaConf.to_container(cfg, resolve=True)`) — this is what makes sweeps over kernel / normalization / quality / $q$ / $\alpha$ filterable in the wandb UI without code changes every time a new option is added. Keep the hand-picked fields as top-level keys if the flat names are convenient for existing dashboards.

### 1.3 Consistent metric namespaces

- `al/*` with `step_metric="al_iter"` is good — keep it and document the convention in the module docstring.
- Route the query-time diagnostics currently buried in `extra_info` (acquired/unacquired score max/min/median, `eig_time_s`) into `al/query/*` metrics instead of only `extra_info_{i}.npz` files, so score-distribution drift across rounds is visible in wandb directly.
- Add per-round metrics the gradient/qVS work will need (see `gradient_implementation_plan.md`): batch-internal Vendi of the acquired batch, and quality vs. diversity factors logged separately.

### 1.4 Small fixes

- `notes=str(cfg.trainer.wandb_notes) or None` — `str()` of an empty value is `""` only if the config is `""`; if it's `None` this becomes `"None"`. Guard properly.
- Wrap the `git rev-parse` call in try/except (fails when running outside a git checkout, e.g. on a cluster copy) with fallback `"nogit"`.
- `trainer.py:151–153` creates a `WandbLogger` per AL iteration on the same run — confirm this doesn't reset step counters; if it does, pass `resume`-style settings or reuse a single logger object.

## 2. More information in the wandb experiment name

Current: `name=f"{cfg.query.name}/seed-{cfg.trainer.seed}"` → `vendi/seed-12345`. Once kernel/normalization/quality sweeps start, every vendi run looks identical.

### 2.1 Build a method tag from query hyperparameters

For `query.name == "vendi"`, encode the settings that distinguish runs:

```
vendi-{emb}-{kernel}-{norm}-q{q}[-a{alpha}][-s{quality}]/seed-{seed}
e.g.  vendi-grad-cos-l2-q1.0-a0.5-sgradnorm/seed-12345
```

where `emb ∈ {feat, grad, fisher}`, `kernel ∈ {cos, lin, rbf}`, `norm ∈ {l2, minmax, zscore, none}`. Implement as `make_run_name(cfg)` in `wandb_utils.py` with a per-method registry (badge/bald/etc. keep their short names) so it degrades gracefully for methods without extra hyperparameters.

### 2.2 Group and tags

- Extend `group` the same way so a sweep groups by method-variant rather than lumping all vendi variants together: `{data}/{active}/{method_tag}`.
- Add the swept hyperparameters as individual tags (`kernel:cos`, `norm:l2`, …) for cross-cutting filtering; keep the existing `session_tag` (date + commit).
- Keep names short: only include a field in the name when it differs from the config default, or cap the tag length — wandb truncates long names in the sidebar. (Full config is in `wandb.config` per 1.2, so the name only needs to disambiguate.)

## 3. Order of work

1. Extract `wandb_utils.py`, unify the duplicated per-iteration logging (1.1) — pure refactor, verify with a `dry_run=False` toy run that logged keys are unchanged.
2. Full resolved config + name/group/tags from hyperparameters (1.2, 2.x) — do before starting sweeps, since names/groups can't be fixed retroactively.
3. `extra_info` → `al/query/*` metrics and new qVS diagnostics (1.3) alongside the gradient implementation.
