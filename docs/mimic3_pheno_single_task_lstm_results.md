# MIMIC-III phenotype single-task: LSTM AL results

Snapshot of the LSTM single-task phenotyping sweep, pulled from wandb on
**2026-10-07 ~12:40 (+0800)**. 75 of 84 runs had finished. Only plain `vendi`
(q=1) cells are still pending, so every other number below is final. Re-run the
pull (see [Reproducing](#reproducing-the-numbers)) once the `vendi` runs finish.

## Setup

| | |
|---|---|
| Launcher | [launchers/sweep_exp_mimic3_pheno_single_task_lstm.py](../launchers/sweep_exp_mimic3_pheno_single_task_lstm.py), launched 2026-10-06 20:19 (+0800) at commit `3bb8f11fd`. The working tree differed from that commit: the cohort sweep was reduced to `data.single_stay_only=[True]` and the queries were reordered. |
| Model | `model=lstm`, the Harutyunyan et al. 2017 benchmark LSTM ([models/networks/bayesian_lstm.py](../src/models/networks/bayesian_lstm.py)): dim 256, depth 1, dropout 0.3. Same (24, 387) SAND input tensor plus the demographic embedding (`use_static=true`). |
| Training recipe | Benchmark recipe: Adam lr 1e-3, weight decay 0, batch 8, no gradient clipping, constant lr, 50 epochs, fp32, deterministic, BCE with `pos_weight = #neg/#pos`. |
| Tasks | `data.multitask=false`. **t08** = congestive heart failure, nonhypertensive (prevalence 23.9% train / 23.0% test). **t11** = diabetes mellitus without complication (19.0% / 18.9%). |
| Cohort | Single-stay only (`data.single_stay_only=true`; 21,288 train / 5,321 test stays). |
| Budgets | `active=mimic3_med` with `num_iter=10`, (start, step) = (100, 100) → 100…1000 labels, and (500, 500) → 500…5000 labels. The `acq_size` value below is the step. |
| Methods | random, entropy, bald, badge, kcentergreedy, vendi (q=1), vendi-q2. Both vendi variants use the pinned single-task block: feature embedding, rbf kernel, gamma = median heuristic, l2 normalisation. |
| Seeds | 12345, 12346, 12347 |
| wandb | Project `realistic-al`. `config.data_name` = `mimic3_pheno_t08_lstm` / `mimic3_pheno_t11_lstm`. The SAND runs it is compared against are `mimic3_pheno_t08` / `mimic3_pheno_t11`. |

**Metrics.** Headline = AUBC of test AUPRC (`al_summary/aubc_test_auprc`), the
area under the AUPRC-vs-labels learning curve, normalised by the label range.
AUPRC is the right minority-sensitive metric here, because a random classifier
scores roughly the prevalence (0.23 / 0.19). Test AUROC AUBC is reported
alongside it.

**Pairing check.** For each (task, budget, seed), all methods start from an
identical round-0 test AUPRC (same initial labelled set and training path), and
there are no duplicate runs. So the Δ-vs-random columns are true per-seed paired
differences.

## Results: mean AUBC over 3 seeds

How to read the tables:
- **Δ vs random** is the mean of the per-seed differences against `random`.
- **seeds > random** counts how many of the 3 seeds beat their paired `random` run.
- **rank** is by mean AUBC AUPRC, among the methods with results.
- `*` marks a paired t-test with p < 0.05 (n = 3, uncorrected). Treat it as a hint, not a verdict.
- **Bold** = best mean in that column.

#### Task 8, acq_size=100

| method | AUBC AUPRC (mean ± sd) | Δ vs random | seeds > random | rank | AUBC AUROC (mean ± sd) | Δ vs random | seeds > random |
|---|---|---|---|---|---|---|---|
| random | 0.4832 ± 0.0057 | — | — | 5/7 | 0.7586 ± 0.0040 | — | — |
| entropy | 0.4915 ± 0.0097 | +0.0083 | 2/3 | 2/7 | **0.7696 ± 0.0036** | +0.0110 | 3/3 |
| bald | 0.4848 ± 0.0013 | +0.0016 | 1/3 | 3/7 | 0.7609 ± 0.0040 | +0.0023 * | 3/3 |
| badge | **0.4919 ± 0.0031** | +0.0087 | 3/3 | 1/7 | 0.7665 ± 0.0054 | +0.0079 * | 3/3 |
| kcentergreedy | 0.4772 ± 0.0173 | -0.0060 | 1/3 | 7/7 | 0.7556 ± 0.0104 | -0.0030 | 1/3 |
| vendi | 0.4828 ± 0.0230 | -0.0004 | 2/3 | 6/7 | 0.7586 ± 0.0140 | -0.0000 | 2/3 |
| vendi-q2 | 0.4838 ± 0.0167 | +0.0006 | 2/3 | 4/7 | 0.7601 ± 0.0131 | +0.0015 | 2/3 |

#### Task 8, acq_size=500

| method | AUBC AUPRC (mean ± sd) | Δ vs random | seeds > random | rank | AUBC AUROC (mean ± sd) | Δ vs random | seeds > random |
|---|---|---|---|---|---|---|---|
| random | 0.5392 ± 0.0124 | — | — | 6/6 | 0.7923 ± 0.0045 | — | — |
| entropy | 0.5437 ± 0.0055 | +0.0045 | 2/3 | 5/6 | 0.7944 ± 0.0023 | +0.0021 | 1/3 |
| bald | 0.5486 ± 0.0072 | +0.0093 | 2/3 | 2/6 | **0.7960 ± 0.0021** | +0.0037 | 2/3 |
| badge | 0.5439 ± 0.0090 | +0.0047 | 2/3 | 4/6 | 0.7937 ± 0.0030 | +0.0014 | 1/3 |
| kcentergreedy | **0.5506 ± 0.0050** | +0.0113 | 3/3 | 1/6 | 0.7922 ± 0.0044 | -0.0001 | 1/3 |
| vendi | *pending* (running, round 5/9) | | | | *pending* | | |
| vendi-q2 | 0.5462 ± 0.0054 | +0.0070 | 2/3 | 3/6 | 0.7904 ± 0.0018 | -0.0019 | 1/3 |

#### Task 11, acq_size=100

| method | AUBC AUPRC (mean ± sd) | Δ vs random | seeds > random | rank | AUBC AUROC (mean ± sd) | Δ vs random | seeds > random |
|---|---|---|---|---|---|---|---|
| random | 0.2960 ± 0.0107 | — | — | 3/6 | 0.6632 ± 0.0097 | — | — |
| entropy | 0.2964 ± 0.0059 | +0.0004 | 2/3 | 2/6 | 0.6627 ± 0.0128 | -0.0005 | 1/3 |
| bald | 0.2843 ± 0.0131 | -0.0117 | 1/3 | 6/6 | 0.6503 ± 0.0076 | -0.0130 | 0/3 |
| badge | **0.3006 ± 0.0110** | +0.0046 | 1/3 | 1/6 | **0.6633 ± 0.0084** | +0.0001 | 1/3 |
| kcentergreedy | 0.2943 ± 0.0050 | -0.0017 | 1/3 | 4/6 | 0.6552 ± 0.0056 | -0.0081 | 1/3 |
| vendi | *pending* (not started) | | | | *pending* | | |
| vendi-q2 | 0.2922 ± 0.0092 | -0.0038 | 1/3 | 5/6 | 0.6560 ± 0.0053 | -0.0072 | 1/3 |

#### Task 11, acq_size=500

| method | AUBC AUPRC (mean ± sd) | Δ vs random | seeds > random | rank | AUBC AUROC (mean ± sd) | Δ vs random | seeds > random |
|---|---|---|---|---|---|---|---|
| random | 0.3649 ± 0.0053 | — | — | 3/6 | **0.7215 ± 0.0046** | — | — |
| entropy | **0.3677 ± 0.0034** | +0.0028 | 2/3 | 1/6 | 0.7213 ± 0.0039 | -0.0002 | 1/3 |
| bald | 0.3633 ± 0.0054 | -0.0016 | 1/3 | 4/6 | 0.7176 ± 0.0043 | -0.0039 | 1/3 |
| badge | 0.3661 ± 0.0095 | +0.0012 | 1/3 | 2/6 | 0.7199 ± 0.0064 | -0.0016 | 1/3 |
| kcentergreedy | 0.3538 ± 0.0058 | -0.0111 * | 0/3 | 5/6 | 0.7103 ± 0.0058 | -0.0112 * | 0/3 |
| vendi | *pending* (not started) | | | | *pending* | | |
| vendi-q2 | 0.3483 ± 0.0030 | -0.0166 | 0/3 | 6/6 | 0.7066 ± 0.0008 | -0.0149 * | 0/3 |

#### Summary matrix (AUBC AUPRC, mean of 3 seeds)

| method | t08 / 100 | t08 / 500 | t11 / 100 | t11 / 500 |
|---|---|---|---|---|
| random | 0.4832 | 0.5392 | 0.2960 | 0.3649 |
| entropy | 0.4915 | 0.5437 | 0.2964 | **0.3677** |
| bald | 0.4848 | 0.5486 | 0.2843 | 0.3633 |
| badge | **0.4919** | 0.5439 | **0.3006** | 0.3661 |
| kcentergreedy | 0.4772 | **0.5506** | 0.2943 | 0.3538 |
| vendi | 0.4828 | *pending* | *pending* | *pending* |
| vendi-q2 | 0.4838 | 0.5462 | 0.2922 | 0.3483 |

#### Final-round test AUPRC (1000 labels for acq 100, 5000 labels for acq 500)

| method | t08 / 100 | t08 / 500 | t11 / 100 | t11 / 500 |
|---|---|---|---|---|
| random | **0.5294** | 0.5644 | 0.3279 | 0.3907 |
| entropy | 0.5102 | 0.5744 | 0.3179 | 0.4041 |
| bald | 0.5006 | 0.5703 | 0.3179 | 0.4025 |
| badge | 0.5174 | **0.5823** | **0.3323** | **0.4055** |
| kcentergreedy | 0.5103 | 0.5817 | 0.3190 | 0.3867 |
| vendi | 0.5087 | *pending* | *pending* | *pending* |
| vendi-q2 | 0.5163 | 0.5706 | 0.3088 | 0.3822 |

## LSTM vs SAND (same tasks, cohort, budgets, seeds)

The SAND runs are the single-stay cells of the 2026-09-29 → 10-06 SAND
single-task sweep. Both models share the AL protocol and the vendi block; only
the network and its training recipe differ.

| setting | random AUBC AUPRC: LSTM | random AUBC AUPRC: SAND | best method: LSTM | best method: SAND | vendi-q2 Δ: LSTM | vendi-q2 Δ: SAND | method spread: LSTM | method spread: SAND |
|---|---|---|---|---|---|---|---|---|
| t08 / 100 | 0.4832 | 0.4684 | badge | bald | +0.0006 | -0.0183 | 0.0147 | 0.0391 |
| t08 / 500 | 0.5392 | 0.5228 | kcentergreedy | badge | +0.0070 | +0.0091 | 0.0113 | 0.0185 |
| t11 / 100 | 0.2960 | 0.2853 | badge | badge | -0.0038 | -0.0111 | 0.0163 | 0.0145 |
| t11 / 500 | 0.3649 | 0.3433 | entropy | badge | -0.0166 | -0.0259 | 0.0178 | 0.0304 |

"Method spread" is max minus min mean AUBC AUPRC over the methods present for
*both* models in that setting. SAND has no finished `entropy` run for t11 / 500
(all three failed or crashed), so that column excludes it.

Spearman correlation of the method ranking, LSTM vs SAND:

| setting | AUBC AUPRC | AUBC AUROC |
|---|---|---|
| t08 / 100 | +0.79 | +0.79 |
| t08 / 500 | +0.14 | +0.77 |
| t11 / 100 | +0.83 | +0.60 |
| t11 / 500 | +1.00 | +1.00 |

## Takeaways

1. **The LSTM beats SAND.** Its random baseline is higher in all four settings,
   by +0.011 to +0.022 AUBC AUPRC.
2. **Methods separate even less with the LSTM.** The spread between methods is
   0.011–0.018 AUBC AUPRC, the same size as random's seed-to-seed sd
   (0.005–0.012). Only two effects hold on all 3 seeds *and* reach p < 0.05:
   - On t08 / 100, badge and bald beat random on AUROC (+0.008 and +0.002).
   - On t11 / 500, kcentergreedy (AUPRC and AUROC) and vendi-q2 (AUROC) lose to
     random. vendi-q2's AUPRC loss (−0.017, 0/3 seeds) just misses p < 0.05.
3. **Rankings carry over from SAND.** Rank correlations are +0.6 to +1.0, except
   AUPRC on t08 / 500 (+0.14), where kcentergreedy jumps from 6th to 1st.
   Uncertainty-type methods (badge, entropy, bald) are at or near the top in
   every setting, as with SAND.
4. **Diversity methods still don't help.**
   - vendi-q2 is about equal to random on t08 (+0.001 / +0.007).
   - It is below random on t11, worst at the larger budget (−0.017 AUPRC, 0/3 seeds).
   - With the LSTM it loses less than with SAND (t08 / 100: +0.001 vs −0.018).
   - The sign is the same as with SAND, so the SAND finding that diversity
     methods fall below random on single-task phenotyping is not a SAND artefact.
5. **Final-round numbers don't simply track AUBC.** At 1000 labels on t08, random
   has the best final AUPRC even though badge and entropy have the higher AUBC.
   AL is ahead early in the curve, not at the end.

## Caveats

- n = 3 seeds; the p-values are uncorrected across 6 methods × 8 tables.
- Plain `vendi` (q=1) is missing in 3 of 4 settings. Its two pending cells and
  one not-yet-started setting will be filled in once the remaining 9 runs finish.
- The LSTM runs launched at `3bb8f11fd` ("default secular to true"). That means
  the LSTM `vendi` (q=1) uses the secular eigen-solver, while the earlier SAND
  `vendi` runs used dense. The two agree to within 1e-12 except at exact score
  ties, and none have been seen for feat + rbf-median + l2. vendi-q2 doesn't use
  this solver path.
- The launcher's inline comment `"query.vendi.gamma": "median",  # 1/(2 D)` is
  wrong: `median` is the median heuristic 1/(2·median²), and `dim` is 1/(2D).
  The runs used the median heuristic, as the config shows.

## Reproducing the numbers

```python
import wandb
api = wandb.Api()
runs = api.runs("jingwusince01-national-taiwan-uni/realistic-al", filters={
    "config.data_name": {"$in": ["mimic3_pheno_t08_lstm", "mimic3_pheno_t11_lstm"]},
    "state": "finished",
})
# group by (config["data/task_id"], config["acq_size"], config["query_name"]) and
# average run.summary["al_summary/aubc_test_auprc"] / ["al_summary/aubc_test_auroc"]
# over config["seed"]. For the SAND comparison use data_name mimic3_pheno_t08 /
# mimic3_pheno_t11 with config.data.single_stay_only = True.
```

No date filter is needed. All LSTM runs come from one launch. The SAND
single-task runs were launched in several waves (2026-09-29 → 10-06), but for
each (task, budget, cohort, seed) every wave has an identical round-0 test
AUPRC, so they are comparable. Keep `state=finished`: SAND t11 has failed or
crashed duplicates.
