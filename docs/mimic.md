# MIMIC-III preprocessing for SAnD (upstream STraTS pipeline)

Reference for how raw MIMIC-III v1.4 becomes the tensor SAnD consumes, as implemented in the
upstream STraTS repo (`~/Desktop/STraTS`). This documents the *original* pipeline — see
[`mimic3_sand_integration.md`](mimic3_sand_integration.md) for how it was adapted into realistic-al.

All figures below were verified by running the pipeline and inspecting its output, not taken from
documentation.

---

## TL;DR — it is two files, not one

| | file | output | model-specific? |
|---|---|---|---|
| **Stage 1** | `src/preprocess_mimic_iii_large.py` | `data/processed/mimic_iii.pkl` (2.3 GB) | **no** — sparse/tidy, serves every model |
| **Stage 2** | `src/dataset.py` | `X: (N, 24, 387)` in RAM | **yes** — this is the SAnD-specific part |

Stage 1 runs once, offline, for hours. Stage 2 runs at the start of every training job, in ~1 minute.
The commonly-made mistake is to call stage 1 "the preprocessing" — it deliberately stops at the sparse
format, and SAnD cannot consume its output.

---

## Stage 1 — `preprocess_mimic_iii_large.py`

### Inputs: 9 CSVs, 384,062,682 event rows

**Wide / self-describing** (named columns, no dictionary needed):

| file | rows | supplies |
|---|---|---|
| `ICUSTAYS` | 61,532 | one row per ICU stay; `INTIME` = **t₀**; `ICUSTAY_ID` → `ts_id` |
| `PATIENTS` | 46,520 | `DOB` → age, `GENDER` → the 2 static features |
| `ADMISSIONS` | 58,976 | `DEATHTIME` (filter), `HOSPITAL_EXPIRE_FLAG` (**label**). Read twice. |

**EAV / event tables** (opaque `ITEMID` + value; need a dictionary):

| file | rows | variables | notes |
|---|---|---|---|
| `CHARTEVENTS` | 330,712,483 | 18 | 86% of raw rows, 58% of kept observations. Only table read in chunks. |
| `LABEVENTS` | 27,854,055 | 50 | **no `ICUSTAY_ID` column** |
| `INPUTEVENTS_CV` | 17,527,935 | 43 | point events, `AMOUNT` not `VALUENUM` |
| `OUTPUTEVENTS` | 4,349,218 | 10 | no `VALUENUM` column at all |
| `INPUTEVENTS_MV` | 3,618,991 | 54 | **interval-valued**; read twice (2nd for `PATIENTWEIGHT`) |

**Dictionary:** `D_ITEMS` (12,487 rows). Used *only* to keyword-search output item labels.
`D_LABITEMS` is never read — lab ITEMIDs are hardcoded.

Not used: `NOTEEVENTS`, `PRESCRIPTIONS`, `DIAGNOSES_ICD`, `PROCEDUREEVENTS_MV`. No diagnosis codes,
no notes, no medications-as-ordered.

### Step 1 — cohort: adults

`AGE = INTIME.year - DOB.year`, keep `>= 18`. (Ages are a year-difference only, so ±1 year.)

Patients over 89 have `DOB` shifted back ~300 years for HIPAA. **Stage 1 does not fix this**; it is
patched in stage 2 (`dataset.py:26`, `Age > 200 → 91.4`).

### Step 2 — ITEMID harmonization (~80% of the script, lines 57–738)

This is the part usually glossed as "combine the variables". It is not a concatenation — it is a
hand-curated mapping of **402 ITEMIDs → 131 variable names**, solving three problems per variable:

**(a) Many ITEMIDs, one concept.** CareVue (< 100000) and MetaVision (>= 220000) use disjoint ID
spaces for the same measurement, plus multiple sites/methods. Diastolic BP alone is 14 ITEMIDs
(`Arterial BP [Diastolic]` carevue, `Arterial Blood Pressure diastolic` metavision, …) all collapsing
to `DBP`.

*The variable names (`HR`, `DBP`, …) exist nowhere in MIMIC.* They are invented by
`ch_hr['NAME'] = 'HR'`. There are three naming layers: `ITEMID` (in the data) → `LABEL` (MIMIC's own
name, in `D_ITEMS`) → `NAME` (the script's vocabulary). Keyword-matching `LABEL` cannot replace the
curated lists: searching "heart rate" also returns `Heart Rate Alarm - High` (a monitor *setting*), and
FiO2's label is `Inspired O2 Fraction` (the string "FiO2" is only in `ABBREVIATION`).

Verified: no ITEMID appears in two different variable lists, so nothing is double-counted.

**(b) Inconsistent units.** °F→°C, lb→kg, in→cm, mcg→mg, L→mL, applied *before* merging. FiO2 is the
interesting case — it is charted as both `0.40` and `40` under the *same* ITEMID, so the conversion is
keyed on the value (`> 1.0` ⇒ percentage), not the ITEMID.

**(c) Categorical values.** `"Normal <3 Seconds"` → 0, `"INTUBATED"` → 1, `M`/`F` → 0/1. Doubled-up
string comparisons exist because the two ICU systems spell options differently.

**Two different outlier policies** — the subtlety most readers miss:

| policy | applied to | out-of-range value becomes | rationale |
|---|---|---|---|
| **drop row** | measurements (vitals, labs) | deleted | an impossible sodium is noise |
| **median-replace** | inputs & outputs (fluids, drugs) | median of in-range values | an implausible volume still means **an administration happened**; deleting the row erases the event. Since the model gets a missingness mask, "an event occurred" is real signal. |

Bounds are physiological (pH ∈ [0,14]), chosen to catch data-entry errors — *not* percentile clipping,
which would delete the extreme values that predict mortality.

### Step 3 — MetaVision interval splitting (lines 405–432)

`INPUTEVENTS_MV` records infusions as `(STARTTIME, ENDTIME, AMOUNT)` spans. Spans > 1h are exploded
into one event per hour carrying `60*amount/duration`, plus a partial-hour remainder.

* assumes a **uniform infusion rate** (real rates get titrated) — the main modelling approximation
  hidden in preprocessing
* hour markers land at the **end** of each hour, so a drug is never recorded before it was delivered
* total amount is conserved exactly
* the `.iterrows()` loop here dominates runtime

### Step 4 — assign labs to a stay, convert to relative time

Lab rows have no `ICUSTAY_ID`. For each admission, build its list of `(stay, INTIME, OUTTIME)` windows
and assign the stay whose window contains the lab's timestamp. Events matching no window (ward, ED,
post-discharge) are **dropped**.

Then `minute = (CHARTTIME - INTIME) // 60` seconds. Negative values are normal (pre-ICU labs) and
survive stage 1; stage 2 filters them.

MIMIC's absolute dates are randomly shifted per patient, so only within-patient differences are
meaningful — this step is what makes patients comparable.

### Step 5 — cohort filters (the supervised cohort)

Verified funnel:

```
52,871   adult stays with >=1 extracted event
45,031   filter 1: ICU stay lasted >= 24h        -7,840   <- 97% of the loss
44,823   filter 2: alive at the 24h mark           -208
44,812   filter 3: >=1 event in first 24h           -11
```

**Filters 1 and 2 are the anti-leakage guarantee**, and are the most important two lines in the file.
Without filter 1, an early-discharged patient contributes a mostly-empty window whose emptiness leaks
"survived". Without filter 2, a patient who dies at hour 3 is in the data and the model learns
"observations stop ⇒ death". Filter 3 is nearly a no-op.

The 8,059 excluded stays cluster just under the cutoff (median LOS 20.0h, IQR 16.7–22.3h) — routine
post-op overnight observations. They are **kept in the pickle** as `unsup_icustays` for STraTS's
self-supervised forecasting pretraining. SAnD never uses them.

> Gotcha when reproducing filter 3: it runs *before* `Age`/`Gender` are injected at `minute=0`. Apply
> it to the finished pickle and every stay trivially passes.

### Step 6 — statics, label, split

* `Age`/`Gender` appended as pseudo-events at `minute = 0` (pulled back out in stage 2)
* label = `HOSPITAL_EXPIRE_FLAG`, merged `on='HADM_ID'`
* split on **`SUBJECT_ID`**, `np.random.seed(0)`, **0.64 / 0.16 / 0.20**

Subject-level splitting is required, not cosmetic: 653 fatal admissions have multiple ICU stays
contributing 1,449 identically-labelled correlated rows. Stay-level splitting would straddle them.

**Label semantics.** `HOSPITAL_EXPIRE_FLAG` is admission-scoped, and internally consistent (5,854
flagged; all have `DEATHTIME`; all have `DISCHARGE_LOCATION = DEAD/EXPIRED`; zero unflagged rows have
a `DEATHTIME`). But it is *in-hospital*, not ICU, mortality:

* **20% of deaths occur after the patient left the ICU** (median 2.5 days later)
* `PATIENTS.EXPIRE_FLAG` (died *ever*, incl. post-discharge via the SSN index) is 15,759 patients vs
  5,854 in-hospital deaths — **9,946 patients died after discharge and are labelled 0**

### Step 7 — aggregate and save

`groupby(['ts_id','minute','variable']).agg({'value':'mean'})` averages exact same-*minute*
duplicates (e.g. a CareVue and a MetaVision ITEMID both mapping to `HR`). Same-*hour* collisions are
not touched here — stage 2 resolves those, destructively.

```python
pickle.dump([events, oc, train_ids, valid_ids, test_ids], ...)
```

| element | shape | |
|---|---|---|
| `events` | 78,440,995 × 5 | **the main dataframe**: `ts_id, minute, variable, value, TABLE` |
| `oc` | 52,871 × 4 | per-stay IDs + `in_hospital_mortality` |
| `train/valid/test_ids` | 28,790 / 7,144 / 8,878 | supervised cohort membership |

Sorted lexicographically by `(ts_id, minute, variable)` — an undocumented property that stage 2
silently depends on (see below).

**Verified output stats:** 52,871 stays · **131** variables · mortality **0.123** · 19,977,118 events
inside a 0–24h window · mean 378 obs/stay in-window.

Provenance (`TABLE` column, in-window rows):

| source | variables | rows | share |
|---|---|---|---|
| `CHARTEVENTS` | 18 | 11.56M | 57.9% |
| `INPUTEVENTS_CV`+`_MV` | 55 | 3.52M | 17.6% |
| `LABEVENTS` | 50 | 3.42M | 17.1% |
| `OUTPUTEVENTS` | 10 | 0.95M | 4.8% |
| statics / `PATIENTWEIGHT` | 2 | 0.11M | 0.5% |

Note the inversion: CHARTEVENTS supplies 14% of variables but 58% of observations (vitals are charted
every few minutes; labs twice a day).

---

## Stage 2 — `dataset.py` (the SAnD-specific part)

One class, everything in `__init__`, everything materialized in RAM. No `DataLoader`.

### Shared prologue (all models)

| lines | step |
|---|---|
| 14–20 | **`train_frac` / `run` subsetting** — train and val only |
| 25 | **the 24h window**: `0 <= minute <= 1440` |
| 26 | `Age > 200 → 91.4` |
| 29–33 | drop variables absent from the training split |
| 34–49 | `ts_id → ts_ind` (0…N−1), build `y`, `N` |
| 62 | `pos_class_weight = neg/pos` = **7.04** |
| 74 | `get_static_data` → `demo (N,2)`; **131 → 129 variables** |

**Why subsetting exists:** it is the experiment, not an optimization. STraTS claims pretraining helps
most under label scarcity, so `run_main.sh` sweeps `train_frac ∈ {0.5,…,0.1}` × 10 folds. Pretraining
has **no** `--train_frac` — it always sees everything, including the 8,059 unsupervised stays.
Validation *is* subsetted (a val set is labelled data; keeping it full would cheat); **test never is**,
so all runs score on the same 8,878 stays.

> Caveat: folds are contiguous windows at evenly spaced offsets, so they overlap — verified **90.0%**
> between consecutive folds at `train_frac=0.5`, **1.2%** at 0.1. Reported ± is seed-and-window
> jitter, not independent resampling, and understates variance most at large fractions.

### The SAnD branch (lines 108–137)

```python
data['minute'] = data['minute'].apply(lambda x: max(1,int(np.ceil(x/60)))-1)   # hour bins, T=24
values[row.ts_ind, tstep, vind] = row.value      # scatter -- ASSIGNS, does not accumulate
obs[row.ts_ind, tstep, vind] = 1
delta[:,t,:] = obs[:,t,:]*0 + (1-obs[:,t,:])*(1+delta[:,t-1,:])
...
means = (values[tr]*obs[tr]).sum(axis=(0,1)) / obs[tr].sum(axis=(0,1))   # observed cells only
values = values*obs + (1-obs)*means                                      # impute
values = (values - values[tr].mean(...)) / values[tr].std(...)           # normalize
self.X = np.concatenate((values, obs, delta), axis=-1)                   # (N, 24, 3V)
```

Bin `t` covers `(60t, 60t+60]`; the `max(1, …)` guard puts minute 0 in bin 0.

**Three channels:**

| channel | meaning |
|---|---|
| `values` | the measurement, mean-filled where missing |
| `obs` | 1 if genuinely observed — the only thing distinguishing real from imputed |
| `delta` | hours since last observation, ÷24. From GRU-D. |

The `delta` recurrence is a branchless if/else: `obs=1` ⇒ both terms vanish ⇒ 0; `obs=0` ⇒
`delta[t-1]+1`. The `obs*0` term is dead code.

Note the model is **~88% imputed** (mean 378 observations against 24 × 129 = 3,096 cells), which is
why `obs` carries real signal — ordering hourly blood gases means the clinician is worried.

**Normalization is train-only** in all four places that compute statistics (lines 62, 129–130,
133–134, 231–232). This matters because train/val/test all live in **one array** as contiguous blocks
(`X[0:n_tr]` train, then val, then test) — `values.mean()` instead of `values[train_ind].mean()` would
look almost identical, run fine, and silently leak.

### Output

| attribute | shape |
|---|---|
| `X` | `(N, 24, 387)` float64 — values ‖ mask ‖ delta |
| `demo` | `(N, 2)` |
| `y` | `(N,)` |
| `splits` | dict of row indices, incl. `eval_train` = first 2000 train (not random) |

Also written onto `args`: `V=129`, `T=24`, `D=2`, `pos_class_weight=7.04`.

`get_batch` is a plain slice — no padding, unlike the STraTS/GRU-D branches.

### How much data a single run actually touches

| | `train_frac=0.5` | `train_frac=0.1` |
|---|---|---|
| stays loaded into `X` | 26,845 | 12,472 |
| supervised stays *not* loaded | 17,967 | 32,340 |
| events in `X` | 10,374,002 (**13.2%** of pickle) | 4,960,018 (**6.3%**) |
| `X` memory (float64) | 2.0 GB | 0.9 GB |

`X` is float64 but cast to float32 per batch — building it as float32 would halve the footprint for free.

---

## Known quirks

**Within-hour collisions discard 20% of observations.** `values[...] = row.value` assigns, so one
value per `(stay, hour, variable)` cell survives. Because the pickle is sorted by
`(ts_id, minute, variable)` and every step before the scatter loop is an order-preserving boolean
filter, the survivor is deterministically the **chronologically last observation of the hour** — a
defensible last-observation-carried rule. But:

| | |
|---|---|
| observations in window | 19,871,376 |
| cells receiving >1 observation | **12.5%** |
| observations overwritten | **3,967,319 (20.0%)** |

Worst hit are exactly the variables SAnD leans on: `Weight` −46.9%, `Solution` −37.2%,
`O2 Saturation` −32.8%, `RR` −31.8%, `HR` −30.2%, `SBP` −27.3%. Using `mean` would be a one-line
change. The rule also depends on an *undocumented* sort order — the STraTS branch does
`data.sample(frac=1)` at line 78, and any similar reordering upstream would silently change the result.

**`delta` initialization is inverted.** `delta[:,0,:] = obs[:,0,:]` sets `delta=1` where the variable
*was* observed at hour 0 — backwards from the recurrence below it. A variable observed in hour 0 then
reads one hour too stale for its entire subsequent missing run.

**One example per stay — no sliding windows.** A 25-hour stay yields exactly one training row
(hours 0–24); a 20-day stay also yields one. `N` is the number of stays. **75% of the pickle's events
fall outside any 24h window and are never used.**

**Age fix lives in stage 2 only.** Anything consuming the pickle directly must apply
`Age > 200 → 91.4` itself.

**Cosmetic:** `import yaml` unused; `'Vacomycin'` typo is a real variable name in the output; a
variable literally named `Unknown` (ITEMID 30140, 108,568 rows) is an unidentified CareVue fluid kept
as an anonymous feature; the `TABLE`-joining helper at lines 843–848 has an inverted length check that
never fires.

**Pickle compatibility:** written under pandas 1.4.3. Loading under pandas 2.x raises
`ModuleNotFoundError: No module named 'pandas.core.indexes.numeric'`. Loads in 3.0 s, ~4.4 GB RSS.
(`memory_usage(deep=True)` reports 11.8 GB — wrong here, since only 131 distinct strings are shared
across 78M rows.)
