# PhysioNet 2012 (P12) Dataset

## Overview

The P12 dataset is derived from the [PhysioNet/Computing in Cardiology Challenge 2012](https://www.physionet.org/content/challenge-2012/1.0.0/) and contains ICU time-series data from three patient sets (A, B, C). The task is **in-hospital mortality prediction** — a binary classification problem.

## Dataset Statistics

| Property | Value |
|---|---|
| Total patients | 11,988 |
| Survived (label 0) | 10,281 (85.8%) |
| Died (label 1) | 1,707 (14.2%) |
| Class imbalance ratio | ~6:1 |

## Label Distribution

| Label | Class | Count | Percentage |
|---|---|---|---|
| 0 | Survived | 10,281 | 85.8% |
| 1 | In-hospital death | 1,707 | 14.2% |

### Per-Split Distribution (Split 1, 80/10/10)

| Split | Samples | Positive (died) | Positive rate |
|---|---|---|---|
| Train | 9,590 | 1,381 | 14.4% |
| Val | 1,199 | 172 | 14.3% |
| Test | 1,199 | 154 | 12.8% |

Five pre-defined stratified splits are available in `datasets/P12data/splits/`.

## Features

### Time-Series Features (36 clinical variables)

Irregular time-series sampled over a 48-hour ICU stay, padded to a maximum of 215 timesteps. Timestamps are stored in minutes from ICU admission.

| # | Feature | Description |
|---|---|---|
| 1 | ALP | Alkaline phosphatase |
| 2 | ALT | Alanine aminotransferase |
| 3 | AST | Aspartate aminotransferase |
| 4 | Albumin | Serum albumin |
| 5 | BUN | Blood urea nitrogen |
| 6 | Bilirubin | Total bilirubin |
| 7 | Cholesterol | Serum cholesterol |
| 8 | Creatinine | Serum creatinine |
| 9 | DiasABP | Invasive diastolic arterial blood pressure |
| 10 | FiO2 | Fraction of inspired oxygen |
| 11 | GCS | Glasgow coma score |
| 12 | Glucose | Serum glucose |
| 13 | HCO3 | Bicarbonate |
| 14 | HCT | Hematocrit |
| 15 | HR | Heart rate |
| 16 | K | Potassium |
| 17 | Lactate | Serum lactate |
| 18 | MAP | Invasive mean arterial pressure |
| 19 | MechVent | Mechanical ventilation (binary) |
| 20 | Mg | Magnesium |
| 21 | NIDiasABP | Non-invasive diastolic arterial BP |
| 22 | NIMAP | Non-invasive mean arterial pressure |
| 23 | NISysABP | Non-invasive systolic arterial BP |
| 24 | Na | Sodium |
| 25 | PaCO2 | Partial pressure of CO₂ in arterial blood |
| 26 | PaO2 | Partial pressure of O₂ in arterial blood |
| 27 | Platelets | Platelet count |
| 28 | RespRate | Respiration rate |
| 29 | SaO2 | O₂ saturation in hemoglobin |
| 30 | SysABP | Invasive systolic arterial blood pressure |
| 31 | Temp | Body temperature |
| 32 | TroponinI | Troponin-I |
| 33 | TroponinT | Troponin-T |
| 34 | Urine | Urine output |
| 35 | WBC | White blood cell count |
| 36 | pH | Arterial blood pH |

**Sequence length:** min=1, max=214, mean=73.9, median=71.0 observations per patient.

### Static Features (9 dimensions, one-hot encoded)

| Feature | Type | Stats / Notes |
|---|---|---|
| Age | Continuous | mean=64.5, std=17.2, no missing |
| Gender=0 | Binary | 43.5% (female) |
| Gender=1 | Binary | 55.4% (male) |
| Height | Continuous | mean=164.3 cm, std=33.5, **46.9% missing** |
| ICUType=1 | Binary | 14.9% — Coronary Care Unit |
| ICUType=2 | Binary | 20.6% — Cardiac Surgery Recovery Unit |
| ICUType=3 | Binary | 35.1% — Medical ICU |
| ICUType=4 | Binary | 27.7% — Surgical ICU |
| Weight | Continuous | mean=81.0 kg, std=26.0, 8.3% missing |

Missing static values (stored as -1) are replaced with 0 after z-score normalization.

## ICU Type Breakdown

| ICU Type | Name | Count | Percentage |
|---|---|---|---|
| 1 | Coronary Care Unit (CCU) | 1,782 | 14.9% |
| 2 | Cardiac Surgery Recovery Unit (CSRU) | 2,474 | 20.6% |
| 3 | Medical ICU (MICU) | 4,203 | 35.1% |
| 4 | Surgical ICU (SICU) | 3,325 | 27.7% |

## Data Format

### Transformer Format (`P12TransformerDataset`)

Each sample is a 4-tuple `((ts, static, times, length), label)`:

| Field | Shape | Description |
|---|---|---|
| `ts` | `(215, 36)` | Normalized time-series (0 where unobserved) |
| `static` | `(9,)` | Normalized static/demographic features |
| `times` | `(215,)` | Timestamps in minutes from ICU admission |
| `length` | scalar `int64` | Actual number of observed timesteps |
| `label` | scalar | 0 = survived, 1 = died |

### GRU-D Format (`P12Dataset`)

Each sample is a 3-channel tensor of shape `(3, 36, 49)`:

| Channel | Content |
|---|---|
| 0 | Normalized feature values (X) |
| 1 | Observation mask (M) |
| 2 | Time-since-last-observation in hours (Δ) |

## Normalization

- **Time-series:** Per-feature mean and std computed on observed (non-zero) values from the training set only. Unobserved positions remain 0.
- **Static:** Per-feature z-scoring ignoring missing values (< 0); missing values set to 0 post-normalization.

## Class Imbalance Handling

Two strategies are supported via config:

| Config | Strategy |
|---|---|
| `p12_transformer.yaml` | `balanced_sampling: true` — `WeightedRandomSampler` for balanced mini-batches during training |
| `p12_transformer_baleval.yaml` | `balanced_test_val: true` — 50/50 test and validation splits for unbiased evaluation |

Active learning pool initialization also supports balanced seeding (`p12_med_bal.yaml`) versus natural-imbalance seeding (`p12_med.yaml`).

## File Locations

```
datasets/P12data/
├── processed_data/
│   ├── PTdict_list.npy          # Per-patient dicts (ts, static, times, length)
│   ├── arr_outcomes.npy         # Outcome labels (col -1 = mortality)
│   ├── ts_params.npy            # Names of 36 time-series features
│   ├── extended_static_params.npy  # Names of 9 static features
│   └── static_params.npy        # Original 5 static parameter names
├── splits/
│   ├── phy12_split{1..5}.npy    # 5 pre-defined train/val/test splits (80/10/10)
│   └── phy12_split_subset{1..5}.npy
└── process_scripts/
    ├── ParseData.py
    ├── IrregularSampling.py
    └── Generate_splitID.py
```

## Source

PhysioNet/CinC Challenge 2012 — "Predicting in-hospital mortality of ICU patients"  
Processed following the [Raindrop](https://github.com/mims-harvard/Raindrop) pipeline.
