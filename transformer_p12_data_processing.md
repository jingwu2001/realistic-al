# Transformer Encoder: Training Data Processing on P12

This document describes the full data pipeline that feeds `BayesianTransformerModel2` (registered as `transformer2`) in experiments launched by `launchers/exp_p12_transformer.py`.

> **Note:** An earlier version of this document described the original Raindrop `TransformerModel2` baseline
> (`code/baselines/Transformer_baseline.py` + `code/baselines/models.py`). The current document
> reflects the actual implementation in `src/`.

---

## 1. Raw Data Structure

The P12 dataset is stored in two pre-processed numpy files loaded in [p12_dataset.py:96-97](src/data/p12_dataset.py#L96-L97):

```
<root>/PTdict_list.npy   # list of per-patient dicts
<root>/arr_outcomes.npy  # outcome labels
```

Each element of `PTdict_list` is a dictionary with:

| Key | Shape | Description |
|---|---|---|
| `arr` | `(215, 36)` | Time-series observations. `T ≤ 215` time steps, 36 clinical variables. Missing values are **0**. |
| `time` | `(215, 1)` | Observation timestamps in **minutes** since ICU admission (0-padded). |
| `length` | `int` | Actual number of observed time steps for this patient. |
| `extended_static` | `(9,)` | Static demographic features: Age, Gender=0, Gender=1, Height, ICUType={1,2,3,4}, Weight. Missing values encoded as **−1**. |

The label array `arr_outcomes` stores in its last column the in-hospital mortality label (0 or 1).

---

## 2. Train / Val / Test Split

Splits are managed by the active-learning framework (stratified splitting of patient indices). There are no external `phy12_split{k}.npy` files involved.

---

## 3. Normalization Statistics (computed on train set only)

### 3a. Time-series statistics

[p12_dataset.py:56-66](src/data/p12_dataset.py#L56-L66)

Per-feature mean and std are computed **only over the observed (non-zero) values** within `arr[:length]` of training patients, avoiding letting missing-value zeros bias the statistics.

### 3b. Static feature statistics

[p12_dataset.py:69-80](src/data/p12_dataset.py#L69-L80)

Per-feature mean and std are computed ignoring entries with value `< 0` (i.e., missing). **All 9 static features** (including binary Gender/ICUType dummies) are z-scored. Missing static values (original `−1`) are replaced with `0` after normalization.

---

## 4. `P12TransformerDataset` Output

[p12_dataset.py:167-226](src/data/p12_dataset.py#L167-L226)

Each `__getitem__` call returns `((ts, static, times, length), label)`:

| Tensor | Shape | Description |
|---|---|---|
| `ts` | `(215, 36)` | Normalized time-series values (0 where missing). **No mask channel.** |
| `static` | `(9,)` | Normalized static features (0 where originally missing). |
| `times` | `(215,)` | Timestamps in **minutes** (0-padded). Not converted to hours. |
| `length` | scalar `int64` | Actual sequence length (from `p['length']`). |
| `label` | scalar `int` | Mortality: 0 = survived, 1 = died. |

Key differences from the Raindrop baseline:
- No observation mask is concatenated — tensors are `(215, 36)` not `(N, 215, 72)`.
- Timestamps remain in **minutes**; there is no `/ 60.0` conversion.
- `length` comes directly from `p['length']`, not derived from `Ptime > 0`.

---

## 5. Mini-batch Sampling

Standard PyTorch `DataLoader` with the active-learning framework's sampler (no custom balanced-batch Strategy 2). Class imbalance is handled via the `model.weighted_loss` config flag (off by default for `transformer2`).

---

## 6. `BayesianTransformerModel2` Forward Pass

[bayesian_transformer.py:229-373](src/models/networks/bayesian_transformer.py#L229-L373)

The model is registered via [transformer2.py](src/models/networks/transformer2.py). A typical P12 instantiation uses:

```python
BayesianTransformerModel2(
    d_inp=36, d_model=36, nhead=1, nhid=2*36, nlayers=1,
    dropout_p=0.5, max_len=215, d_static=9, MAX=100,
    perc=0.25, aggreg='mean', n_classes=2, use_static=True
)
```

Input arrives batch-first as a 4-tuple `(ts, static, times, lengths)`.

### Step 1: Permute to time-first

```python
src      = ts.permute(1, 0, 2)    # (B, 215, 36) → (215, B, 36)
times_TB = times.permute(1, 0)    # (B, 215)     → (215, B)
```

### Step 2: Linear feature embedding

```python
src = self.encoder(src)   # Linear(36, 36) → (215, B, 36)   d_enc = d_inp = 36
```

### Step 3: Temporal positional encoding (concatenated, not added)

```python
pe  = self.pos_encoder(times_TB)    # PositionalEncodingTF → (215, B, 16)   d_pe = 16
src = torch.cat([pe, src], dim=2)   # (215, B, 52)   d_pe(16) + d_enc(36)
```

`PositionalEncodingTF` ([bayesian_transformer.py:55-88](src/models/networks/bayesian_transformer.py#L55-L88)) encodes actual timestamps (in minutes) using sinusoidal functions at 8 timescales. PE is **concatenated** (not added) with the feature embedding.

### Step 4: Input dropout

```python
src = self.input_dropout(src)   # Dropout(dropout_p), applied before transformer
```

### Step 5: Padding mask

```python
positions = torch.arange(T).unsqueeze(0)       # (1, 215)
pad_mask  = positions >= lengths.unsqueeze(1)  # (B, 215) — True for padded positions
```

### Step 6: Transformer encoder

```python
out = self.transformer_encoder(src, src_key_padding_mask=pad_mask)
# out: (215, B, 52)
out = out.permute(1, 0, 2)   # (B, 215, 52)
```

A single `TransformerEncoderLayer` (`nhead=1`, feedforward dim `nhid`, `dropout=dropout_p`).

### Step 7: Masked mean pooling

```python
valid   = (~pad_mask).unsqueeze(2).float()       # (B, 215, 1)
n_valid = valid.sum(dim=1).clamp(min=1)          # (B, 1)
pooled  = (out * valid).sum(dim=1) / n_valid     # (B, 52)
```

Divides by the **actual valid count** (not `lengths + 1`).

### Step 8: Concatenate static embedding

```python
static_emb = self.emb(static)                       # Linear(9, 36) → (B, 36)
pooled     = torch.cat([pooled, static_emb], dim=1) # (B, 88)
```

Final representation: `d_enc(36) + d_pe(16) + d_inp(36) = 88` dimensions.

### Step 9: Deterministic MLP hidden layer

```python
h = self.mlp_hidden(pooled)   # Linear(88, 88) → ReLU → (B, 88)
```

### Step 10: Stochastic Bayesian head (MC-Dropout)

```python
# Called k times via BayesianModule.mc_tensor for uncertainty estimation:
logits = self.classifier(self.mc_dropout(h_BK))
# mc_dropout: ConsistentMCDropout(dropout_p)
# classifier: Linear(88, 2)
# logits: (B*k, 2) → reshaped to (B, k, 2)
```

`ConsistentMCDropout` applies the same dropout mask across all elements of a batch during a single forward pass, which is required for consistent MC uncertainty estimates.

---

## Summary: Data Flow Diagram

```
PTdict_list.npy  (N patients)
    │
    ├── arr      (215, 36)  raw time-series     ─┐
    ├── time     (215, 1)   timestamps (min)     ├── per patient
    ├── length   int        actual seq length    │
    └── extended_static (9,) demographics       ─┘
                │
    [Step 2] Train/val/test split via AL framework
                │
    [Step 3] Compute feat mean/std (train only, non-zero values)
             Compute static mean/std (train only, non-negative values)
                │
    [Step 4] P12TransformerDataset.__getitem__:
             z-score features (zeros stay 0), z-score static (−1 → 0)
             → ts (215, 36), static (9,), times (215,) in minutes, length int
                │
    [Step 5] Standard DataLoader batching → (B, 215, 36) batch-first
                │
    ┌────────────────────── BayesianTransformerModel2._encode ─────────────────────┐
    │  Permute time-first   → (215, B, 36)                                         │
    │  Linear encoder       → (215, B, 36)   d_enc=36                             │
    │  Positional encoding  → (215, B, 16)   d_pe=16, uses timestamps in minutes  │
    │  Concatenate [PE|enc] → (215, B, 52)                                         │
    │  Input dropout                                                                │
    │  Transformer encoder  → (215, B, 52)   self-attention, padding masked        │
    │  Permute + masked mean pool → (B, 52)  divide by valid count                 │
    │  Static embed & cat   → (B, 88)        Linear(9→36) + concat                │
    │  MLP hidden           → (B, 88)        Linear(88,88) + ReLU  [deterministic] │
    └──────────────────────────────────────────────────────────────────────────────┘
                │
    ┌───── mc_forward_impl (called k times) ─────┐
    │  ConsistentMCDropout  → (B*k, 88)          │
    │  Linear(88, 2)        → (B*k, 2)           │
    └─────────────────────────────────────────────┘
                │
        (B, k, 2) logits  →  CrossEntropyLoss (over k=1 det pass)
                             or MC uncertainty (over k>1 passes)
```
