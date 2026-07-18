# Gradient Implementation Plan

Plan for incorporating gradient embeddings into the Vendi active learning experiments. Builds on `ideas.md` (Ideas 1–3, 5) and the current `src/query/query_diversity.py::_get_vendi`. Greedy batch acquisition (Idea 4) is **out of scope for now** — first verify that Vendi with cosine/linear kernels is fast enough, then revisit.

## 1. Gradient embeddings

### 1.1 Which layers

- **Last layer (default, start here).** BADGE-style gradient of the cross-entropy loss w.r.t. the classifier weights $W \in \mathbb{R}^{C \times Z}$: $g_x = (p_x - e_{\hat y_x}) h_x^\top$. `get_grad_embedding` already computes this (materialized, $C \cdot Z$-dim); the factorized path below avoids materializing it.
- **More layers (later).** Last $B$ blocks (e.g. last residual stage + head) via `torch.func.vmap(grad(...))`, with per-layer random projection so the full gradient is never materialized (TracIn recipe, Idea 2). Optionally concatenate 2–3 checkpoints. Gate this behind the sanity check: only build it if the last-layer kernel and the multi-layer kernel induce noticeably different rankings on one dataset.

### 1.2 Pseudo-label: $\arg\max \hat p$ vs. expectation w.r.t. $\hat p$

- **Argmax (hard pseudo-label):** $g_x = (p_x - e_{\hat y_x}) h_x^\top$ with $\hat y_x = \arg\max_c p_c(x)$. Simple, matches BADGE, enables the factorization/acceleration below.
- **Expectation:** note $\mathbb{E}_{y \sim p_x}[g_x] = 0$, so a naive expected gradient is useless. The meaningful "soft" option is the **second moment / Fisher variant** (Idea 1): $\mathrm{tr}(\mathcal{I}_x \mathcal{I}_y) = \mathrm{tr}(A_x A_y) \cdot \langle h_x, h_y \rangle^2$ with $A_x = \mathrm{diag}(p_x) - p_x p_x^\top$. Also factorized, also cheap. Treat argmax vs. Fisher as a sweep axis; expect Fisher to help early rounds where hard pseudo-labels are noisy.

### 1.3 Dimensionality reduction (if needed)

Only needed when gradients are materialized (multi-layer, or last-layer for kernels that don't factorize):

- **JL / random projection (TracIn):** fixed Rademacher matrix $R \in \{\pm 1\}^{d \times D}/\sqrt{d}$, $d \approx 1024$. Preserves inner products, hence the Gram spectrum and Vendi score approximately. For the last-layer tensor structure, apply the sketch to the factors ($\tilde\phi(x)_i = a_x^\top R_i b_x$, or TensorSketch) so $\phi(x)$ is never materialized.
- Draw $R$ **once per run** (fixed seed) so scores are comparable across AL iterations within a run.

### 1.4 Acceleration: argmax + cosine kernel (from `ideas.md`, Idea 1)

When using hard pseudo-labels and the cosine kernel, do **not** build $C \cdot Z$-dim embeddings. The gradient-cosine Gram matrix factorizes exactly as a Hadamard product of two small kernels:

$$K^{\mathrm{grad}} = K^{\mathrm{err}} \odot K^{\mathrm{feat}}, \qquad K^{\mathrm{err}}_{xy} = \cos(p_x - e_{\hat y_x},\, p_y - e_{\hat y_y})$$

Implementation: one forward pass collects $h_x$ (already done in `get_embeddings`) and $p_x$; compute the $C$-dim error kernel and elementwise-multiply with the existing feature kernel. Exact (bit-for-bit), PSD, unit diagonal — drops into `vendi_from_features` unchanged. Same trick works for the linear kernel (Frobenius inner products factorize) and for RBF (squared distance expands into the same small inner products).

Optional knob: $K = K^{\mathrm{feat}} \odot (K^{\mathrm{err}})^{\odot \alpha}$ or the blend $\alpha K^{\mathrm{grad}} + (1-\alpha) K^{\mathrm{feat}}$; $\alpha = 0$ recovers the current method.

## 2. q-Vendi

$$\mathrm{qVS}(\mathcal{S}) = \Big(\tfrac{1}{|\mathcal{S}|}\textstyle\sum_{x \in \mathcal{S}} s(x)\Big)\cdot \mathrm{VS}_q(\mathcal{S}; K)$$

### 2.1 Quality score $s(x)$ options (sweep)

- **Gradient norm:** $s(x) = \|g_x\|_F = \|p_x - e_{\hat y_x}\| \cdot \|h_x\|$ — free given the factorization; a margin-style uncertainty scaled by feature magnitude.
- **BALD:** MC-dropout mutual information (`models/bayesian_module.py` already supports this; reuse the `query_uncertainty.py` machinery).
- **Error-vector norm only:** $s(x) = \|p_x - e_{\hat y_x}\|$ — pure uncertainty, decoupled from feature magnitude (useful ablation vs. gradient norm, especially with L2-normalized features where the two coincide).

Consider a temperature/exponent $s(x)^\gamma$ to balance quality against diversity.

### 2.2 Problem to fix: quality of $\mathcal{L}$ drowning the signal of $u$

When scoring candidate $u$ via $\mathrm{qVS}(\mathcal{L} \cup \{u\})$, the average quality $\tfrac{1}{L+1}\big(\sum_{x \in \mathcal{L}} s(x) + s(u)\big)$ is dominated by the $L$ labeled points, so $s(u)$ contributes only $O(1/L)$ and shrinks every round. Candidate fixes (implement as a config option, compare):

1. **Exclude $\mathcal{L}$ from the quality average** (equivalently set $s(x) = 1$ for $x \in \mathcal{L}$): score $= s(u) \cdot \mathrm{VS}_q(\mathcal{L} \cup \{u\})$. Simplest; the qVS structure is kept only on candidates. Recommended default.
2. **Marginal-gain quality:** score $= s(u) \cdot \big[\mathrm{VS}_q(\mathcal{L} \cup \{u\}) - \mathrm{VS}_q(\mathcal{L})\big]$ — both factors are now purely about $u$; $\mathrm{VS}_q(\mathcal{L})$ is computed once per round.
3. **Log-domain additive form:** $\log s(u) + \log \mathrm{VS}_q(\mathcal{L} \cup \{u\})$ with a tradeoff weight $\beta$: $\beta \log s(u) + \log \mathrm{VS}_q$ — makes the quality–diversity balance explicit and sweepable.

Also verify: whichever variant is used must not let labeled-set quality shift *rankings* across candidates (any $u$-independent term is harmless for top-$k$; option 1 already has this property).

## 3. Experiment settings to sweep

All axes below, plus everything in Sections 1–2 (layers, argmax/Fisher, $\alpha$, quality function, quality-aggregation fix, $\gamma$, Rényi order $q$).

### 3.1 Kernel functions

- **Cosine** — theoretical support from TracIn (gradient inner products ≈ influence); factorizes (Section 1.4).
- **Linear** — same TracIn support, unnormalized influence; factorizes. Note: linear kernel needs care for Vendi (diagonal not 1); normalize by $\|g_x\|$, which recovers cosine, or track the unnormalized spectrum deliberately.
- **RBF** — current default; gamma fixed vs. median heuristic (both already in config).
- **Candidates for "anything else":** Hadamard-power / blended kernel with $\alpha$ (Section 1.4) as an interpolation axis; Fisher kernel (Section 1.2); arc-cosine or polynomial kernels are possible but low priority — prefer spending the budget on $\alpha$, $q$, and quality sweeps.

### 3.2 Embedding normalization

L2, min-max, z-score, none (all already in `normalize_features`). Interactions to keep in mind so the sweep doesn't waste runs: cosine kernel is invariant to per-sample scaling, so L2 vs. none are identical under cosine; L2 + linear = cosine; min-max/z-score are feature-wise and do change cosine. Gradient-norm quality $s(x)$ should be computed **before** normalization or it degenerates under L2.

### 3.3 Harness

Datasets MNIST → CIFAR-10/100; baselines already in `src/query/`: random, entropy, BALD, BADGE, kcenter-greedy, current feature-Vendi. Log per-round: acquisition wall-clock and eig time (already in `timing.csv`), batch-internal Vendi of the acquired batch, and the quality/diversity factors separately so the qVS balance is diagnosable.

## 4. Order of work

1. Factorized gradient kernel ($K^{\mathrm{err}} \odot K^{\mathrm{feat}}$) for cosine/linear + timing check — this answers the "fast enough?" question that gates greedy acquisition.
2. qVS with quality options and the labeled-set fix (Section 2.2).
3. Sweep kernels × normalization × pseudo-label × quality.
4. Fisher variant; multi-layer sketched gradients only if the sanity check justifies it.
5. Revisit greedy batch acquisition (Idea 4) once timings are in.
