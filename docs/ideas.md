# Ideas for a More Sophisticated Vendi-Based Active Learning Method

## 0. Where the current method stands

The current implementation (`src/query/query_diversity.py::_get_vendi`) scores each unlabeled candidate $u$ **independently**: it forms the $(L+1)\times(L+1)$ kernel matrix of $\mathcal{L} \cup \{u\}$ (labeled set $\mathcal{L}$, penultimate-layer features, RBF/cosine kernel), computes the Vendi score $\mathrm{VS}_q = \exp\!\big(\mathrm{RenyiEntropy}_q(\bar\lambda)\big)$, and takes the top-$k$ scores.

Two structural weaknesses:

1. **Representation**: penultimate features capture *where* a sample lives, not *what the model would learn from it*. There is no uncertainty signal at all.
2. **Batch redundancy**: scores are marginal gains w.r.t. $\mathcal{L}$ only. Two near-duplicate candidates far from $\mathcal{L}$ both get high scores and both get queried — exactly the failure mode BADGE (1906.03671) and BatchBALD were designed to avoid.

There is also a **cost** weakness worth fixing on the way: the code performs $U$ eigendecompositions of $(L+1)\times(L+1)$ matrices per round, i.e. $O(U L^3)$, which forces the `approx`/lobpcg path and will not scale as $L$ grows.

The ideas below address representation (Ideas 1–3), batch redundancy (Idea 4), and scalability (Idea 5), then combine into one proposed method (Section 6).

---

## 1. Factorized gradient kernel: get gradient embeddings "for free"

**Problem being solved.** You noted that last-layer gradient embeddings are "just scaled image embeddings." Make this exact and then *exploit* it instead of fighting it. For cross-entropy with hallucinated (pseudo) label $\hat y = \arg\max_c p_c(x)$ (the BADGE construction), the gradient w.r.t. the last-layer weights $W \in \mathbb{R}^{C\times Z}$ is a rank-1 outer product:

$$g_x = \frac{\partial \ell}{\partial W} = (p_x - e_{\hat y_x})\, h_x^\top,$$

where $h_x$ is the penultimate feature and $p_x$ the softmax output. Therefore the inner product **factorizes exactly**:

$$\langle g_x, g_y\rangle_F = \underbrace{\langle p_x - e_{\hat y_x},\, p_y - e_{\hat y_y}\rangle}_{\text{disagreement/uncertainty term}} \cdot \underbrace{\langle h_x, h_y\rangle}_{\text{feature term}}.$$

**Consequence.** A Vendi score over BADGE gradient embeddings never needs the $C\!\cdot\!Z$-dimensional embeddings at all. The gradient-cosine kernel is simply a **Hadamard (elementwise) product of two small kernels**:

$$K^{\mathrm{grad}} = K^{\mathrm{err}} \odot K^{\mathrm{feat}},$$

with $K^{\mathrm{err}}_{xy} = \cos(p_x - e_{\hat y_x},\, p_y - e_{\hat y_y})$ computed from $C$-dim vectors and $K^{\mathrm{feat}}$ the existing feature kernel.

*Why the Hadamard product, step by step.* Write $a_x = (p_x - e_{\hat y_x})/\|p_x - e_{\hat y_x}\|$ (unit error vector, $\mathbb{R}^C$) and $b_x = h_x/\|h_x\|$ (unit feature, $\mathbb{R}^Z$). Because $g_x = (p_x - e_{\hat y_x})\, h_x^\top$ is rank-1, both its Frobenius inner products and its norm factorize:

$$\langle g_x, g_y\rangle_F = \langle p_x - e_{\hat y_x},\, p_y - e_{\hat y_y}\rangle\,\langle h_x, h_y\rangle, \qquad \|g_x\|_F = \|p_x - e_{\hat y_x}\|\,\|h_x\|.$$

Dividing the first by the norms, the cosine of two gradients is exactly the product of the two small cosines:

$$\cos(g_x, g_y) = \langle a_x, a_y\rangle \cdot \langle b_x, b_y\rangle = K^{\mathrm{err}}_{xy}\, K^{\mathrm{feat}}_{xy}.$$

A matrix whose every entry is the product of the corresponding entries of two matrices *is* the Hadamard product — so the entrywise identity above, holding for all pairs $(x,y)$ simultaneously, is the matrix identity $K^{\mathrm{grad}} = K^{\mathrm{err}} \odot K^{\mathrm{feat}}$. Nothing is approximated: computing two cheap kernels ($C$-dim and $Z$-dim inner products) and multiplying them elementwise gives *bit-for-bit* the same Gram matrix as materializing all $C\!\cdot\!Z$-dimensional BADGE gradients and taking their cosines.

*Why the Vendi machinery still applies.* The Vendi score needs $K$ to be PSD with unit diagonal. Unit diagonal: $\cos(g_x,g_x)=1$. PSD: the Schur product theorem says the Hadamard product of PSD matrices is PSD, but here there is a more transparent argument that also answers the factorization question below — the identity $\langle a_x, a_y\rangle\langle b_x, b_y\rangle = \langle a_x \otimes b_x,\, a_y \otimes b_y\rangle$ (a defining property of the Kronecker/tensor product) exhibits $K^{\mathrm{grad}}$ as an ordinary Gram matrix of the vectors $a_x \otimes b_x$, and Gram matrices are always PSD. The RBF variant works too, since $\|g_x - g_y\|_F^2 = \|g_x\|_F^2 + \|g_y\|_F^2 - 2\langle g_x, g_y\rangle_F$ expands into these same small inner products.

**Is $K^{\mathrm{grad}}$ still $X^\top X$ for some explicit $X$? Yes.** By the tensor-product identity above,

$$K^{\mathrm{grad}} = \Phi\,\Phi^\top, \qquad \Phi \in \mathbb{R}^{n \times CZ} \text{ with rows } \phi(x)^\top = (a_x \otimes b_x)^\top = \mathrm{vec}(a_x b_x^\top)^\top,$$

i.e. the explicit feature map is just the *normalized* BADGE gradient itself, $\phi(x) = \mathrm{vec}(a_x b_x^\top)$, computable in $O(C + Z)$ storage per sample via its two factors. So the dual/covariance trick of Idea 5 carries over: the nonzero eigenvalues of $K^{\mathrm{grad}}$ equal those of $\Sigma = \Phi^\top\Phi \in \mathbb{R}^{CZ \times CZ}$. Two practical points:

- **Dimensionality.** The dual dimension is $CZ$ (e.g. $10 \times 512 = 5120$ for CIFAR-10/ResNet-18, $100 \times 512 = 51{,}200$ for CIFAR-100), so diagonalizing $\Sigma$ directly only beats the $n \times n$ primal once $n \gtrsim CZ$. The fix is a JL sketch $\tilde\phi(x) = R\,\phi(x)$ with $R \in \mathbb{R}^{d \times CZ}$, $d \approx 1024$, which never materializes $\phi(x)$: reshape each row $r_i$ of $R$ into $R_i \in \mathbb{R}^{C\times Z}$, so $\tilde\phi(x)_i = a_x^\top R_i\, b_x$ — $O(dCZ)$ per sample of pure GEMM, or $O(C + Z + d\log d)$ with TensorSketch, which is designed exactly for sketching tensor products from their factors. This preserves inner products (hence the spectrum and Vendi score approximately) and restores the $d \times d$ eigenproblem of Idea 5, independent of $L$, $U$, and $C$.
- **Rank.** $\mathrm{rank}(K^{\mathrm{grad}}) \le \min\big(n,\ \mathrm{rank}(K^{\mathrm{err}})\cdot\mathrm{rank}(K^{\mathrm{feat}})\big) \le \min(n, CZ)$; after sketching the spectrum is additionally capped at $d$ effective dimensions — the same caveat as Idea 5, and benign for the same reason (acquisition only compares *relative* marginal gains, and $d$ can be raised if needed).

**Why this is more than a speed trick.** Two samples are now "similar" only if they are close in feature space **and** the model is wrong about them in the same way. Confidently-classified samples of the same class have nearly identical error directions, so dense easy regions collapse in the eigenvalue spectrum and stop inflating Vendi scores — the score concentrates on samples that would push $W$ in genuinely new directions. This is the qualitative behavior BADGE gets from k-means++ on gradients, but expressed inside the Vendi framework.

**Generalization knob.** Interpolate between pure-feature and pure-gradient diversity:

$$K = K^{\mathrm{feat}} \odot \big(K^{\mathrm{err}}\big)^{\odot \alpha}, \qquad \alpha \in [0, 1],$$

($\alpha$-th Hadamard power; PSD-safe for integer powers, and in practice usable for fractional $\alpha$ with a small diagonal jitter, or via $\alpha$-blending $\alpha K^{\mathrm{grad}} + (1-\alpha) K^{\mathrm{feat}}$ which is always PSD). $\alpha=0$ recovers the current method; $\alpha=1$ is the full gradient kernel. This becomes a single sweepable hyperparameter in `config/query/vendi.yaml`.

Feature-map view of the knob (relevant if the dual formulation of Idea 5 is used): integer Hadamard powers keep an explicit map, $(K^{\mathrm{err}})^{\odot m}$ has $\phi(x) = a_x^{\otimes m}$ (dimension $C^m$, sketchable as above); fractional powers generally admit *no* finite feature map and are not even guaranteed PSD — the reason for the jitter. The convex blend is the dual-friendly choice: its exact feature map is the concatenation $\big[\sqrt{\alpha}\,(a_x \otimes b_x);\ \sqrt{1-\alpha}\,b_x\big]$ of dimension $CZ + Z$, since concatenation adds Gram matrices.

**Fisher variant (soft labels).** Using hard pseudo-labels injects noise where the model is uncertain — arguably where it matters most. The expectation of $g_x$ over $y \sim p_x$ is zero, so use second moments instead: the last-layer Fisher information also factorizes,

$$\mathcal{I}_x = \big(\mathrm{diag}(p_x) - p_x p_x^\top\big) \otimes h_x h_x^\top, \qquad \mathrm{tr}(\mathcal{I}_x \mathcal{I}_y) = \mathrm{tr}(A_x A_y)\cdot \langle h_x, h_y\rangle^2,$$

with $A_x = \mathrm{diag}(p_x) - p_x p_x^\top$ ($C \times C$). A normalized version of $\mathrm{tr}(\mathcal{I}_x\mathcal{I}_y)$ is a valid kernel, giving a **Fisher-kernel Vendi** that connects directly to the BAIT objective (2106.09675) and to the EIG/EPIG view of Kirsch & Gal (2208.00549), who show BADGE/BAIT are last-layer approximations of information-theoretic acquisition. That framing gives the method a principled story for a paper, not just an empirical one.

---

## 2. Beyond the last layer: sketched multi-layer gradients

If last-layer gradients feel too feature-tied even after Idea 1, the blocker for deeper gradients is dimension, not FLOPs — and two of the papers solve exactly this:

- **TracIn (2002.08484)** uses per-sample gradients with **random projections** and **checkpoint ensembling**, and finds last-layer-only gradients are a reasonable but improvable approximation.
- **G-Vendi (2505.20161)** computes the Vendi score of **Rademacher-projected loss gradients** ($d = 1024$) from a small *off-the-shelf proxy model*, and shows it predicts OOD generalization far better than embedding-based diversity — strong external evidence for gradient-space Vendi.

Concrete recipe:

1. Pick the last $B$ blocks of the network (e.g., last residual stage + head), giving gradient dimension $D$.
2. Draw a fixed Rademacher matrix $R \in \{\pm 1\}^{d\times D}/\sqrt{d}$, $d \approx 1024$. Johnson–Lindenstrauss guarantees inner products are preserved, so the Vendi spectrum is approximately preserved.
3. Per-sample projected gradients $\tilde g_x = R\, \nabla_{\theta_{1:B}}\ell(x, \hat y_x)$ in one pass with `torch.func.vmap(grad(...))`; project layer-by-layer so the full $D$-dim gradient is never materialized.
4. (Optional, TracIn-style) Concatenate $\tilde g_x$ from 2–3 training checkpoints to denoise the single-checkpoint gradient and capture training dynamics: $\tilde g_x = [\tilde g_x^{(t_1)}; \tilde g_x^{(t_2)}; \dots]$.

Cost: one extra forward+backward over the pool with a $d$-dim output — comparable to the embedding extraction the code already does. This sits strictly between "last layer only" (degenerate, per your concern) and "whole model" (intractable), and the choice of $B$ and $d$ are clean ablation axes.

**Cheap sanity check before building it:** compare the eigenvalue spectra (and downstream AL curves) of $K^{\mathrm{feat}}$, $K^{\mathrm{grad}}$ (Idea 1), and sketched multi-layer $K$ on one dataset. If Idea 1 and Idea 2 kernels induce nearly identical rankings, skip Idea 2's extra machinery.

---

## 3. Uncertainty as quality: gradient-norm-weighted Vendi (qVS)

The quality-weighted Vendi score (2405.02449) is the natural home for your "gradient norm as the q" idea — with one terminology caution: in that paper $q$ is the **Rényi order** and quality is a *score function* $s(x)$; what you want is $s(x)$. The batch objective becomes

$$\mathrm{qVS}\big(\mathcal{L}\cup\mathcal{B}\big) = \Big(\tfrac{1}{n}\textstyle\sum_{x} s(x)\Big) \cdot \mathrm{VS}_q\big(\mathcal{L}\cup\mathcal{B}; K\big).$$

For the quality function, the last-layer gradient norm again factorizes:

$$s(x) = \|g_x\| = \|p_x - e_{\hat y_x}\| \cdot \|h_x\|,$$

and $\|p_x - e_{\hat y_x}\|$ is a margin-style uncertainty: $\to 0$ as confidence $\to 1$, maximal near uniform predictions. So the gradient norm *is* an uncertainty measure scaled by feature magnitude — free to compute given Idea 1. Alternatives to ablate: predictive entropy, BALD via MC-dropout (1703.02910, `models/bayesian_module.py` already supports this), and margin.

Design detail that matters: labeled points should contribute $s=1$ (or be excluded from the quality average) so that quality only modulates *candidate* selection; otherwise a large confident labeled set drags the average and flattens the acquisition signal. The Rényi order $q$ (already in the config) then independently controls diversity-sensitivity, exactly as characterized in the qVS paper: larger $q$ punishes near-duplicates harder — itself a partial mitigation of the redundancy problem.

---

## 4. Fixing top-k redundancy: greedy submodular batch selection

This is the highest-value change, and there is now theory to back it: **the log Vendi score is monotone submodular** (2605.29448 — it is a matrix spectral function $\mathrm{tr}[\phi(B_X)]$ with $\phi(x) = -x\log x$, whose derivative is matrix-monotone). Two consequences:

1. **Greedy selection has a $(1 - 1/e)$ approximation guarantee.** Replace top-$k$ with: start from $\mathcal{B}=\emptyset$, and repeat $k$ times
   $$x^* = \arg\max_{x \in \mathcal{U}\setminus\mathcal{B}} \; \mathrm{qVS}\big(\mathcal{L}\cup\mathcal{B}\cup\{x\}\big),$$
   i.e. condition each selection on the batch built so far. After picking $u_1$, any near-duplicate of $u_1$ has almost zero marginal gain and is skipped. This is precisely the mechanism the qVS paper (2405.02449, Algorithm 1) uses for diverse active search, and mirrors BatchBALD vs. top-k BALD.

2. **It can be made fast with rank-1 eigenvalue updates.** Naive greedy is $k \times U$ eigendecompositions. The 2026 paper's secular-equation trick computes the spectrum of $\Lambda + v v^\top$ (rank-1 update of an already-diagonalized kernel) in $O(m^2)$, reporting ~35,000× speedups for greedy Vendi maximization. Between greedy steps only one full re-diagonalization is needed. Additional standard accelerations that compose with this: **lazy greedy** (Minoux — valid because of submodularity) and **stochastic greedy** (sample $\tfrac{U}{k}\log\tfrac{1}{\epsilon}$ candidates per step, keeping $(1-1/e-\epsilon)$).

Note the current per-candidate scoring is *already* $O(U L^3)$; batched secular updates are $O(U L^2)$ per greedy step, so a $k$-step greedy costs about $k L^{-1} \cdot$ (relative to current) × current cost — with stochastic greedy it is typically *cheaper* than the existing code while solving the redundancy problem.

**Lightweight fallbacks** (worth having as baselines/ablations even if greedy wins):

- **Score-then-diversify**: take top-$\beta k$ ($\beta \approx 4$) by the current independent Vendi score, then run greedy qVS (or k-means++ seeding on embeddings, or a k-DPP) inside that shortlist. Two-stage, trivially cheap, usually captures most of the gain.
- **Vendi-weighted k-means++**: D²-style sampling where seeding probability $\propto$ (marginal Vendi gain)². Stochastic like BADGE, which its ablations found beats k-DPP.
- **Forward–backward greedy** (BAIT, 2106.09675): greedily select $2k$, then greedily *remove* $k$ — reported to beat pure forward greedy for non-submodular quality-weighted objectives (qVS with the quality average is no longer exactly submodular; this hedge is cheap).

---

## 5. Scalability rewrite: dual (covariance) formulation

Currently the kernel view forces $(L+1)$-sized eigenproblems. Switch to the **dual**: for any finite-dimensional embedding map $\phi$ (cosine kernel = normalized features; RBF ≈ random Fourier features; Ideas 1–2 give explicit $\phi$ via small products/sketches), the nonzero eigenvalues of the Gram matrix $\tfrac{1}{n}\Phi\Phi^\top$ equal those of the $d\times d$ covariance

$$\Sigma = \tfrac{1}{n}\Phi^\top \Phi = \tfrac{1}{n}\textstyle\sum_i \phi(x_i)\phi(x_i)^\top.$$

This is exactly how G-Vendi (2505.20161) computes Vendi at the 100k–1.5M scale. Benefits:

- Adding a candidate is a **rank-1 update of a fixed $d\times d$ matrix**, independent of $L$ and of batch-so-far size — greedy marginal gains in $O(d^2)$ via the same secular-equation machinery, $O(U d^2)$ per greedy step, with $d \approx 512$–2048.
- Memory drops from $O(\text{batch}\cdot L^2)$ (current `K` tensor) to $O(d^2)$.
- The `approx: lobpcg` path and its degenerate-eigenvalue warning (the `max_diff < 1e-8` check in `vendi_from_features`) become unnecessary.

One caveat: with $n > d$ samples the spectrum saturates at $d$ effective dimensions, slightly compressing Vendi values for huge $\mathcal{L}$; in practice this is fine because acquisition only needs *relative* marginal gains, and $d$ can be raised if needed.

---

## 6. The proposed method: greedy quality-weighted gradient-Vendi (put together)

Name candidate: **GQ-Vendi** (or qVS-AL-grad). Per acquisition round:

1. One forward pass over $\mathcal{L}\cup\mathcal{U}$: collect features $h_x$, softmax $p_x$; form the factorized gradient embedding $\phi(x) = \big(\text{normalized } (p_x - e_{\hat y_x})\big) \otimes \big(\text{normalized } h_x\big)$ realized implicitly via the product kernel (Idea 1), or sketched multi-layer gradients (Idea 2). Quality $s(x) = \|p_x - e_{\hat y_x}\|\cdot\|h_x\|$ (Idea 3).
2. Build $\Sigma_{\mathcal{L}} = \sum_{x\in\mathcal{L}} \phi(x)\phi(x)^\top$ and diagonalize once ($O(d^3)$) (Idea 5).
3. Greedy loop with stochastic-greedy candidate sampling + lazy evaluations: marginal gain of $x$ = $\mathrm{qVS}$ after the rank-1 update $\Sigma \mathrel{+}= \phi(x)\phi(x)^\top$, computed by secular equation in $O(d^2)$ (Idea 4). Select $k$ points.
4. Log the per-step marginal gains — they give an interpretable "effective new samples per query" curve, a diagnostic the current `extra_info` dict approximates crudely.

Interactions worth noting: greedy selection (Idea 4) *needs* a batch-aware objective to help, which qVS provides; the dual form (Idea 5) *needs* explicit finite features, which Ideas 1–2 provide. The pieces are complementary rather than independent.

---

## 7. Risks, caveats, and what to test

- **Diversity ≠ value.** The 2026 paper (2605.29448) found that when the Vendi score is *directly maximized*, its correlation with downstream test accuracy is much weaker than expected — pure diversity maximization loves outliers and label noise. This is an argument *for* the quality weighting (Idea 3) and possibly a facility-location or coverage hybrid term; treat pure-Vendi greedy as a baseline to beat, not the end state.
- **Pseudo-label noise early in training.** Hard $\hat y$ gradients are unreliable in the first rounds (cold start). Mitigations: Fisher/soft-label kernel (Idea 1), checkpoint averaging (Idea 2), or annealing $\alpha$ from 0 (features) toward 1 (gradients) across rounds.
- **Quality–diversity balance is dataset-dependent.** Sweep the Rényi order $q$ and an exponent on $s(x)$ ($s^\gamma$) jointly; the qVS paper shows $q$ meaningfully changes selection geometry.
- **Class imbalance in the error kernel**: with few classes ($C=10$), $K^{\mathrm{err}}$ has limited expressiveness; verify Idea 1 still helps on CIFAR-100-like settings, where it should shine.

**Experiment plan** (fits the existing harness): MNIST → CIFAR-10/100; baselines already in `src/query/`: random, entropy, BALD, BatchBALD, kcenter-greedy, BADGE, current vendi. Ablate: (a) kernel = feat / grad-factorized / Fisher / sketched-multilayer; (b) selection = top-k / score-then-diversify / greedy / forward–backward; (c) quality = none / grad-norm / entropy / BALD; (d) batch size $k \in \{64, 256, 1024\}$ — redundancy failures of top-k grow with $k$, so the gap should widen at large $k$, which is the headline plot. Report batch-internal Vendi (diversity of the acquired batch itself) alongside accuracy to show the mechanism, not just the outcome.

---

## References (papers/ folder)

- 1703.02910 — Gal et al., *Deep Bayesian Active Learning with Image Data* (MC-dropout, BALD)
- 1906.03671 — Ash et al., *BADGE: Batch AL by Diverse Gradient Embeddings*
- 2002.08484 — Pruthi et al., *TracIn: Estimating Training Data Influence* (checkpoints, random projections)
- 2106.09675 — Ash et al., *BAIT: Fisher Embeddings for Neural AL* (forward–backward greedy)
- 2208.00549 — Kirsch & Gal, *Unifying AL via Fisher Information* (EIG/EPIG view, last-layer critique)
- 2210.02410 — Friedman & Dieng, *The Vendi Score*
- 2405.02449 — Nguyen & Dieng, *Quality-Weighted Vendi Scores* (qVS, greedy batch construction)
- 2505.20161 — Jung et al., *Prismatic Synthesis / G-Vendi* (projected-gradient Vendi at scale)
- 2605.29448 — Bhatt et al., *Scaling Laws, the Vendi Score, and Matrix Spectral Functions* (submodularity, secular-equation rank-1 updates, diversity≠value caveat)
