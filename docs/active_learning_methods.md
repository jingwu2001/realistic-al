# Data Subset Selection Methods: Objectives and Estimators

Based on Kirsch & Gal, *Unifying Approaches in Active Learning and Active Sampling via Fisher Information and Information-Theoretic Quantities* (arXiv:2208.00549v2, TMLR 2022).

For each method: **(a)** the information quantity it optimizes, **(b)** how that quantity is estimated in practice, in formulas.

---

## 1. Abbreviations

| Abbreviation | Meaning |
|---|---|
| BALD | Bayesian Active Learning by Disagreement (Houlsby et al., 2011) |
| BatchBALD | Batch version of BALD with joint mutual information (Kirsch et al., 2019) |
| EIG | Expected Information Gain (= BALD's objective; Lindley, 1956) |
| EPIG | Expected Predictive Information Gain (Kirsch et al., 2021) |
| JEPIG | Joint Expected Predictive Information Gain |
| IG / PIG / JPIG | (Joint) (Predictive) Information Gain — active-sampling analogues where labels are known |
| BADGE | Batch Active learning by Diverse Gradient Embeddings (Ash et al., 2019) |
| BAIT | Batch Active learning via Information maTrices, from the paper "Gone Fishing" (Ash et al., 2021) |
| SIMILAR | Submodular Information Measures based actIve LeARning (Kothawade et al., 2021) |
| PRISM | Parameterized submodular Information Measures framework (Kothawade et al., 2022) |
| EGL | Expected Gradient Length (Settles et al., 2007) |
| GraNd | Gradient Norm score, from "Deep Learning on a Data Diet" (Paul et al., 2021) |
| RHO-loss | Reducible Holdout Loss (Mindermann et al., 2022) — practical estimator related to PIG/JPIG |
| GGN | Generalized Gauss-Newton approximation (replace the Hessian by the Fisher information) |
| GLM | Generalized Linear Model |
| DPP / k-DPP | (k-)Determinantal Point Process — distribution over subsets with $P(\text{batch}) \propto \det S_{\text{batch}}$ |
| MC | Monte Carlo |

---

## 2. Notation and Setup

Bayesian discriminative model with parameters $\Omega$ (outcome $\omega$), inputs $x$, predictions $Y$:

$$p(y, \omega \mid x) = p(y \mid x, \omega)\, p(\omega), \qquad p(y \mid x) = \mathbb{E}_{p(\omega)}[p(y \mid x, \omega)].$$

- $\mathcal{D}^{\text{train}}$: labeled training set; $p(\omega)$ implicitly means the current posterior $p(\omega \mid \mathcal{D}^{\text{train}})$.
- $\mathcal{D}^{\text{pool}}$: unlabeled pool; $x^{\text{acq}}$ (or a batch $\{x_i^{\text{acq}}\}_{i=1}^B$) is a candidate to acquire; $Y^{\text{acq}}$ its unknown label, $y^{\text{acq}}$ a known label.
- $\mathcal{D}^{\text{eval}}$ / $X^{\text{eval}}$: evaluation data used by *transductive* objectives (often the pool itself).
- Entropy $H[X] = \mathbb{E}[-\log p(x)]$; for a probability vector $\pi$: $H[\pi] = -\sum_c \pi_c \log \pi_c$.
- Mutual information $I[X; Y] = H[X] - H[X \mid Y]$.
- $\omega^*$: a MAP / trained parameter estimate.

**Derivative notation** (the paper's key notational device). With $H[\cdot] = -\log p(\cdot)$:

$$H'[y \mid x, \omega^*] := -\nabla_\omega \log p(y \mid x, \omega^*) \quad (\text{the loss gradient}),$$
$$H''[y \mid x, \omega^*] := -\nabla^2_\omega \log p(y \mid x, \omega^*) \quad (\textbf{observed information}, \text{ the loss Hessian}),$$
$$H''[Y \mid x, \omega^*] := \mathbb{E}_{p(y \mid x, \omega^*)}\big[H''[y \mid x, \omega^*]\big] \quad (\textbf{Fisher information}).$$

Equivalently, $H''[Y \mid x, \omega^*] = \mathbb{E}_{p(y\mid x,\omega^*)}\big[H'[y \mid x, \omega^*]^\top H'[y \mid x, \omega^*]\big]$ (outer product of gradients). Both are additive over data points. The **only difference between active-learning and active-sampling objectives after approximation is Fisher information (label-free, capital $Y$) vs. observed information (uses the actual label $y$).**

### 2.1 Taxonomy of objectives

| | Active learning (label unknown) | Active sampling (label known) |
|---|---|---|
| Non-transductive | **EIG/BALD**: $I[\Omega; Y^{\text{acq}} \mid x^{\text{acq}}]$ | **IG**: $I[\Omega; y^{\text{acq}} \mid x^{\text{acq}}]$ |
| Transductive, expectation | **EPIG**: $\mathbb{E}_{\hat p(x^{\text{eval}})} I[Y^{\text{eval}}; Y^{\text{acq}} \mid x^{\text{eval}}, x^{\text{acq}}]$ | **PIG**: $\mathbb{E}_{\hat p(x^{\text{eval}}, y^{\text{eval}})} I[y^{\text{eval}}; y^{\text{acq}} \mid x^{\text{eval}}, x^{\text{acq}}]$ |
| Transductive, joint | **JEPIG**: $I[\{Y_i^{\text{eval}}\}; Y^{\text{acq}} \mid \{x_i^{\text{eval}}\}, x^{\text{acq}}]$ | **JPIG**: $I[\{y_i^{\text{eval}}\}; y^{\text{acq}} \mid \{x_i^{\text{eval}}\}, x^{\text{acq}}]$ |

Batch versions substitute $\{x_i^{\text{acq}}\}$, $\{Y_i^{\text{acq}}\}$ treated as *joint* random variables.

### 2.2 The two estimation toolkits

Every method below is one of these two recipes applied to one objective from the taxonomy.

**Toolkit A — prediction space (MC sampling).** Draw $N$ posterior samples $\omega_1, \dots, \omega_N \sim p(\omega \mid \mathcal{D}^{\text{train}})$ (MC dropout, deep ensembles, SWAG, ...) and estimate entropies of predictions directly from softmax outputs. No Gaussian assumption, but joint entropies over a batch require enumerating/sampling label configurations (exponential in batch size $B$).

**Toolkit B — weight space (Laplace + Fisher).** Approximate the posterior by a Gaussian via a second-order Taylor expansion:

$$p(\omega \mid \mathcal{D}) \approx \mathcal{N}\big(\omega^*,\ H''[\omega^* \mid \mathcal{D}]^{-1}\big), \qquad H[\Omega] \approx -\tfrac{1}{2} \log \det H''[\omega^* \mid \mathcal{D}] + \tfrac{k}{2}\log 2\pi e,$$

with $H''[\omega^* \mid \mathcal{D}] = \sum_i H''[y_i \mid x_i, \omega^*] + H''[\omega^*]$ (prior term). Mutual informations become log-det ratios of Hessians. Key results:

$$I[\Omega; \{Y_i^{\text{acq}}\} \mid \{x_i^{\text{acq}}\}, \mathcal{D}^{\text{train}}] \;\overset{\approx}{\le}\; \tfrac{1}{2} \log \det \Big( \sum_i H''[Y_i^{\text{acq}} \mid x_i^{\text{acq}}, \omega^*] \; H''[\omega^* \mid \mathcal{D}^{\text{train}}]^{-1} + I_d \Big) \tag{log-det}$$

$$\le\; \tfrac{1}{2} \sum_i \mathrm{tr}\Big( H''[Y_i^{\text{acq}} \mid x_i^{\text{acq}}, \omega^*] \; H''[\omega^* \mid \mathcal{D}^{\text{train}}]^{-1} \Big) \tag{trace}$$

using $\log\det(A + I_d) \le \mathrm{tr}(A)$. **The trace bound is additive over samples** → top-$k$ selection → cannot see redundancy between batch candidates (batch-acquisition pathology). The log-det keeps interactions.

Practical simplifications:
- **Exponential-family likelihood** (softmax, Gaussian): the Fisher needs no expectation over $y$. For softmax with logits $\hat f(x;\omega)$, Jacobian $J = \nabla_\omega \hat f(x;\omega^*)$ and $\pi = \mathrm{softmax}(\hat f(x;\omega^*))$:
$$H''[Y \mid x, \omega^*] = J^\top (\mathrm{diag}(\pi) - \pi\pi^\top) J.$$
- **GGN**: also use this Fisher as a stand-in for the observed information $H''[y \mid x, \omega^*]$ (which is not PSD in general). Under GGN/GLM, active learning and active sampling approximations coincide.
- **Last layer**: freeze the encoder $z = f(x)$, treat the head $p(y \mid z, W)$ as a GLM. Then $H''[Y \mid x, \omega^*] = (\mathrm{diag}(\pi) - \pi\pi^\top) \otimes z z^\top$.
- **Similarity-matrix duality.** Stack per-sample gradients $g_i = H'[y_i \mid x_i, \omega^*]$ as rows of $\hat H'$. Then $\hat H'^\top \hat H'$ ($k \times k$) is a one-sample estimate of the total Fisher, while $S := \hat H' \hat H'^\top$ ($n \times n$, $S_{ij} = \langle g_i, g_j \rangle$) is the **gradient similarity matrix**, and by the matrix-determinant lemma (uninformative prior, sampled labels):
$$I[\Omega; \{Y_i^{\text{acq}}\} \mid \{x_i^{\text{acq}}\}, \mathcal{D}^{\text{train}}] \;\overset{\approx}{\lessapprox}\; \tfrac{1}{2} \log \det S[\mathcal{D}^{\text{acq}} \mid \omega^*]. \tag{similarity}$$
Using **hard pseudo-labels** $\hat y = \arg\max_y p(y \mid x, \omega^*)$ instead of sampling $y \sim p(y \mid x, \omega^*)$ gives a *biased* estimate.

---

## 3. The Methods

### 3.1 BALD — prediction-space EIG

**Optimizes** the EIG (expected reduction in parameter uncertainty from labeling $x^{\text{acq}}$):

$$\alpha_{\text{BALD}}(x) = I[\Omega; Y \mid x] = H[Y \mid x] - \mathbb{E}_{p(\omega)}\big[ H[Y \mid x, \omega] \big].$$

**Estimated** with Toolkit A. Draw $\omega_1, \dots, \omega_N \sim p(\omega \mid \mathcal{D}^{\text{train}})$, compute probability vectors $\pi_n = p(\cdot \mid x, \omega_n)$ (softmax outputs of $N$ stochastic forward passes), then:

$$\hat\alpha_{\text{BALD}}(x) = \underbrace{H\Big[\tfrac{1}{N}\sum_{n=1}^N \pi_n\Big]}_{\text{entropy of mean prediction}} - \underbrace{\tfrac{1}{N}\sum_{n=1}^N H[\pi_n]}_{\text{mean entropy per sample}}, \qquad H[\pi] = -\sum_c \pi_c \log \pi_c.$$

High score = the ensemble members *disagree* (each confident, but about different classes). Batch selection: top-$B$ scores — which is exactly the additive/top-$k$ pathology (can pick $B$ duplicates).

### 3.2 BatchBALD — prediction-space joint EIG

**Optimizes** the joint EIG of the whole batch:

$$\alpha_{\text{BatchBALD}}(\{x_i\}_{i=1}^B) = I[\Omega; Y_1, \dots, Y_B \mid x_1, \dots, x_B] = H[Y_{1:B} \mid x_{1:B}] - \mathbb{E}_{p(\omega)}\Big[\sum_{i=1}^B H[Y_i \mid x_i, \omega]\Big],$$

where the second term splits into a sum because predictions are independent *given* $\omega$; the first joint entropy does **not** split — that is where redundancy between candidates is accounted for.

**Estimated** with Toolkit A. The joint predictive distribution over a label configuration $y_{1:B}$:

$$\hat p(y_{1:B}) = \tfrac{1}{N} \sum_{n=1}^N \prod_{i=1}^B p(y_i \mid x_i, \omega_n), \qquad \hat H[Y_{1:B}] = -\sum_{y_{1:B}} \hat p(y_{1:B}) \log \hat p(y_{1:B}),$$

with the sum over the $C^B$ configurations enumerated for small $B$ and importance-sampled for large $B$. Since the joint EIG is **monotone submodular**, the batch is built greedily — fix already-chosen points, add the $x$ maximizing the joint score — with a $(1 - 1/e)$ optimality guarantee.

### 3.3 EPIG — prediction-space transductive objective

**Optimizes** the expected reduction of uncertainty about *predictions on test-like inputs* (not parameters):

$$\alpha_{\text{EPIG}}(x^{\text{acq}}) = \mathbb{E}_{\hat p(x^{\text{eval}})}\, I[Y^{\text{eval}}; Y^{\text{acq}} \mid x^{\text{eval}}, x^{\text{acq}}],$$

with $\hat p(x^{\text{eval}})$ an empirical distribution of evaluation inputs (e.g., the pool, or held-out unlabeled data). Rationale: an outlier can have high BALD score (very informative about $\Omega$) yet be useless for predictions where they matter. Equivalent proxy: $\arg\max_x \text{EPIG} = \arg\min_x I[\Omega; Y^{\text{eval}} \mid X^{\text{eval}}, Y^{\text{acq}}, x^{\text{acq}}]$.

**Estimated** with Toolkit A. Sample $M$ eval points $x_m^{\text{eval}} \sim \hat p(x^{\text{eval}})$ and $N$ posterior samples; the joint predictive for the pair $(Y^{\text{acq}}, Y^{\text{eval}})$ is

$$\hat p(y^a, y^e \mid x^{\text{acq}}, x_m^{\text{eval}}) = \tfrac{1}{N} \sum_{n=1}^N p(y^a \mid x^{\text{acq}}, \omega_n)\, p(y^e \mid x_m^{\text{eval}}, \omega_n),$$

$$\hat\alpha_{\text{EPIG}}(x^{\text{acq}}) = \tfrac{1}{M} \sum_{m=1}^M \sum_{y^a, y^e} \hat p(y^a, y^e) \log \frac{\hat p(y^a, y^e)}{\hat p(y^a)\, \hat p(y^e)}.$$

EPIG is **not** submodular, so greedy batch construction has no optimality guarantee.

### 3.4 JEPIG — joint transductive variant

**Optimizes** $I[\{Y_i^{\text{eval}}\}; Y^{\text{acq}} \mid \{x_i^{\text{eval}}\}, x^{\text{acq}}]$ — information about the *joint* labeling of the whole eval set rather than an average over single eval points. JEPIG converges to BALD as the eval set grows (when the pool has no outliers); EPIG does not.

**Estimated** with the same MC machinery as EPIG/BatchBALD, now requiring joint entropies over eval-set label configurations (sampled, since exact enumeration is infeasible). In weight space it has log-det/trace approximations analogous to Section 2.2; notably the **trace** approximations of EPIG and JEPIG coincide (up to a constant), i.e. the trace bound is too loose to distinguish them.

### 3.5 IG, PIG, JPIG — active sampling (labels known)

**Optimize** the taxonomy's right column: e.g. $\text{IG} = I[\Omega; y^{\text{acq}} \mid x^{\text{acq}}]$, how much *training on* the already-labeled $(x^{\text{acq}}, y^{\text{acq}})$ would reduce parameter (IG) or predictive (PIG/JPIG) uncertainty. Used for data pruning / curriculum ("active sampling").

**Estimated** in weight space by replacing Fisher with observed information in the formulas of Section 2.2, e.g.

$$I[\Omega; \{y_i^{\text{acq}}\} \mid \{x_i^{\text{acq}}\}, \mathcal{D}^{\text{train}}] \;\overset{\approx}{\le}\; \tfrac{1}{2} \log\det\Big( \sum_i H''[y_i^{\text{acq}} \mid x_i^{\text{acq}}, \omega^*]\, H''[\omega^* \mid \mathcal{D}^{\text{train}}]^{-1} + I_d \Big) \le \tfrac{1}{2}\sum_i \mathrm{tr}(\cdots).$$

Under GLM/GGN, $H''[y \mid x, \omega^*] = H''[Y \mid x, \omega^*]$, so these estimates **equal the EIG estimates — the label buys nothing**. A prediction-space estimator for (J)PIG is the **RHO-loss** (Mindermann et al., 2022): prioritize points by *reducible holdout loss*,

$$\hat\alpha_{\text{RHO}}(x, y) = \underbrace{H[y \mid x; \mathcal{D}^{\text{train}}]}_{\text{loss of current model}} - \underbrace{H[y \mid x; \mathcal{D}^{\text{ho}}]}_{\text{loss of a model trained on holdout data}},$$

i.e., cross-entropy of the current model minus that of an auxiliary "irreducible loss" model — skip points that are already learned, unlearnable, or atypical for the target distribution.

### 3.6 BADGE — weight-space EIG via gradient embeddings

**Optimizes** (implicitly — the original paper is intuition-only) the **batch EIG with an uninformative prior**, in the similarity-matrix form:

$$\alpha_{\text{BADGE}}(\{x_i\}) \approx \tfrac{1}{2} \log\det S[\mathcal{D}^{\text{acq}} \mid \omega^*], \qquad S_{ij} = \langle g_i, g_j \rangle.$$

**Estimated** as follows. For each pool point $x$, compute the **last-layer gradient embedding with a hard pseudo-label**: with penultimate embedding $z = f(x)$, probabilities $\pi = \mathrm{softmax}(W z)$, pseudo-label $\hat y = \arg\max_c \pi_c$, the gradient of the loss w.r.t. the last-layer weights $W$ is

$$g_x = \nabla_W \big[-\log p(\hat y \mid x, \omega^*)\big] = (\pi - e_{\hat y})\, z^\top \in \mathbb{R}^{C \times d} \quad (\text{flattened}), \qquad \|g_x\|_2 = \|\pi - e_{\hat y}\|_2 \, \|z\|_2,$$

where $e_{\hat y}$ is the one-hot vector of $\hat y$. So $\|g_x\|$ encodes uncertainty (small when confident) and the *direction* encodes what the update would change. Then select a batch of $B$ diverse embeddings via **k-means++ seeding**: pick $g_{(1)}$ (proportional to norm), then iteratively pick $x$ with probability $\propto \min_{j < b} \|g_x - g_{(j)}\|_2^2$. This is a cheap stand-in for a **k-DPP**, which samples $P(\text{batch}) \propto \det S_{\text{batch}}$ — a stochastic relaxation of maximizing $\log\det S$, i.e. of the batch EIG.

**Relation to BatchBALD:** same objective (batch EIG), different everything else — weight space (last layer, hard pseudo-labels → biased Fisher estimate, uninformative prior) vs. prediction space; DPP-style sampling vs. greedy submodular maximization.

### 3.7 BAIT — weight-space EPIG (trace form)

**Optimizes** — despite being derived without information theory — the **trace approximation of (J)EPIG** with the pool as eval set (Proposition 7.1):

$$\arg\min_{\{x_i^{\text{acq}}\}} \; \mathrm{tr}\Big( \underbrace{H''[\{Y_i^{\text{eval}}\} \mid \{x_i^{\text{eval}}\}, \omega^*]}_{F_{\text{eval}} \;=\; \sum_{x \in \text{pool}} F_x} \Big( \underbrace{H''[\{Y_i^{\text{acq}}\} \mid \{x_i^{\text{acq}}\}, \omega^*]}_{F_{\text{acq}}} + F_{\text{train}} + \lambda I_d \Big)^{-1} \Big),$$

which matches $\arg\min_{x^{\text{acq}}} I[\Omega; Y^{\text{eval}} \mid X^{\text{eval}}, \{Y_i^{\text{acq}}\}, \{x_i^{\text{acq}}\}]$, the EPIG proxy. Intuition: choose the batch whose Fisher information, added to what training data already provides, best "covers" the directions in weight space that pool predictions depend on.

**Estimated** with last-layer Fishers (a GLM, so no label expectation issue):

$$F_x = (\mathrm{diag}(\pi) - \pi \pi^\top) \otimes z z^\top, \qquad z = f(x), \; \pi = \mathrm{softmax}(Wz).$$

Batch selection uses a **forward–backward greedy heuristic**: greedily add $2B$ points one at a time (each minimizing the objective, with rank-one-style updates of the inverse), then greedily *remove* the $B$ least useful — a fix for EPIG's non-submodularity that empirically beats pure greedy.

### 3.8 SIMILAR and PRISM — submodular functions; LogDet variants ≈ EIG / EPIG

**Optimize** submodular information measures built from an "information function" $f$ (non-negative, monotone, submodular over subsets of the pool):

$$H_f(A \mid B) := f(A \cup B) - f(B) \quad (\text{submodular conditional gain}), \qquad I_f(A; B) := f(A) + f(B) - f(A \cup B) \quad (\text{submodular MI}),$$

instantiated with various $f$: facility location, graph cut, set cover, and **log-determinants of similarity matrices**. The framework targets guided selection: pick points informative about a query set / avoid points similar to already-covered sets (useful for rare classes, distribution shift, de-duplication).

**Estimated** with the same ingredients as BADGE — last-layer gradient embeddings with hard pseudo-labels, similarity kernel $S_{ij} = \langle g_i, g_j \rangle$ — and greedy submodular maximization. The paper's result: the **LogDet** instantiations are precisely approximations of Shannon quantities:

$$f_{\text{LogDet}}(A) = \log\det S[A \mid \omega^*] \;\approx\; 2 \cdot \text{EIG}, \qquad \text{(Proposition 7.3)}$$

$$\text{LogDetMI}(A; E) = \log\det S[A] - \log\det\big( S[A] - S[A; E]\, S[E]^{-1} S[E; A] \big) \;\approx\; \text{proxy for EPIG},$$

where $S[A; E]$ is the cross-similarity block between acquisition candidates $A$ and eval/query set $E$. Notably, the LogDet variants are reported among the *best-performing* in both papers — supporting the thesis that "good informativeness score" ≈ "approximate Shannon information quantity."

### 3.9 EGL — Expected Gradient Length

**Optimizes** (modern squared form) the expected squared gradient norm under the model's own predictive distribution:

$$\alpha_{\text{EGL}}(x) = \mathbb{E}_{p(y \mid x, \omega^*)} \big\| H'[y \mid x, \omega^*] \big\|_2^2 = \sum_y p(y \mid x, \omega^*)\, \big\| \nabla_\omega \log p(y \mid x, \omega^*) \big\|_2^2 = \mathrm{tr}\, H''[Y \mid x, \omega^*],$$

i.e., exactly the **trace of the Fisher information** — and via the diagonal/trace approximation (Proposition 7.4), an approximate upper bound on twice the EIG:

$$2\, I[\Omega; Y^{\text{acq}} \mid x^{\text{acq}}] \;\overset{\approx}{\le}\; \alpha_{\text{EGL}}(x^{\text{acq}}) + \text{const}.$$

**Estimated** directly: forward pass for $\pi$, per-class backward passes (or last-layer closed form $\|g\|^2 = \|\pi - e_y\|^2 \|z\|^2$ per class $y$), weighted sum; select top-$k$. Being a per-point scalar, it is additive → same batch pathology as any trace/top-$k$ method, and ignores the posterior "preconditioner" $H''[\omega^* \mid \mathcal{D}]^{-1}$ (an isotropic-posterior assumption).

### 3.10 GraNd — gradient norm score for active sampling

**Optimizes** (as an upper-bound proxy for the **IG**) the expected squared gradient norm of the *known-label* loss over early-training weights:

$$\alpha_{\text{GraNd}}(x, y) = \mathbb{E}_{q(\omega)} \big\| H'[y \mid x, \omega] \big\|_2^2,$$

where $q(\omega)$ is the distribution of parameters at initialization or after a few epochs (not a proper posterior). Proposition 7.5:

$$2\, I[\Omega; y^{\text{acq}} \mid x^{\text{acq}}] \;\overset{\approx}{\le}\; \mathbb{E}_{q(\omega)}\big[\|H'[y^{\text{acq}} \mid x^{\text{acq}}, \omega]\|^2\big] - \mathbb{E}_{q(\omega)}\Big[\mathrm{tr}\,\frac{\nabla_\omega^2\, p(y \mid x, \omega)}{p(y \mid x, \omega)}\Big] + \text{const},$$

where the second (curvature) term is *not* obviously negligible — GraNd may deviate from the information gain.

**Estimated** by averaging $\|\nabla_\omega \log p(y \mid x, \omega_r)\|^2$ over $R$ independent training runs/checkpoints $\omega_r$ (in practice often approximated by the last-layer norm, giving the "EL2N"-style variants); keep the top-scoring fraction of the training set, prune the rest.

---

## 4. Summary Table

| Method | Objective (quantity) | Space | Estimator | Batch selection |
|---|---|---|---|---|
| BALD | EIG $I[\Omega; Y \mid x]$ | Prediction | MC: $H[\bar\pi] - \overline{H[\pi_n]}$ over $N$ posterior samples | top-$k$ (pathological) |
| BatchBALD | joint EIG $I[\Omega; Y_{1:B} \mid x_{1:B}]$ | Prediction | MC joint entropy over label configs | greedy (submodular, $1{-}1/e$) |
| EPIG | $\mathbb{E}_{x^{e}} I[Y^{e}; Y^{a}]$ | Prediction | MC joint predictive of (acq, eval) pairs | greedy (no guarantee) |
| JEPIG | $I[\{Y_i^{e}\}; Y^{a}]$ | Prediction | MC joint entropy over eval configs | greedy (no guarantee) |
| IG/PIG/JPIG | labeled analogues | Weight (or RHO-loss) | observed info in log-det/trace; RHO: loss − holdout loss | top-$k$ / greedy |
| BADGE | EIG (uninf. prior) | Weight (last layer) | $\log\det S$ via pseudo-label gradient embeddings; k-means++/k-DPP | diversity sampling |
| BAIT | EPIG (trace form) | Weight (last layer) | $\mathrm{tr}(F_{\text{eval}}(F_{\text{acq}} + F_{\text{train}} + \lambda I)^{-1})$ | forward–backward greedy |
| SIMILAR/PRISM (LogDet) | EIG / EPIG proxies | Weight (last layer) | $\log\det$ of (conditional) similarity matrices | greedy submodular |
| EGL | $\lessapprox$ bound on $2\,$EIG | Weight | $\mathrm{tr}\, F_x = \sum_y \pi_y \|\nabla \log p(y \mid x)\|^2$ | top-$k$ (pathological) |
| GraNd | $\lessapprox$ bound on $2\,$IG | Weight | $\mathbb{E}_{q(\omega)} \|\nabla \log p(y \mid x, \omega)\|^2$ over runs | top-$k$ |

---

## 5. Caveats (paper's own)

- **Laplace quality**: the Gaussian posterior approximation is worst exactly where active learning matters (little data, multimodal loss landscape).
- **Last layer**: selects data on *fixed embeddings*, ignoring feature learning — fine for fine-tuning pretrained models, questionable for training from scratch.
- **Hard pseudo-labels** (BADGE, SIMILAR, PRISM): biased Fisher estimates; one-sample soft-label estimates ($y \sim p(y \mid x, \omega^*)$) are unbiased and untested.
- **Trace vs. log-det**: trace bounds are additive → top-$k$ batch pathologies; log-det captures redundancy between candidates.
- **Prediction vs. weight space**: prediction space avoids the Gaussian assumption but pays a combinatorial cost in batch size and MC variance; weight space is cheap per batch but inherits all approximations above.
