# Vendi Score & Active Learning: Speedup Methods

## Context

The **Vendi Score (VS)** is a diversity metric defined as the exponential of the Shannon entropy of the eigenvalues of a kernel matrix:

$$\text{VS}_k(x_1, \ldots, x_n) = \exp\left(-\sum_{i=1}^n \lambda_i \log \lambda_i\right)$$

where $\lambda_1, \ldots, \lambda_n$ are the eigenvalues of $K/n$, and $K_{ij} = k(x_i, x_j)$ is the kernel matrix induced by a user-defined similarity function $k$.

### The Active Learning Problem

In active learning with labeled set $L$ and unlabeled set $U$, a naive diversity-based acquisition loop looks like:

```python
for u in U:
    scores.append(calc_vendi_score(L + {u}))
query = argmax(scores)
```

This requires computing eigenvalues of an $(|L|+1) \times (|L|+1)$ matrix **$|U|$ times per iteration**, which is $O(|U| \cdot |L|^3)$ — very slow.

---

## Notation

| Symbol | Meaning |
|--------|---------|
| $n = \|L\|$ | Size of labeled set |
| $m$ | Number of Nyström landmarks ($m \ll n$) |
| $\|U\|$ | Size of unlabeled candidate set |
| $k$ | Number of top eigenvalues to approximate |
| $d$ | Embedding dimension |

---

## Method 1: Nyström Approximation

### Idea

Approximate $K_L \approx C W^{-1} C^\top$ using $m$ landmark points $z_1, \ldots, z_m$ sampled from $L$.

- $W \in \mathbb{R}^{m \times m}$: $W_{ij} = k(z_i, z_j)$
- $C \in \mathbb{R}^{n \times m}$: $C_{ij} = k(x_i, z_j)$ for all $x_i \in L$

The nonzero eigenvalues of $CW^{-1}C^\top$ equal the eigenvalues of the $m \times m$ matrix:

$$M = W^{-1/2} C^\top C W^{-1/2}$$

This follows from the identity: for any matrix $A$, $AA^\top$ and $A^\top A$ share the same nonzero eigenvalues. Set $A = CW^{-1/2}$.

### Setup Steps

1. Compute $W^{-1/2}$ via Cholesky or eigendecomposition of $W$: $O(m^3)$
2. Form $B = W^{-1/2} C^\top \in \mathbb{R}^{m \times n}$: $O(nm^2)$
3. Form $M = BB^\top = W^{-1/2} C^\top C W^{-1/2} \in \mathbb{R}^{m \times m}$: $O(nm^2)$
4. Eigendecompose $M = P\Gamma P^\top$: $O(m^3)$

### Per-Candidate Update

When evaluating candidate $u$, append one row $c_u = [k(u, z_1), \ldots, k(u, z_m)]$ to $C$:

$$M_\text{ext} = M + \mathbf{v}\mathbf{v}^\top, \quad \mathbf{v} = W^{-1/2} c_u \in \mathbb{R}^m$$

This is a **rank-1 update** of $M$. Apply the secular equation (see Method 3) on $M$ instead of $K_L$. Cost: $O(m^2)$ per candidate.

### Complexity

| Step | Cost |
|------|------|
| Compute $W$ | $O(m^2)$ |
| Compute $C$ | $O(nm)$ kernel evals |
| Compute $W^{-1/2}$ | $O(m^3)$ |
| Form $M = W^{-1/2}C^\top C W^{-1/2}$ | $O(nm^2)$ |
| Eigendecompose $M$ | $O(m^3)$ |
| Per candidate: compute $\mathbf{v} = W^{-1/2} c_u$ | $O(m^2)$ |
| Per candidate: rank-1 secular update on $M$ | $O(m^2)$ |
| **Total** | $O(nm^2 + \|U\| \cdot m^2)$ |

> **Note:** The naive version (recomputing SVD from scratch each candidate) costs $O(nm^2)$ per candidate due to SVD of the $(n+1) \times m$ matrix $B_\text{ext}$. The key optimization is working entirely in the $m$-dimensional space using the rank-1 update.

### Speedup Factor

$(n/m)^2$ over naively eigendecomposing the full $n \times n$ matrix. If $n=500$, $m=50$: **100× faster**.

### References

- Williams & Seeger (2001). *Using the Nyström Method to Speed Up Kernel Machines.* NeurIPS 13, 682–688. — origin of the method in ML; the $K \approx CW^{-1}C^\top$ form used here.
- Drineas & Mahoney (2005). *On the Nyström Method for Approximating a Gram Matrix for Improved Kernel-Based Learning.* JMLR 6, 2153–2175. — error bounds; shows uniform landmark sampling is the weak link.
- Gittens & Mahoney (2016). *Revisiting the Nyström Method for Improved Large-Scale Machine Learning.* JMLR 17(117), 1–65. — spectral-norm/trace-norm bounds; when $m$ landmarks suffice depends on the spectral decay of $K$.
- Musco & Musco (2017). *Recursive Sampling for the Nyström Method.* NeurIPS 30. — ridge-leverage-score landmark selection, strictly better than the uniform `rng.choice` used in the benchmark.
- Kumar, Mohri & Talwalkar (2012). *Sampling Methods for the Nyström Method.* JMLR 13, 981–1006. — empirical comparison of landmark sampling schemes.

---

## Method 2: Randomized SVD

### Idea

Estimate the top-$k$ eigenvalues of $K$ using random projections instead of full eigendecomposition.

1. Draw random Gaussian $\Omega \in \mathbb{R}^{n \times k}$
2. Form $Y = K\Omega$ (sketches the range of $K$): $O(n^2 k)$
3. Orthonormalize $Y$ via QR to get $Q \in \mathbb{R}^{n \times k}$
4. Form $B = Q^\top K Q \in \mathbb{R}^{k \times k}$
5. Eigendecompose $B$: $O(k^3)$

The eigenvalues of $B$ approximate the top-$k$ eigenvalues of $K$.

### Why This Fixes LOBPCG

LOBPCG fails when eigenvalues are clustered or near-duplicate (common in low-rank kernels). Randomized SVD:
- Does not rely on spectral gap between eigenvalues
- Targets only the top-$k$ eigenvalues, never touching ill-conditioned small ones
- Is embarrassingly parallel (all columns of $\Omega$ at once)

### Relationship to Krylov Methods

| | Krylov (Lanczos/LOBPCG) | Randomized SVD |
|---|---|---|
| Subspace construction | Structured: $Kv, K^2v, \ldots$ | Random: $K\Omega$ for random $\Omega$ |
| Convergence | Fast for well-separated eigenvalues | Uniform, independent of spectrum |
| Failure mode | Clustered/duplicate eigenvalues | Rarely fails |
| Parallelism | Sequential | Embarrassingly parallel |
| Cost | $O(kn^2)$ total | $O(n^2 k)$ total, one BLAS call |

**Power iteration variant:** Compute $Y = (KK^\top)^q K\Omega$ for $q=1$ or $q=2$ to improve accuracy for slowly-decaying spectra.

```python
from sklearn.utils.extmath import randomized_svd
import numpy as np

def randomized_vendi(K, k=20):
    n = K.shape[0]
    U, s, Vt = randomized_svd(K / n, n_components=k, random_state=0)
    lambdas = s  # top-k eigenvalues of K/n

    # Assign residual mass to preserve sum-to-1
    residual = max(1.0 - lambdas.sum(), 0)
    if residual > 1e-10:
        lambdas = np.append(lambdas, residual)

    lambdas = lambdas[lambdas > 1e-10]
    return np.exp(-np.sum(lambdas * np.log(lambdas)))
```

**Choosing $k$:** Plot eigenvalue spectrum; pick $k$ to capture ~95% of the trace.

### References

- Halko, Martinsson & Tropp (2011). *Finding Structure with Randomness: Probabilistic Algorithms for Constructing Approximate Matrix Decompositions.* SIAM Review 53(2), 217–288. — the canonical reference; Algorithm 4.3 (randomized range finder) and 4.4 (power iteration) are exactly what `sklearn.utils.extmath.randomized_svd` implements.
- Rokhlin, Szlam & Tygert (2010). *A Randomized Algorithm for Principal Component Analysis.* SIAM J. Matrix Anal. Appl. 31(3), 1100–1124. — the power-iteration variant $Y = (KK^\top)^q K\Omega$.
- Musco & Musco (2015). *Randomized Block Krylov Methods for Stronger and Faster Approximate Singular Value Decomposition.* NeurIPS 28. — block Krylov gets the same accuracy in $O(1/\sqrt{\epsilon})$ rather than $O(1/\epsilon)$ passes; the better choice if `n_iter` has to be raised.
- Martinsson & Tropp (2020). *Randomized Numerical Linear Algebra: Foundations and Algorithms.* Acta Numerica 29, 403–572. — survey, incl. guidance on oversampling ($k + p$ with $p \approx 5$–10, which the code above omits).
- Knyazev (2001). *Toward the Optimal Preconditioned Eigensolver: Locally Optimal Block Preconditioned Conjugate Gradient Method.* SIAM J. Sci. Comput. 23(2), 517–541. — LOBPCG, the method this one replaces; its convergence rate depends on relative eigenvalue gaps, hence the failure on clustered spectra.

---

## Method 3: Rank-1 Secular Equation Update ⭐ (Recommended)

### Setup

Precompute eigendecomposition of $K_L$ once per AL iteration:

$$K_L = Q\Lambda Q^\top, \quad \Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n)$$

When adding candidate $u$, the augmented matrix is:

$$K_{L \cup \{u\}} = \begin{bmatrix} K_L & \mathbf{k}_u \\ \mathbf{k}_u^\top & 1 \end{bmatrix}$$

### Deriving the Secular Equation

Rotate into the eigenbasis of $K_L$:

$$\tilde{K} = \begin{bmatrix} Q^\top & 0 \\ 0 & 1 \end{bmatrix} K_{L \cup \{u\}} \begin{bmatrix} Q & 0 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} \Lambda & \mathbf{d} \\ \mathbf{d}^\top & 1 \end{bmatrix}$$

where $\mathbf{d} = Q^\top \mathbf{k}_u \in \mathbb{R}^n$.

Apply the matrix determinant lemma to $\det(\tilde{K} - \mu I) = 0$:

$$\det(\tilde{K} - \mu I) = \prod_{i=1}^n (\lambda_i - \mu) \cdot \left(1 - \mu - \sum_{i=1}^n \frac{d_i^2}{\lambda_i - \mu}\right) = 0$$

Setting the second factor to zero gives the **secular equation**:

$$\boxed{f(\mu) = 1 - \mu - \sum_{i=1}^n \frac{d_i^2}{\lambda_i - \mu} = 0}$$

### Root Structure and Interlacing

$f(\mu)$ has poles at each $\lambda_i$, dividing the real line into $n+1$ intervals. In each interval $f$ is strictly monotone decreasing (since $f'(\mu) = -1 - \sum_i d_i^2/(\lambda_i - \mu)^2 < 0$), so there is **exactly one root per interval**: $n-1$ interior roots plus one root above $\lambda_n$.

This yields the **interlacing property** (Cauchy interlacing theorem):

$$\mu_1 \leq \lambda_1 \leq \mu_2 \leq \lambda_2 \leq \cdots \leq \mu_n \leq \lambda_n \leq \mu_{n+1}$$

### Numerical Solution

Use bisection (guaranteed to converge) or Newton's method (typically 2–3 steps):

$$f'(\mu) = -1 - \sum_{i=1}^n \frac{d_i^2}{(\lambda_i - \mu)^2}$$

Safe bracket for the rightmost root: $(\lambda_n, \; \lambda_n + \|\mathbf{d}\|^2 + 1)$.

### Edge Case: $d_i = 0$

If $d_i = 0$ for some $i$, then $\lambda_i$ is itself a root — $u$ has no component along eigenvector $i$. In floating point, near-zero $d_i$ causes near-pole instability; add a small `eps` guard.

```python
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import brentq

def precompute_eigen(K_L):
    # eigh: stable, for symmetric matrices, returns ascending eigenvalues
    lambdas, Q = eigh(K_L + 1e-8 * np.eye(len(K_L)))
    return lambdas, Q

def secular_roots(lambdas, d, eps=1e-12):
    n = len(lambdas)

    def f(mu):
        denom = lambdas - mu
        denom = np.where(np.abs(denom) < eps, eps, denom)
        return 1.0 - mu - np.sum(d**2 / denom)

    roots = []

    # One root in each interior interval (lambda_i, lambda_{i+1})
    for i in range(n - 1):
        lo, hi = lambdas[i] + eps, lambdas[i+1] - eps
        try:
            roots.append(brentq(f, lo, hi, xtol=1e-10))
        except ValueError:
            pass  # d_i = 0 case: lambda_i is the root

    # One root above lambda_n
    lo = lambdas[-1] + eps
    hi = lambdas[-1] + np.sum(d**2) + 1.0
    try:
        roots.append(brentq(f, lo, hi, xtol=1e-10))
    except ValueError:
        pass

    # One root below lambda_1
    lo = lambdas[0] - np.sum(d**2) - 1.0
    hi = lambdas[0] - eps
    try:
        roots.append(brentq(f, lo, hi, xtol=1e-10))
    except ValueError:
        pass

    return np.sort(roots)

def secular_vendi(lambdas, Q, k_u):
    n = len(lambdas)
    d = Q.T @ k_u                     # O(n^2)
    mu = secular_roots(lambdas, d)    # O(n^2)

    mu_norm = np.array(mu) / (n + 1)
    mu_norm = mu_norm[mu_norm > 1e-10]
    return np.exp(-np.sum(mu_norm * np.log(mu_norm)))

# --- Active learning loop ---
# lambdas, Q = precompute_eigen(K_L)   # O(n^3), once per iteration
# K_LU = X_L_norm @ X_U_norm.T        # O(n * d * |U|), once
# for i, u in enumerate(U):
#     vs = secular_vendi(lambdas, Q, K_LU[:, i])
```

### Complexity

| Step | Cost |
|------|------|
| Precompute $K_L = Q\Lambda Q^\top$ | $O(n^3)$ once per AL iteration |
| Compute $\mathbf{d} = Q^\top \mathbf{k}_u$ | $O(n^2)$ per candidate |
| Solve secular equation ($n+1$ roots) | $O(n^2)$ per candidate |
| **Total per AL iteration** | $O(n^3 + \|U\| \cdot n^2)$ |

### References

- Golub (1973). *Some Modified Matrix Eigenvalue Problems.* SIAM Review 15(2), 318–334. — the original derivation of the secular equation for rank-1 modifications of a symmetric eigenproblem.
- Bunch, Nielsen & Sorensen (1978). *Rank-One Modification of the Symmetric Eigenproblem.* Numerische Mathematik 31, 31–48. — the standard algorithm, including the deflation rules for the $d_i \approx 0$ and repeated-$\lambda_i$ edge cases noted above.
- Cuppen (1981). *A Divide and Conquer Method for the Symmetric Tridiagonal Eigenproblem.* Numerische Mathematik 36, 177–195. — where rank-1 updating became the inner loop of a practical eigensolver.
- Gu & Eisenstat (1994). *A Stable and Efficient Algorithm for the Rank-One Modification of the Symmetric Eigenproblem.* SIAM J. Matrix Anal. Appl. 15(4), 1266–1276. — fixes the loss of eigenvector orthogonality in Bunch–Nielsen–Sorensen; the algorithm LAPACK actually ships.
- Li (1993). *Solving Secular Equations Stably and Efficiently.* LAPACK Working Note 89 / UT-CS-93-260. — the rational-interpolation root finder used by LAPACK `dlaed4`, which converges in ~2–3 iterations versus `brentq`'s ~40 and is the single biggest available speedup to the implementation in the notebook.
- LAPACK routines `dlaed4` (one secular root), `dlaed9`/`dlaed3` (all roots + eigenvectors) — reference implementations.
- Horn & Johnson (2013). *Matrix Analysis*, 2nd ed., §4.3. — Cauchy interlacing theorem, which guarantees exactly one root per interval and hence the bracketing used by the solver.

---

## Method 4: Batch Kernel Precomputation

Not an approximation — eliminates redundant kernel evaluations.

If using an embedding-based kernel $k(x, y) = \phi(x)^\top \phi(y) / (\|\phi(x)\| \|\phi(y)\|)$:

```python
X_L_norm = X_L / np.linalg.norm(X_L, axis=1, keepdims=True)  # (n, d)
X_U_norm = X_U / np.linalg.norm(X_U, axis=1, keepdims=True)  # (|U|, d)

K_LL = X_L_norm @ X_L_norm.T   # (n, n) — once
K_LU = X_L_norm @ X_U_norm.T   # (n, |U|) — once, all cross-similarities

for i, u in enumerate(U):
    k_u = K_LU[:, i]            # free
    # ... pass k_u to secular_vendi or other method
```

Combine with Method 3 for the best overall approach.

### References

- Schölkopf & Smola (2002). *Learning with Kernels*, MIT Press, §2.2. — kernel matrices from normalised embeddings; the cosine kernel as an inner product in feature space.
- Goto & van de Geijn (2008). *Anatomy of High-Performance Matrix Multiplication.* ACM TOMS 34(3). — why one $n \times d \times |U|$ GEMM beats $|U|$ separate $n \times d$ GEMVs by an order of magnitude at equal FLOP count (BLAS-3 vs BLAS-2).
- Charlier, Feydy, Glaunès et al. (2021). *Kernel Operations on the GPU, with Autodiff, without Memory Overflows.* JMLR 22(74) (KeOps). — the alternative when $K_{LU}$ does not fit in memory: symbolic kernel matrices with no $n \times |U|$ materialisation.

---

## Method 5: Residual Norm Surrogate

Replace VS computation with a cheap proxy. Adding $u$ increases diversity most when $u$ is not well-explained by the existing eigenspace of $K_L$.

$$\text{residual}(u) = 1 - \|Q_k^\top \mathbf{k}_u\|^2$$

where $Q_k$ contains the top-$k$ eigenvectors of $K_L$. Higher residual = more diverse = better candidate. This is equivalent to the greedy DPP criterion.

```python
def residual_surrogate(Q_topk, k_u):
    # Q_topk: (n, k) top-k eigenvectors
    # k_u: (n,) kernel vector for candidate u
    projection = Q_topk.T @ k_u   # (k,)
    return 1.0 - np.dot(projection, projection)

# Q_topk = Q[:, -k:]  # eigh returns ascending order
```

Cost: $O(nk)$ per candidate. Not exactly VS, but theoretically motivated.

### References

- Kulesza & Taskar (2012). *Determinantal Point Processes for Machine Learning.* Foundations and Trends in ML 5(2–3), 123–286. — §2.2 gives the Schur-complement identity $\det(K_{L \cup u}) = \det(K_L)\,(k_{uu} - \mathbf{k}_u^\top K_L^{-1} \mathbf{k}_u)$ that this surrogate is a rank-truncated form of.
- Chen, Zhang & Zhou (2018). *Fast Greedy MAP Inference for Determinantal Point Process to Improve Recommendation Diversity.* NeurIPS 31. — the $O(nk)$ incremental-Cholesky greedy DPP loop; the correct implementation if the goal is the exact greedy-DPP criterion rather than a proxy for it.
- Sener & Savarese (2018). *Active Learning for Convolutional Neural Networks: A Core-Set Approach.* ICLR. — the closest AL analogue: select the candidate farthest from the current labeled set's coverage.
- Seeger, Williams & Lawrence (2003). *Fast Forward Selection to Speed Up Sparse Gaussian Process Regression.* AISTATS. — the same quantity read as GP posterior variance at $u$ given $L$; the standard justification for using it as an informativeness score.
- Alaoui & Mahoney (2015). *Fast Randomized Kernel Ridge Regression with Statistical Guarantees.* NeurIPS 28. — ridge leverage scores, the principled version of "how much of $\mathbf{k}_u$ lies outside the top-$k$ eigenspace".

---

## Summary

| Method | Setup | Per-candidate | Exact? | Best when |
|--------|-------|---------------|--------|-----------|
| Nyström + secular | $O(nm^2)$ | $O(m^2)$ | No | $n$ large, kernel low-rank |
| Randomized SVD | $O(n^2 k)$ | $O(nk)$ | No | LOBPCG broken, need top-$k$ |
| Secular equation | $O(n^3)$ | $O(n^2)$ | Yes | $n$ moderate, $\|U\|$ large |
| Batch kernel | $O(nd\|U\|)$ | $O(1)$ | Yes | Kernel eval is bottleneck |
| Residual surrogate | $O(n^3)$ | $O(nk)$ | No (proxy) | Speed >> fidelity |

### Recommended Combination

- Always use **Method 4** (batch kernel precomputation) — it's free.
- For moderate $n$ (up to ~1000): combine with **Method 3** (secular equation) for exact VS at $O(n^2)$ per candidate.
- For large $n$: use **Nyström + secular** (Method 1 + 3 combined on the $m \times m$ matrix) for $O(m^2)$ per candidate.
- If you just need a fast proxy: **Method 5** (residual surrogate) at $O(nk)$.

---

## References

Per-method references are listed at the end of each method section. The entries below cover the Vendi Score itself and prior work on making it scalable.

### The Vendi Score

- Friedman & Dieng (2023). *The Vendi Score: A Diversity Evaluation Metric for Machine Learning.* TMLR. [arXiv:2210.02410](https://arxiv.org/abs/2210.02410) — the definition used throughout; note §3 already observes the $O(n^3)$ cost and the $\exp(H)$-of-eigenvalues form.
- Pasarkar & Dieng (2024). *Cousins of the Vendi Score: A Family of Similarity-Based Diversity Metrics for Science and Machine Learning.* AISTATS. [arXiv:2310.12952](https://arxiv.org/abs/2310.12952) — the order-$q$ family $\text{VS}_q$; $q \neq 1$ (e.g. $q=2$, a trace of $K^2$) avoids eigendecomposition entirely and is worth benchmarking as a sixth method.
- Nguyen & Dieng (2024). *Quality-Weighted Vendi Scores and Their Application to Diverse Experimental Design.* ICML. [arXiv:2405.02449](https://arxiv.org/abs/2405.02449) — the closest published analogue to the AL loop targeted here (diversity-aware batch selection).
- Ospanov, Zhang, Jalali, Cao, Bogdanov & Farnia (2024). *Towards a Scalable Reference-Free Evaluation of Generative Models.* NeurIPS. [arXiv:2407.02961](https://arxiv.org/abs/2407.02961) — FKEA: approximates the Vendi Score with random Fourier features in $O(n)$ time for shift-invariant kernels. Directly competes with Methods 1–2 and should be cited as prior art for any scalability claim.

### Approximation methods

- Nyström: Williams & Seeger (2001); Drineas & Mahoney (2005); Gittens & Mahoney (2016); Musco & Musco (2017); Kumar, Mohri & Talwalkar (2012). — see Method 1.
- Randomized SVD: Halko, Martinsson & Tropp (2011); Rokhlin, Szlam & Tygert (2010); Musco & Musco (2015); Martinsson & Tropp (2020); Knyazev (2001). — see Method 2.
- Secular equation / rank-1 update: Golub (1973); Bunch, Nielsen & Sorensen (1978); Cuppen (1981); Gu & Eisenstat (1994); Li (1993, LAWN 89); LAPACK `dlaed4`; Horn & Johnson (2013, §4.3). — see Method 3.
- Kernel batching: Schölkopf & Smola (2002); Goto & van de Geijn (2008); Charlier et al. (2021, KeOps). — see Method 4.
- DPP / residual surrogate: Kulesza & Taskar (2012); Chen, Zhang & Zhou (2018); Sener & Savarese (2018); Seeger, Williams & Lawrence (2003); Alaoui & Mahoney (2015). — see Method 5.
