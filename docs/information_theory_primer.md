# Information Theory Primer for Kirsch & Gal (arXiv:2208.00549v2)

A self-contained primer for reading Section 2 of the paper and the EPIG-related material in `active_learning_methods.md`. Assumes probability theory but no information theory. All quantities below use discrete sums; replace with integrals for densities.

---

## 1. Entropy: uncertainty as expected surprise

For an outcome $x$ with probability $p(x)$, define the **information content** (surprise)

$$H(p(x)) := -\log p(x). \tag{paper eq. 1}$$

Rare outcomes are more surprising: $p(x) \to 0 \Rightarrow -\log p(x) \to \infty$; a certain outcome has zero surprise. The log makes surprise additive for independent events: $-\log p(x)q(y) = -\log p(x) - \log q(y)$.

The **entropy** of a random variable $X$ is the *expected* surprise:

$$H[X] := \mathbb{E}_{p(x)}[-\log p(x)] = -\sum_x p(x)\log p(x). \tag{paper eqs. 2, 4}$$

Properties: $H[X] \ge 0$ (discrete case); $H[X] = 0$ iff $X$ is deterministic; maximized by the uniform distribution ($\log C$ for $C$ outcomes). Interpretation: how uncertain you are about $X$ before observing it, or (Shannon) the minimum expected number of nats/bits to encode a draw of $X$.

**Cross-entropy** (eq. 3) generalizes this to two distributions: $H(p \,\|\, q) = \mathbb{E}_{p(x)}[-\log q(x)]$ is your expected surprise when reality follows $p$ but you model it with $q$. Always $H(p\,\|\,q) \ge H(p\,\|\,p) = H[X]$, and the gap is the KL divergence:

$$D_{\mathrm{KL}}(p \,\|\, q) = \mathbb{E}_{p(x)}\left[\log \frac{p(x)}{q(x)}\right] = H(p\,\|\,q) - H[X] \;\ge\; 0,$$

with equality iff $p = q$. KL is the fundamental "distance" (asymmetric) of the subject; everything below reduces to it.

### Notation convention in the paper

The paper overloads $H$: applied to an *outcome* or distribution expression it means pointwise information content, $H(p(x)) = -\log p(x)$; applied to a *random variable* in brackets it means the expectation, $H[X]$. Capital letters ($X$, $Y$, $\Omega$) are random variables and get averaged over; lowercase ($x$, $y$, $\omega$) are fixed outcomes and do not. This distinction is the entire content of eq. (7) — see §4.

---

## 2. Joint and conditional entropy

Joint entropy is just entropy of the pair: $H[X, Y] = \mathbb{E}_{p(x,y)}[-\log p(x,y)]$.

Two conditional notions must be kept apart:

**(a) Conditioned on a specific outcome** $Y = y$:

$$H[X \mid Y = y] = \mathbb{E}_{p(x \mid y)}[-\log p(x \mid y)].$$

This is an ordinary entropy — of the distribution $p(x \mid y)$ — and is a function of $y$.

**(b) The conditional entropy** $H[X \mid Y]$, which additionally averages over $y$:

$$H[X \mid Y] = \mathbb{E}_{p(y)}\big[H[X \mid Y = y]\big] = \mathbb{E}_{p(x,y)}[-\log p(x \mid y)]. \tag{paper eq. 5}$$

This is the point the paper flags with "note that we also take an expectation over $y$": $H[X \mid Y]$ is a single number, the uncertainty about $X$ you *expect* to have left after observing $Y$, averaged over what $Y$ might turn out to be.

**Chain rule** (from $p(x,y) = p(y)\,p(x \mid y)$ and linearity of expectation):

$$H[X, Y] = H[Y] + H[X \mid Y].$$

Conditioning never hurts *in expectation*: $H[X \mid Y] \le H[X]$. (A specific outcome can increase uncertainty — $H[X \mid Y=y] > H[X]$ is possible — but not on average.)

---

## 3. Mutual information

$$I[X; Y] := H[X] - H[X \mid Y]. \tag{paper eq. 6}$$

Read it as **expected uncertainty reduction**: how much observing $Y$ is expected to shrink your uncertainty about $X$. Equivalent forms, all worth internalizing:

$$I[X; Y] = H[X] - H[X \mid Y] = H[Y] - H[Y \mid X] = H[X] + H[Y] - H[X, Y]$$

$$= D_{\mathrm{KL}}\big(p(x,y) \,\|\, p(x)\,p(y)\big) = \mathbb{E}_{p(y)}\Big[D_{\mathrm{KL}}\big(p(x \mid y) \,\|\, p(x)\big)\Big].$$

Consequences: $I[X;Y] \ge 0$ (it is a KL); $I[X;Y] = 0$ iff $X \perp Y$; it is **symmetric**, $I[X;Y] = I[Y;X]$ — even when only one direction is easy to compute, the other is available for free (this gets used constantly in the paper). The last form is the Bayesian reading: MI is the expected KL between posterior and prior, i.e. the expected size of the belief update.

**Conditional mutual information** is the same object under an extra conditioning:

$$I[X; Y \mid Z] := H[X \mid Z] - H[X \mid Y, Z] = \mathbb{E}_{p(z)}\Big[D_{\mathrm{KL}}\big(p(x,y \mid z) \,\|\, p(x \mid z)\,p(y \mid z)\big)\Big] \ge 0.$$

It measures the dependence between $X$ and $Y$ that remains once $Z$ is accounted for. Note that $I[X;Y \mid Z]$ can be larger *or* smaller than $I[X;Y]$: conditioning can destroy dependence ($Z$ a common cause) or create it ($Z$ a common effect — "explaining away").

---

## 4. Equation (7): mixing random variables and outcomes

$$H[X, y \mid Z] := \mathbb{E}_{p(x, z \mid y)}[-\log p(x, y \mid z)]. \tag{paper eq. 7}$$

This defines what an entropy means when some arguments are random variables and some are fixed outcomes. The recipe:

1. **Fixed outcomes stay fixed everywhere.** Here $y$ is a specific value; it is not averaged over.
2. **The expectation runs over all remaining random variables, *conditioned on* the fixed outcomes.** Hence $\mathbb{E}_{p(x, z \mid y)}$ — you average $X$ and $Z$ under their joint distribution given that $Y = y$ actually happened.
3. **Inside the log sits the full expression as written**, $-\log p(x, y \mid z)$, with $y$ plugged in as a constant.

So eq. (7) is not a new concept — it is bookkeeping. $H[X, y \mid Z]$ is "the expected surprise of the pair $(X, Y=y)$ given $Z$, in the world where $y$ was observed." The capital/lowercase convention tells you exactly which expectations to take: every capital letter gets integrated out; every lowercase letter is data.

Sanity checks with the recipe:

- $H[Y \mid x] = \mathbb{E}_{p(y \mid x)}[-\log p(y \mid x)]$ — the predictive entropy at a *specific* input $x$. No expectation over $x$.
- $H[Y \mid x, \Omega] = \mathbb{E}_{p(\omega)}\,\mathbb{E}_{p(y \mid x, \omega)}[-\log p(y \mid x, \omega)]$ — average over the random parameters $\Omega$, keep $x$ fixed.
- $H[X \mid Y]$ with everything capital recovers eq. (5).

A caveat specific to this paper's discriminative setting: inputs $x$ are never modeled as random (eq. 9 conditions on them throughout), so conditioning on $x^{\text{acq}}$, $x^{\text{eval}}$ merely *parametrizes* the distributions. Only labels $Y$ and parameters $\Omega$ ever carry probability mass. When you see $X^{\text{eval}}$ capitalized (e.g. in the EPIG proxy), it denotes an expectation over the empirical distribution $\hat p(x^{\text{eval}})$ of evaluation inputs, not a modeled generative distribution.

---

## 5. The Bayesian model and EIG/BALD

Setup (paper eq. 8): parameters $\Omega \sim p(\omega)$; predictions $p(y \mid x, \omega)$; labels conditionally independent given $\omega$. The marginal predictive is

$$p(y \mid x) = \mathbb{E}_{p(\omega)}[p(y \mid x, \omega)].$$

**EIG/BALD** scores a candidate input $x^{\text{acq}}$ by the mutual information between its (unknown) label and the parameters:

$$I[\Omega; Y^{\text{acq}} \mid x^{\text{acq}}] = \underbrace{H[Y^{\text{acq}} \mid x^{\text{acq}}]}_{\text{total predictive uncertainty}} - \underbrace{H[Y^{\text{acq}} \mid x^{\text{acq}}, \Omega]}_{\text{expected uncertainty if } \omega \text{ were known}}.$$

Apply the eq.-(7) recipe to both terms: $x^{\text{acq}}$ is fixed throughout; the first term is the entropy of the marginal predictive $p(y \mid x^{\text{acq}})$; the second averages the per-$\omega$ predictive entropy over the posterior. Their difference is the **disagreement** between posterior samples: it is large exactly when individual models are confident ($H[Y \mid x, \omega]$ small) but disagree with each other ($H[Y \mid x]$ large). Equivalently, by symmetry and the KL form of MI,

$$I[\Omega; Y \mid x] = \mathbb{E}_{p(y \mid x)}\Big[D_{\mathrm{KL}}\big(p(\omega \mid y, x) \,\|\, p(\omega)\big)\Big]:$$

the expected size of the posterior update from labeling $x$ — hence "expected information gain."

---

## 6. EPIG, term by term

$$\alpha_{\text{EPIG}}(x^{\text{acq}}) = \mathbb{E}_{\hat p(x^{\text{eval}})}\Big[\, I[Y^{\text{eval}}; Y^{\text{acq}} \mid x^{\text{eval}}, x^{\text{acq}}] \,\Big].$$

Decode it with everything above:

**The conditioning.** Both $x^{\text{eval}}$ and $x^{\text{acq}}$ are lowercase — fixed inputs. So the conditional MI here involves *no* extra expectation from the conditioning; it is a plain mutual information between two label variables, computed under distributions parametrized by the two inputs.

**The two random variables.** $Y^{\text{acq}}$ is the label you would get by annotating the candidate; $Y^{\text{eval}}$ is the model's prediction at a test-like input. Neither is observed; both are random through the shared posterior $\Omega$. Given $\omega$ they are independent, so their joint predictive is

$$p(y^a, y^e \mid x^{\text{acq}}, x^{\text{eval}}) = \mathbb{E}_{p(\omega)}\big[\,p(y^a \mid x^{\text{acq}}, \omega)\;p(y^e \mid x^{\text{eval}}, \omega)\,\big],$$

which is generally **not** a product of its marginals — marginalizing $\omega$ couples the labels. That coupling is exactly what the MI measures:

$$I[Y^{\text{eval}}; Y^{\text{acq}} \mid x^{\text{eval}}, x^{\text{acq}}] = D_{\mathrm{KL}}\big(p(y^a, y^e \mid \cdot) \,\|\, p(y^a \mid \cdot)\,p(y^e \mid \cdot)\big) = \sum_{y^a, y^e} p(y^a, y^e) \log \frac{p(y^a, y^e)}{p(y^a)\,p(y^e)}.$$

This is precisely the Monte Carlo estimator in §3.3 of `active_learning_methods.md`: sample $\omega_n$, average products of predictives to get $\hat p(y^a, y^e)$, plug into the KL sum.

**The reading.** Using $I[Y^e; Y^a] = H[Y^e] - H[Y^e \mid Y^a]$ and the expected-KL form:

$$I[Y^{\text{eval}}; Y^{\text{acq}} \mid x^{\text{eval}}, x^{\text{acq}}] = \mathbb{E}_{p(y^a \mid x^{\text{acq}})}\Big[D_{\mathrm{KL}}\big(p(y^e \mid x^{\text{eval}}, x^{\text{acq}}, y^a) \,\|\, p(y^e \mid x^{\text{eval}})\big)\Big]:$$

*how much do I expect my prediction at $x^{\text{eval}}$ to change if someone tells me the label at $x^{\text{acq}}$?* (Here $p(y^e \mid \ldots, y^a)$ is the predictive after a Bayesian update on the hypothetical label — conditioning on $y^a$ reweights the posterior over $\omega$.)

**The outer expectation.** $\hat p(x^{\text{eval}})$ is an empirical distribution of test-like inputs (the pool, or held-out unlabeled data). EPIG averages the above over where predictions will actually be needed.

**Why this fixes BALD.** BALD asks "does this label tell me about $\Omega$?" — an outlier can score high by pinning down parameter directions that no test point ever exercises. EPIG routes the information *through* $\Omega$ to the predictions: information about parameters only counts insofar as it propagates to $Y^{\text{eval}}$ at typical inputs. Formally the chain $Y^{\text{acq}} \leftrightarrow \Omega \leftrightarrow Y^{\text{eval}}$ gives, by the data-processing inequality, $I[Y^{\text{eval}}; Y^{\text{acq}} \mid \cdots] \le I[\Omega; Y^{\text{acq}} \mid x^{\text{acq}}]$: EPIG is the part of the BALD score that is useful for prediction.

**The proxy form.** Your notes state $\arg\max_x \text{EPIG} = \arg\min_x I[\Omega; Y^{\text{eval}} \mid X^{\text{eval}}, Y^{\text{acq}}, x^{\text{acq}}]$. Decode with the recipe: capital $X^{\text{eval}}$ and $Y^{\text{acq}}$ mean "in expectation over eval inputs and over the hypothetical acquired label"; the quantity is the parameter–prediction MI *remaining after* acquiring $x^{\text{acq}}$'s label. Maximizing information delivered to predictions = minimizing information about predictions still locked in the parameters. It follows from the MI chain rule, $I[\Omega, Y^{\text{acq}}; Y^{\text{eval}} \mid \cdot]$ expanded in both orders, noting $I[Y^{\text{acq}}; Y^{\text{eval}} \mid \Omega, \cdot] = 0$ (conditional independence given $\omega$).

---

## 7. Cheat sheet

| Quantity | Formula | Reads as |
|---|---|---|
| $H(p(x))$ | $-\log p(x)$ | surprise of one outcome |
| $H[X]$ | $\mathbb{E}_{p(x)}[-\log p(x)]$ | expected surprise |
| $H(p \,\|\, q)$ | $\mathbb{E}_{p}[-\log q]$ | surprise under wrong model |
| $D_{\mathrm{KL}}(p \,\|\, q)$ | $H(p\|q) - H(p\|p) \ge 0$ | cost of wrong model |
| $H[X \mid Y]$ | $\mathbb{E}_{p(x,y)}[-\log p(x \mid y)]$ | expected remaining uncertainty (averages over $y$!) |
| $I[X;Y]$ | $H[X] - H[X \mid Y]$, symmetric, $\ge 0$ | expected uncertainty reduction |
| $I[X;Y \mid Z]$ | $H[X \mid Z] - H[X \mid Y,Z]$ | dependence left after knowing $Z$ |
| eq. (7) rule | lowercase = fixed, condition the expectation on it; capital = average over it | — |
| EIG/BALD | $I[\Omega; Y^{\text{acq}} \mid x^{\text{acq}}]$ | expected posterior update from this label |
| EPIG | $\mathbb{E}_{\hat p(x^{\text{eval}})} I[Y^{\text{eval}}; Y^{\text{acq}} \mid x^{\text{eval}}, x^{\text{acq}}]$ | expected change in test predictions from this label |
