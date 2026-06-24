# Revision Experiments — Results Memo (Bader review, round 2)

All runs: **GPT-4o, seed 42, n=80 held-out**, leakage-free question split unless
stated, every memory arm audited (SHA-256 frozen-memory check passed). Raw CSVs +
JSON summaries in `results/exp{1,2,3,4}_*_seed42.*`; logs in `experiments/logs/`.

Status: ALL FOUR COMPLETE (exp1, exp2, exp3, exp4), full GPT-4o, saved to results/.

---

## exp2 — Memory-use controls  [STRENGTHENS the paper]

Proves the judge *can* use memory when it is informative, and that real retrieved
memory is not. This is the bracketing Bader asked for.

| Arm | Acc | Balanced | Fail-recall | Flip% vs stateless |
|---|---|---|---|---|
| **cheating_oracle** (inject item's own true verdict) | **78.8%** | 78.8% | 60.0% | 15.0% |
| stateless baseline | 66.2% | 66.2% | 35.0% | — |
| irrelevant_samelen (token-matched filler) | 68.8% | 68.8% | 40.0% | 5.0% |
| random_context | 65.0% | 65.0% | 32.5% | 3.8% |
| label_shuffled | 63.7% | 63.8% | 30.0% | 2.5% |

**Read:** informative memory lifts accuracy +12.5pp, entirely via fail-recall
(35→60). All three negative controls sit within ±3pp of stateless. The judge has
the *capability* to use memory; this benchmark's memory simply lacks the *signal*.
Directly supports the Bayes-ceiling theory (memory helps only when P(Y|X,M)≠P(Y|X)).

---

## exp1 — 2×2 leakage decomposition  [COMPLICATES → must reframe]

Crosses split (question vs row) × memory (frozen vs write-back). Frozen cells
audited identical; write-back cells grew memory as intended.

| Cell | split × write-back | Acc | Balanced | Δ balanced vs A |
|---|---|---|---|---|
| **A** question × frozen (clean) | 66.2% | 66.2% | — |
| B | row × frozen (paired-item channel) | 60.0% | 63.4% | **−2.9** |
| C | question × write-back (write-back channel) | 67.5% | 67.5% | **+1.2** |
| D | row × write-back (both) | 63.7% | 66.8% | +0.6 |

**Read:** *Neither* modeled leakage channel reproduces the conventional +6–10pp
gain at this scale. The raw −6.2pp in B is mostly the 44/36 class-imbalance the
row split induces (balanced acc shows only −2.9). Write-back alone is +1.2pp.

**Implication for the paper:** the current claim "the apparent gain was leakage
(these two channels)" is NOT supported as stated. Honest reframe: the conventional
gain does not survive any clean re-measurement; the two *structurally isolatable*
channels do not by themselves manufacture it; the original +6–10pp is therefore
best attributed to the original interleaved build-and-test protocol (a stronger,
compound leak than either isolated channel) and/or sampling noise — not to a single
clean channel we can point at. This is more defensible than the current wording.

---

## exp4 — Leniency-anchor causal test  [SOFTENS one claim]

Holds retrieval fixed; varies only the contrastive thresholds. Asymmetric =
published (pos≥0.85, neg≥0.92); symmetric = both 0.85. Memory built once, reused.

| Condition | Acc | Balanced | Fail-recall | avg post-thr pos / neg |
|---|---|---|---|---|
| stateless | 66.2% | 66.2% | 35.0% | — |
| asymmetric MAJ (published) | 65.0% | 65.0% | 32.5% | 3.00 / 2.01 |
| symmetric MAJ (balanced) | 66.2% | 66.2% | 35.0% | 3.00 / 2.71 |

**Read:** the mechanism is real — symmetric thresholds admit more failing
exemplars (neg 2.01→2.71). But the accuracy payoff is **+2.5pp fail-recall = 1
item / 40**, and symmetric MAJ only *recovers* the stateless baseline. At n=80 this
is within noise.

**Implication for the paper:** downgrade "leniency anchor, causally confirmed (one-
line fix)" → "the asymmetric retrieval does tilt the contrastive set toward passing
exemplars, and rebalancing it recovers the small MAJ deficit; the effect size
(~1 item) is at the edge of detectability at this scale." Still consistent with the
theory (little signal either way); just not a strong causal headline.

---

## exp3 — Grouped 5-fold CV  [reliability — STRENGTHENS]

Primary modes only, question-level folds, self-memory built once per fold and
reused for asym+sym, question-cluster bootstrap (10k) for paired CIs. Pooled over
all 5 folds = **all 160 samples** (every item tested once), the largest-evidence
view of the central comparison.

| Mode | Acc | Balanced | Fail-recall | Δ vs stateless [95% CI] |
|---|---|---|---|---|
| stateless | 68.1% | 68.1% | 37.5% | — |
| MAJ self (asymmetric) | 68.8% | 68.8% | 38.8% | +0.6 [−3.8, +5.0] |
| MAJ oracle | 64.4% | 64.4% | 30.0% | −3.8 [−7.5, 0.0] |
| MAJ balanced (symmetric) | 68.1% | 68.1% | 37.5% | 0.0 [−4.4, +4.4] |
| balanced − self | | | | −0.6 [−3.1, +1.3] |

**Read:** across all 160 items with paired question-cluster CIs:
- MAJ-self vs stateless: **+0.6pp, CI crosses 0** → no benefit (confirms the null
  on the largest sample, n=160 vs the paper's n=80).
- MAJ-oracle vs stateless: **−3.8pp, CI [−7.5, 0.0]** → oracle (perfectly correct)
  memory does not help and trends slightly *worse*; reinforces "no usable signal."
- MAJ-balanced vs stateless: **exactly 0.0** → symmetric thresholds neither help
  nor hurt overall.
- **balanced − self: −0.6pp, CI [−3.1, +1.3]** → the leniency rebalancing has NO
  reliable effect at fleet scale. This is the decisive number for exp4's claim:
  the leniency-anchor fix does not produce a measurable accuracy gain under CV.

**Implication:** strongly corroborates the central null at n=160 and independently
confirms the exp4 reframe — the balanced/symmetric retrieval change is not a
reliable improvement. The oracle CI touching 0 from below is the cleanest single
statement that this benchmark's memory carries no exploitable label signal.

---

## Net assessment for Bader

The new experiments **strengthen** the methodology and the central null (exp2 is a
clean, new positive control; exp3 adds cross-validated reliability), while
**honestly complicating two mechanistic side-claims** (exp1: leakage-channel
attribution; exp4: leniency-anchor magnitude). Two claims need softening; the core
result — memory does not help an already-capable judge on this benchmark, because
the memory is not conditionally informative — is *reinforced*, now from both the
null side (exp1/exp3/exp4) and the capability side (exp2).
