# Revision Experiments — Report

**Paper:** *When Does Memory Help an LLM Judge?*
**Author:** Khush Patel · **Supervisor:** Prof. Bader Rasheed · Innopolis University
**Run:** GPT‑4o · seed 42 · all memory arms audited (frozen‑memory SHA‑256 check passed)
**Status:** Four experiments complete. Per‑sample CSVs + audit records saved in `results/`.
*(The theory fix — Task 5 — is handled directly in the paper PDF and is not covered here.)*

---

## TL;DR

| # | Experiment | Result | Effect on paper |
|---|------------|--------|-----------------|
| 1 | 2×2 leakage design | Neither channel reproduces the +6–10pp gain | ⚠️ **Reframe a claim** |
| 2 | Memory‑use controls | Informative memory **+12.5pp**; controls at baseline | ✅ **Strong new positive result** |
| 3 | Grouped 5‑fold CV | Null holds at n=160; oracle CI [−7.5, 0.0] | ✅ Strengthens null |
| 4 | Leniency‑anchor test | Mechanism real, effect ≈ 1 item | ⚠️ **Soften a claim** |

**One‑line summary:** the methodology and the central null are now *stronger* (a clean positive control + a larger‑sample CV), and two mechanistic side‑claims need honest softening. The headline question — *does memory help?* — now has a precise answer: **the judge can use memory when it is informative; this benchmark's memory is not informative.**

---

## Experiment 2 — Does memory help when it is informative?  ✅ YES (the key result)

*Bader's ask: prove the judge can actually use memory — a cheating‑oracle condition plus random / label‑shuffled / same‑length‑irrelevant negative controls.*

All five arms run on the same 80 held‑out items; each control routes through the **identical** memory prompt template, so the only thing that varies is the injected block.

| Arm | Accuracy | Fail‑recall | Verdict flips vs stateless |
|-----|----------|-------------|----------------------------|
| stateless (no memory) | 66.2% | 35.0% | — |
| **cheating‑oracle** (item's own correct verdict injected) | **78.8%** | **60.0%** | **15.0%** |
| random context | 65.0% | 32.5% | 3.8% |
| label‑shuffled memory | 63.7% | 30.0% | 2.5% |
| same‑length irrelevant | 68.8% | 40.0% | 5.0% |

**Why this settles the question.** When memory carries the answer, accuracy jumps **+12.5pp**, and the gain is almost entirely on failing items (fail‑recall 35% → 60%) — exactly the class the judge was weakest on. Per item, informative memory flipped **11 verdicts wrong→right and only 1 right→wrong**. Meanwhile all three uninformative controls sit within ±3pp of the no‑memory baseline.

**Conclusion.** The judge *is capable* of using memory; the null elsewhere is therefore a property of the **benchmark's memory carrying no usable signal**, not of a broken or memory‑blind judge. This brackets the null from both sides, as requested.

---

## Experiment 1 — 2×2 leakage decomposition  ⚠️ challenges our wording

*Bader's ask: row split vs question split × write‑back vs frozen, to attribute the conventional +6–10pp gain to specific leakage channels.*

| Cell | Split × write‑back | Accuracy | Balanced acc | Δ balanced vs clean (A) |
|------|--------------------|----------|--------------|-------------------------|
| **A** | question × frozen (clean) | 66.2% | 66.2% | — |
| B | row × frozen (paired‑item channel) | 60.0% | 63.4% | **−2.9** |
| C | question × write‑back (write‑back channel) | 67.5% | 67.5% | **+1.2** |
| D | row × write‑back (both) | 63.7% | 66.8% | +0.6 |

Frozen cells passed the audit (memory unchanged); write‑back cells grew memory as intended.

**Finding.** **Neither isolated leakage channel reproduces the conventional +6–10pp gain.** Paired‑item leakage actually *lowers* accuracy (the −6.2pp raw drop in B is mostly the 44/36 class‑imbalance the row split creates — balanced accuracy shows only −2.9). Write‑back alone is +1.2pp.

**Recommended reframe.** We should **not** keep stating "the apparent gain was these two leakage channels." More defensible wording: *the conventional gain does not survive any clean re‑measurement, and the two structurally isolatable channels do not by themselves manufacture it; it is best attributed to the original interleaved build‑and‑test protocol (a stronger, compound leak) and/or sampling noise.* This is honest and harder to attack.

---

## Experiment 3 — Grouped 5‑fold cross‑validation  ✅ strengthens the null

*Bader's ask: 5‑fold grouped (question‑level) CV for primary modes only (stateless, MAJ, oracle MAJ, balanced MAJ); question‑cluster bootstrap; no MCTS.*

Pooled over all 5 folds = **all 160 samples** (every item tested once). CIs from a question‑cluster bootstrap (10k resamples) that respects pass/fail pairing.

| Mode | Accuracy | Fail‑recall | Δ vs stateless [95% CI] |
|------|----------|-------------|--------------------------|
| stateless | 68.1% | 37.5% | — |
| MAJ (self / asymmetric) | 68.8% | 38.8% | **+0.6 [−3.8, +5.0]** |
| MAJ (oracle) | 64.4% | 30.0% | **−3.8 [−7.5, 0.0]** |
| MAJ (balanced / symmetric) | 68.1% | 37.5% | **0.0 [−4.4, +4.4]** |
| balanced − self | | | **−0.6 [−3.1, +1.3]** |

**Findings.** At double the paper's sample size the null holds: MAJ‑self vs stateless is +0.6pp with a CI through zero. Oracle (perfectly correct) memory does **not** help and trends slightly worse (CI touches 0 from below) — the cleanest single statement that this benchmark's memory carries no exploitable label signal. The leniency rebalancing (balanced − self) is a reliable zero.

---

## Experiment 4 — Leniency‑anchor causal test  ⚠️ needs softening

*Bader's ask: per‑item log of passing/failing exemplar counts, similarities, top‑1 label, memory tokens, verdict flips; then symmetric thresholds / equal‑k retrieval. If balancing removes the false‑pass tilt, the explanation is much stronger.*

The asymmetry in the published code is in the **thresholds** (positive ≥ 0.85, negative ≥ 0.92), so the meaningful manipulation is symmetric thresholds (both 0.85), holding retrieval fixed.

| Condition | Accuracy | Fail‑recall | Avg post‑threshold pos / neg exemplars |
|-----------|----------|-------------|----------------------------------------|
| stateless | 66.2% | 35.0% | — |
| MAJ asymmetric (published) | 65.0% | 32.5% | 3.00 / 2.01 |
| MAJ symmetric (balanced) | 66.2% | 35.0% | 3.00 / 2.71 |

**Finding.** The mechanism is **real**: symmetric thresholds admit more failing exemplars (neg 2.01 → 2.71). But the accuracy payoff is **+2.5pp fail‑recall = 1 item out of 40**, and balanced MAJ only *recovers* the stateless baseline. The CV (Exp 3) confirms balanced − self is a reliable zero.

**Recommended reframe.** Downgrade "leniency anchor, causally confirmed (one‑line fix)" → *"the asymmetric retrieval does tilt the contrastive set toward passing exemplars; rebalancing recovers the small MAJ deficit, but the effect (~1 item) is at the edge of detectability at this scale."* Still consistent with the theory (little signal either way); just not a strong causal headline.

---

## What this means for the paper

- **Stronger now:** a clean positive control (Exp 2) that proves capability, plus an n=160 cross‑validated null (Exp 3). The central claim is more defensible than before.
- **Two honest edits needed:** reframe the leakage‑channel attribution (Exp 1) and soften the leniency‑anchor claim (Exp 4).
- **Net:** this is the deeper, self‑critical analysis the review asked for — including reporting the two results that pushed back on our own wording.
- *(The theory fix is being applied directly in the paper PDF, separately from this report.)*

## Open question for you
The cheating‑oracle (Exp 2) used the **explicit‑oracle‑exemplar** variant. The review mentioned "the paired test twin **or** an explicit oracle exemplar" — one suffices, but if you'd like the **twin‑retrieval** variant as an extra robustness check, it's a quick additional run.

## Reproducibility
All scripts in `experiments/` (one‑command each, see `RUNBOOK.md`); per‑sample CSVs, JSON summaries, and frozen‑memory audit records in `results/exp*_seed42.*`.
