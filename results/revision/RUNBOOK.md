# Bader's revision experiments — runbook

Five tasks from Bader (2026-06). Task 5 (theory) is **already done in the paper**.
Tasks 1–4 are **experiments that need Neo4j running + `OPENAI_API_KEY`** — run them
on a machine with the graph DB up (it was not running when the code was written).

## Prerequisites
```bash
neo4j start                       # port 7687 must be open
source venv/bin/activate
# OPENAI_API_KEY already in .env
```

## Run everything
```bash
python experiments_bader.py --task all --model gpt-4o --seed 42
python analyze_grouped.py         # stats for task 3 after cv runs
```
Or run tasks individually: `--task split2x2 | controls | cv | anchor`.
Outputs land in `results/bader/`.

## What each task proves and what to look for

**Task 1 — 2×2 leakage design** (`--task split2x2`)
Rows: {question-split, row-split} × {frozen, write-back}. The hypothesis:
the conventional +6–10pt gain should appear ONLY in (row-split, write-back),
and vanish in (question-split, frozen). This turns the before/after anecdote
into a controlled 2×2 — the single most important experiment for the paper's
central claim. Files: `2x2_<split>_<wb>_seed42.csv` + audit JSONs (frozen cells only).

**Task 2 — memory-use controls** (`--task controls`)
Positive control `cheating` (paired oracle twin injected in-context) must score
HIGH — this proves the judge *can* use memory when it is informative, killing
the "judge just ignores the context block" objection. Negative controls
`random` / `shuffled` / `irrelevant` should not beat stateless. Memory graph is
untouched (in-context injection only), so the frozen audit still passes.

**Task 3 — grouped 5-fold CV** (`--task cv`, then `analyze_grouped.py`)
Primary modes only: stateless, maj_asymmetric, maj_oracle, maj_balanced. NO MCTS.
`analyze_grouped.py` reports question-cluster bootstrap CIs + paired deltas, and
a `correct ~ mode + (1|question)` mixed-effects model if statsmodels is installed
(`pip install statsmodels`; otherwise the cluster bootstrap is the primary result).
This replaces the single-split n=80 numbers with CV estimates the reviewer asked for.

**Task 4 — leniency-anchor test** (`--task anchor`)
Runs maj_asymmetric vs maj_balanced with full per-item retrieval logs
(`log_n_pos`, `log_n_neg`, `log_sim_*`, `log_top1_label`, `log_memory_tokens`,
`flip_vs_stateless`). KEY RESULT: if `balanced` removes the toward-pass tilt
(printed as `toward_pass=`), remedy (a) is confirmed causally, not just argued.

## Mapping results back into the paper
- Task 1 → new subsection in §5 (or a 2×2 table) replacing the "motivating
  observation" hand-wave with a controlled demonstration of the leakage effect.
- Task 2 → new "memory-use controls" paragraph in §5; cheating-oracle row is the
  positive control proving the pipeline is functional.
- Task 3 → replace/augment the single-seed Table 1 and the cross-seed paragraph
  with CV + cluster-bootstrap CIs.
- Task 4 → strengthens §6.1 (the leniency-anchor claim) and §6.3 remedy (a):
  the balanced run is no longer hypothetical.
- Task 5 → DONE: §6.2 now states the Bayes-optimal ceiling (Prop. 1) and demotes
  self-disagreement to a diagnostic, per Bader's correction.

## Notes / caveats
- `judge_instrumented.py` is additive; it does not modify `judge.py`, so existing
  audited results remain reproducible.
- The cheating twin is drawn from the FULL dataset (the genuine opposite-label
  row for each test question) — this is intentionally a leakage condition used as
  a positive control, NOT part of the clean protocol.
- Costs: task 1 = 4 builds + 4 evals; task 3 = 5 folds × 4 modes. Budget API
  spend accordingly; none of these run MCTS (per Bader's instruction).
