# Revision Experiments — Runbook (Bader review, round 2)

Five tasks from Bader's review. Task 5 (theory) is already done in `paper/paper.tex`.
Tasks 1–4 are runnable scripts in `experiments/`, validated end-to-end against the
live Neo4j + GPT-4o stack on `gpt-4o-mini` smoke runs. Run them on `gpt-4o` for the
numbers that go in the paper.

## Prereqs
- Neo4j running on `bolt://127.0.0.1:7687` (it is).
- Valid `OPENAI_API_KEY` in `.env` (it is, as of this session).
- `cd /Users/khushpatel2002/detailed_research` and use `venv/bin/python`.

## One-liners (full gpt-4o runs)

```bash
# Task 1 — 2x2 leakage decomposition (row vs question split x write-back vs frozen)
venv/bin/python experiments/exp1_2x2_leakage.py --model gpt-4o --seed 42

# Task 2 — memory-use controls (cheating-oracle upper bound + 3 negative controls)
venv/bin/python experiments/exp2_memory_controls.py --model gpt-4o --seed 42

# Task 3 — 5-fold grouped (question-level) CV, primary modes only, cluster bootstrap
venv/bin/python experiments/exp3_grouped_cv.py --model gpt-4o --seed 42

# Task 4 — leniency-anchor instrumentation + symmetric-threshold causal test
venv/bin/python experiments/exp4_leniency_anchor.py --model gpt-4o --seed 42
```

For robustness, repeat Tasks 1/3/4 on seeds 7 and 123 (matches the existing
multi-seed artifacts in `results/`).

## Rough cost / time (gpt-4o)
- exp1: builds self memory twice (per split) + 4x80 eval ≈ 500–700 calls. ~25–40 min.
- exp2: no memory build; stateless + 4 controls x 80 ≈ 400 calls. ~15–25 min.
- exp3: 5 folds x (stateless + self-build 128 + asym 32 + sym 32 + oracle-build 0 + oracle 32). The 5 self-builds dominate ≈ 5x128 = 640 build calls + ~640 eval. ~60–90 min. **Most expensive — run overnight or reduce folds.**
- exp4: 1 self-build (80) + stateless 80 + asym 80 + sym 80 ≈ 320 calls. ~15–25 min.

## What each script outputs (in `results/`)
- `exp1_*_seed42.csv` + `exp1_2x2_summary_seed42.json` — per-cell accuracy, balanced
  accuracy, pass/fail recall, memory-growth audit flag, and per-row leakage **dose**
  (twin_in_memory / twin_retrieved / twin_similarity).
- `exp2_*_seed42.csv` + `exp2_controls_summary_seed42.json` — per-arm accuracy +
  flip-rate manipulation check. Expect: cheating_oracle ≫ stateless ≈ negative controls.
- `exp3_cv_*_seed42.csv` + `exp3_cv_summary_seed42.json` — pooled-over-folds class
  metrics + paired question-cluster bootstrap deltas (incl. balanced−self).
- `exp4_*_seed42.csv` + `exp4_summary_seed42.json` — per-item retrieval log
  (pre/post-threshold pos/neg counts, similarities, top1 label, mem tokens, flip dir)
  + asymmetric-vs-symmetric fail-class recall delta.

## Smoke (cheap, proves correctness, ~3–5 min each on gpt-4o-mini)
```bash
venv/bin/python experiments/exp1_2x2_leakage.py   --model gpt-4o-mini --seed 42 --limit 4
venv/bin/python experiments/exp2_memory_controls.py --model gpt-4o-mini --seed 42 --limit 3
venv/bin/python experiments/exp3_grouped_cv.py    --model gpt-4o-mini --seed 42 --folds 2 --limit-q 6
venv/bin/python experiments/exp4_leniency_anchor.py --model gpt-4o-mini --seed 42 --limit 3
```

## Design decisions forced by the adversarial design review
- **Balanced accuracy + per-class recall everywhere**: row split is 44/36, not 40/40,
  so raw accuracy across split types is confounded.
- **Symmetric thresholds (0.85/0.85) is the real balanced condition**, NOT equal-k:
  the published code already caps negatives to len(pos)+1, so the only live asymmetry
  is the thresholds (pos≥0.85 vs neg≥0.92).
- **Controls route through the published prompt template** so each differs from MAJ
  only by the injected block (else they'd measure "empty-context judge", not stateless).
- **temperature=0** on all custom-predictor calls so verdict flips measure memory,
  not sampling noise. (The published judge.py path is left untouched for fidelity.)
- **Self memory built once per fold** in exp3 and reused for asym+sym, so the
  asymmetric-vs-balanced gap is a pure retrieval-policy effect, not LLM-build noise.
- **Frozen audit asserted identical** on every memory arm; write-back arm in exp1 is
  the only one that (deliberately) grows memory.
- **Do NOT hard-code the 32.5/42.5 fail-class numbers as a target.** exp4 re-establishes
  the class-conditional baseline; "no degradation to remove" is a valid outcome.
```
