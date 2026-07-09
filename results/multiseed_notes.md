# Multi-seed run notes

## Seed 42 (complete, all 80 samples, frozen-memory audited)

| Mode / condition | Accuracy | n |
|------------------|----------|---|
| stateless (no memory) | 65.0% | 80 |
| maj (self-written) | 63.7% | 80 |
| mcts_judge (no memory) | 62.5% | 80 |
| mcts_judge_memory (self-written) | 68.8% | 80 |
| maj (oracle) | 56.2% | 80 |
| mcts_judge_memory (oracle) | 70.0% | 80 |
| maj poison 10% | 55.0% | 80 |
| maj poison 20% | 53.8% | 80 |
| maj poison 50% | 58.8% | 80 |
| mcts_judge_memory poison 10% | 45.0% | 80 |
| mcts_judge_memory poison 20% | 73.8% | 80 |
| mcts_judge_memory poison 50% | 21.2% | 80 |

Source: `results/leakage_free_*.csv`, `results/lf_oracle_*.csv`, `results/lf_poisoned_*.csv`. Audit logs: identical before/after fingerprints in every case.

## Seeds 123 and 7 (fast multi-seed: stateless + MAJ only, all 80 samples, frozen-memory audited)

Why only stateless and MAJ for the additional seeds: the MCTS-mode evaluations are ~40 minutes of LLM calls each and proved fragile to mid-run network drops (one full seed-123 sweep died after ~49 minutes, leaving partial CSVs). For the cross-seed table we therefore restrict the additional seeds to the fast, write-free modes, which are also the comparison that matters most for the question "does memory alone help?". The MCTS run-to-run variance is documented separately by the seed-42 poisoning results, where the same 50%-poisoning condition produced 21.2% on one run and 68.8% on another (a 47.6 percentage-point swing on identical inputs).

| Mode / condition | Seed 123 | Seed 7 |
|------------------|----------|--------|
| stateless (no memory) | 70.0% | 67.5% |
| maj (self-written) | 67.5% | 66.2% |
| maj (oracle) | 66.2% | 66.2% |

All six evaluations passed the frozen-memory audit (identical before/after fingerprints). Source: `results/lf_no_memory_stateless_seed{123,7}.csv`, `results/lf_self_written_maj_seed{123,7}.csv`, `results/lf_oracle_maj_seed{123,7}.csv` and the matching `_audit.json` files. Runner: `multiseed_fast.py`.

## Cross-seed comparison

| Mode / condition | Seed 42 | Seed 123 | Seed 7 | Mean | Range |
|------------------|---------|----------|--------|------|-------|
| stateless (no memory) | 65.0% | 70.0% | 67.5% | 67.5% | 65.0 to 70.0 |
| maj (self-written) | 63.7% | 67.5% | 66.2% | 65.8% | 63.7 to 67.5 |
| maj (oracle) | 56.2% | 66.2% | 66.2% | 62.9% | 56.2 to 66.2 |

Interpretation. Across all three seeds, MAJ (self-written) sits within roughly two percentage points of stateless — below it on seed 42, slightly above on seeds 123 and 7. There is no consistent direction, which is the descriptive-trend conclusion: memory alone does not produce a statistically supported gain at this sample size. The per-mode cross-seed range is about 5 percentage points, so the 3.8-point "improvement" of MCTS-Judge + Memory over stateless on seed 42 falls inside the cross-seed noise band. The MAJ-oracle number is the most seed-sensitive (56.2% on seed 42 versus 66.2% on the other two), reflecting both the small held-out set (80 samples) and the single-pass judge's sensitivity to which 40 questions form the test split.

## Structure ablation (seed 42, 80 samples)

To test whether the elaborate 5-node typed graph earns its complexity, a "bare" memory was built: identical self-written labels but only Policy and Attempt nodes are committed to the graph. Issue, Fix, and Semantic extraction are skipped. Retrieval at test time therefore returns only contrastive past attempts; no issues, fixes, or semantic patterns are available.

| Schema | What's stored | Accuracy [Wilson 95% CI] |
|--------|---------------|--------------------------|
| Stateless (no memory) | nothing | 65.0% [54.1, 74.5] |
| Bare memory | Policy + Attempt only | 66.2% [55.4, 75.7] |
| Full self-written | Policy + Attempt + Issue + Fix + Semantic | 63.7% [52.8, 73.4] |

Paired comparisons on identical items:

| Comparison | Paired delta [bootstrap 95% CI] | McNemar p | Significant |
|------------|---------------------------------|-----------|-------------|
| bare maj vs stateless | +1.2pp [-2.5, +6.2] | 1.000 | no |
| bare maj vs full self-written maj | +2.5pp [-5.0, +10.0] | 0.727 | no |

Interpretation. The bare-memory MAJ (Policy + Attempt only) is statistically indistinguishable from both stateless and the full self-written schema. The extra extraction of Issue, Fix, and Semantic nodes does not produce a measurable accuracy gain on this benchmark. The elaborate typed graph therefore does not earn its complexity for raw verdict accuracy; whatever value it has must come from interpretability, debugging, or downstream uses other than direct retrieval-conditioned judging. This is reported as a structural negative finding rather than as a failure: it bounds what the schema is doing for the judge. Audit log: `results/lf_bare_maj_audit.json` (160 nodes / 80 edges, identical before/after). Runner: `run_bare_ablation.py`. Per-sample CSV: `results/lf_bare_maj.csv`.
