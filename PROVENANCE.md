# Provenance Manifest

**Repository commit for all reported results and the hardened audit:**
`4787e06f4c28` on branch `experiments/bader-review-round2`
(https://github.com/khushpatel1102/maj-research)

## Predictions (per-sample, June 2026 runs)
- Primary series (seed 42): `results/leakage_free_{stateless,maj,mcts_judge,mcts_judge_memory}.csv`
- Provenance conditions: `results/lf_oracle_*.csv`, `results/lf_poisoned_{10,20,50}_*.csv`, `results/lf_bare_maj.csv`
- Multi-seed: `results/lf_*_seed{7,123}.csv`
- Revision experiments: `results/exp1_*` (2x2), `results/exp2_*` (memory-use controls),
  `results/exp3_cv_*` (grouped CV), `results/exp4_*` (leniency instrumentation)
- Reliability harness: `results/harness_*.csv`

Every CSV has per-item fields (`idx, topic, question, expected, predicted, correct, latency_s`,
plus per-experiment logging columns), so all statistics re-derive without model calls.

## Splits
Deterministic from code: `split_by_question(df, train_ratio=0.5, seed=42)` in
`experiments/exp_common.py` (identical logic in `benchmark_leakage_free.py`); seeds 42/123/7.
The row-level (leaky) split used only by exp1 cell B/D: `split_by_row`, same file.
CV folds: `grouped_folds(questions, 5, seed=42)` in `experiments/exp3_grouped_cv.py`.

## Prompts
- Judge prompts (stateless + memory-conditioned): `src/prompts.py`
- Memory-context assembly: `src/judge.py::_format_memory_context`
  (threshold-parametrized copy: `experiments/exp4_leniency_anchor.py::format_context`)
- Control-arm injected blocks: `experiments/exp2_memory_controls.py`

## Audit records (v1 topology-scope fingerprint)
- `results/*_audit.json` (one per audited run: before/after snapshots + diff)
- v1 hashes node identifiers + edge triples. The hardened v2 fingerprint
  (full stored state: labels + all properties + edge properties) ships at this
  commit in `src/graph_manager.py::snapshot` with `fingerprint_scope` recorded.

## Retrieval logs (per item)
- `results/exp4_{asymmetric,symmetric}_seed42.csv`: pre/post-threshold exemplar
  counts, per-exemplar similarities, top-1 retrieved label + similarity,
  memory token counts, stateless verdict, flip flag + direction.
- `results/exp1_{A,B}_*.csv`: twin-in-memory / twin-retrieved / twin-similarity dose logs.

## Mutation tests (audit hardening evidence)
- `tests/test_audit_mutation.py` (7 tests, all passing at this commit):
  property-flip and topology-edit detection via raw Cypher; wrapper- and
  driver-level write prevention under freeze; violation fatality; clean restore.

## Supplementary analyses
- `experiments/round3_analyses.py` -> `results/round3_analyses.json`:
  response-length baseline, clustered TOST for negative controls,
  missing-output sensitivity bounds.
