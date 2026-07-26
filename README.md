# MCTS-MAJ: Memory-Augmented LLM-as-a-Judge with Monte Carlo Tree Search

This repository accompanies the bachelor's thesis "MCTS-MAJ: Memory-Augmented LLM-as-a-Judge with Monte Carlo Tree Search" (Khush Patel, supervised by Professor Bader Rasheed, Innopolis University, 2026). It contains the full implementation, all benchmark scripts, and the per-sample result files for every experiment reported in the thesis and in the workshop paper "When Does Memory Help an LLM Judge? A Leakage-Free, Audited Evaluation of Memory-Augmented Judging" (submitted to ELMKE at ISWC 2026).

---

## Table of Contents

1. [Architecture](#architecture)
2. [Repository Layout](#repository-layout)
3. [Setup](#setup)
4. [Reproducing Thesis Results](#reproducing-thesis-results)
5. [Benchmark Scripts](#benchmark-scripts)
6. [Key Results](#key-results)

---

## Architecture

The framework has three components that operate on the same Neo4j memory graph.

**1. MAJ (Memory-Assisted Judge).** A typed property graph with five node kinds (Policy, Attempt, Issue, Fix, Semantic) and four relations (SATISFIES, CAUSES, RESOLVES, ABSTRACTS_TO). Retrieval is a three-stage pipeline: contrastive attempts, similar issues, semantic patterns. Similarity thresholds are tuned asymmetrically.

**2. MCTS-Judge (Reasoning Layer).** Decomposes evaluation into 5 to 7 dynamically generated, task-specific subtasks. Each rollout selects subtasks via UCT plus an LLM self-assessment, evaluates the response from each perspective, and aggregates a verdict by majority vote and global synthesis.

**3. MCTS-Retrieval (Retrieval Layer).** Replaces the fixed three-stage pipeline with agentic search over the memory graph. The agent invokes graph-level tools (search_by_attempts, search_by_issues, search_by_semantics, four multi-hop traversals) and trajectories are scored by relevance, diversity, and volume.

### Six Evaluation Modes

| Mode | Description |
|------|-------------|
| Stateless | Single-pass LLM judge, no memory, no MCTS |
| MAJ | Single-pass judge with three-stage memory retrieval |
| MCTS-Judge | Tree search reasoning, no memory |
| MCTS-Judge + Memory | Tree search reasoning with MAJ memory context |
| MCTS-Retrieval + Judge | Agentic graph search with single-pass judge |
| Full MCTS | MCTS-Retrieval feeds context into MCTS-Judge |

---

## Repository Layout

```
src/
  models.py            # Pydantic data classes for Policy, Attempt, Issue, Fix, Semantic
  graph_manager.py     # Neo4j connection, vector indexes, typed CRUD, graph tools
  judge.py             # Single-pass judge (stateless + MAJ-conditioned)
  prompts.py           # Prompt templates
  mcts_judge.py        # MCTS reasoning with dynamic subtask generation
  mcts_retrieval.py    # Agentic search over the memory graph
  mcts_pipeline.py     # Composition wrappers for the six evaluation modes

tests/
  test_audit_mutation.py   # Proves the frozen-memory audit detects/blocks mutations
  test_pipeline.py
  test_organic_memory.py
  test_memory_comparison.py
  test_memory_evolution.py
  test_edge_cases.py

data/
  benchmark_df.csv     # EvalsBench dataset (160 samples, 80 unique questions)
  annotation_df.csv    # Annotations

results/
  benchmark_summary.md            # Summary of all results
  leakage_free_*.csv              # Leakage-free Stage 2 per-sample results
  lf_oracle_*.csv                 # Oracle-memory experiments
  lf_poisoned_{10,20,50}_*.csv    # Poisoned-memory experiments
  harness_stability.csv           # Reliability harness: stochastic stability
  harness_label_flip.csv          # Reliability harness: label flip discriminative
  harness_defense_{off,on}.csv    # Reliability harness: defense mechanism
  results_*.csv                   # Earlier (pre-leakage-fix) Stage 2 results
  exp1_* exp2_* exp3_* exp4_*     # Revision experiments (2x2, controls, CV, thresholds)
  round3_analyses.json            # Length baseline, equivalence, sensitivity

reports/
  MAJ_Research_Progress_Report.pdf
  MCTS-MAJ_Progress_Report.pdf

thesis/
  chapter3_methodology.{tex,pdf}
  chapter4_implementation.{tex,pdf}
  additional_achievement.{tex,pdf}
  references.bib

experiments/
  exp_common.py             # Shared splits, memory builders, audited eval loop
  exp1_2x2_leakage.py       # 2x2 leakage decomposition (split x write-back)
  exp2_memory_controls.py   # Answer-injection positive control + negative controls
  exp3_grouped_cv.py        # Grouped five-fold cross-validation
  exp4_leniency_anchor.py   # Per-item retrieval logging + symmetric-threshold test
  round3_analyses.py        # Length baseline, clustered equivalence, sensitivity

paper/
  paper_ceur.tex            # Workshop paper source (CEUR single-column)
  references.bib
  figures/

PROVENANCE.md               # Commit-pinned map of predictions, splits, prompts,
                            # audit records, and retrieval logs

benchmark_mcts.py             # Stage 2 runner: six modes on a configurable sample size
benchmark_leakage_free.py     # Leakage-free runner: question-level split, frozen memory,
                              # four memory provenance conditions
experiments_harness.py        # Reliability harness: stochastic stability, label flip,
                              # defense mechanism (similarity-threshold filter)
benchmark_evalsbench.py       # Original Stage 1 MAJ runner
benchmark_evalsbench_v2.py    # Stage 1 MAJ runner (refined)

requirements.txt
README.md
```

---

## Setup

### Requirements

- Python 3.11
- Neo4j Community Edition (>= 5.x) running locally with vector indexes
- OpenAI API key (and optionally an Anthropic key for Claude experiments)

### Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Environment

Create a `.env` file in the repository root:

```
OPENAI_API_KEY=sk-...
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=...
```

### Start Neo4j

```bash
neo4j start
```

The `graph_manager.py` module creates the required vector indexes automatically on first connection.

---

## Reproducing Thesis Results

All Stage 2 (leakage-free) experiments use a fixed random seed of `42` and 80 held-out test samples. The split is performed at the question level: 40 questions go to memory construction, 40 go to test, and both pass and fail versions of any question stay on the same side of the split.

### Stage 2 results (Chapter 4, Tables 4.4 to 4.6)

The single command below reproduces every Stage 2 experiment under the leakage-free protocol:

```bash
python benchmark_leakage_free.py --model gpt-4o --seed 42
```

This runs the four memory conditions (no memory, self-written, oracle, poisoned at three rates) across the four evaluation modes (stateless, MAJ, MCTS-Judge, MCTS-Judge + Memory) and writes per-sample CSVs to `results/`.

To run a single condition only:

```bash
python benchmark_leakage_free.py --model gpt-4o --condition oracle
python benchmark_leakage_free.py --model gpt-4o --condition poisoned --poison_rate 0.20
```

### Mapping thesis tables to result files

| Thesis location | Result file(s) |
|-----------------|---------------|
| Chapter 4, Table 4.4 (clean memory) | `results/leakage_free_stateless.csv`, `leakage_free_maj.csv`, `leakage_free_mcts_judge.csv`, `leakage_free_mcts_judge_memory.csv` |
| Chapter 4, Table 4.5 (oracle vs self-written) | `results/lf_oracle_maj.csv`, `lf_oracle_mcts_judge_memory.csv` |
| Chapter 4, Table 4.6 (poisoning) | `results/lf_poisoned_{10,20,50}_maj.csv`, `lf_poisoned_{10,20,50}_mcts_judge_memory.csv` |

Each CSV contains per-sample fields (`idx`, `topic`, `expected`, `predicted`, `correct`, `latency_s`) so that any reported accuracy can be re-derived without rerunning the LLM calls.

---

## Benchmark Scripts

### benchmark_leakage_free.py (Stage 2, primary)

Implements the leakage-free protocol:

- Question-level train/test split (40/40, seeded).
- Memory frozen during evaluation (no writes).
- Four memory provenance conditions: `no_memory`, `self_written`, `oracle`, `poisoned`.

Common invocations:

```bash
# Full Stage 2 reproduction
python benchmark_leakage_free.py --model gpt-4o --seed 42

# Just clean self-written memory
python benchmark_leakage_free.py --model gpt-4o --condition self_written

# Poisoned memory at 50 percent
python benchmark_leakage_free.py --model gpt-4o --condition poisoned --poison_rate 0.50
```

### benchmark_mcts.py (Stage 2, six-mode ablation)

Runs any subset of the six evaluation modes:

```bash
# Single mode
python benchmark_mcts.py --mode mcts_judge_memory --samples 30 --model gpt-4o

# All six modes
python benchmark_mcts.py --mode all --samples 30 --model gpt-4o
```

### experiments_harness.py (Reliability Harness)

Implements three reliability stress tests on top of the leakage-free protocol:

- **Stochastic stability**: same input run twice; reliable judges should agree with themselves.
- **Label flip**: each test response is rewritten to flip its ground-truth label; reliable judges should flip their verdict.
- **Defense mechanism**: similarity-threshold filter on memory retrieval, evaluated on 50% poisoned memory.

```bash
# Run all three experiments
python experiments_harness.py --experiment all --model gpt-4o

# Run a single experiment (skip rebuilding memory)
python experiments_harness.py --experiment stability --skip_build --n_subset 20
python experiments_harness.py --experiment flip --skip_build --n_subset 20
python experiments_harness.py --experiment defense
```

### Revision experiments (workshop paper)

```bash
python experiments/exp1_2x2_leakage.py     --model gpt-4o --seed 42
python experiments/exp2_memory_controls.py --model gpt-4o --seed 42
python experiments/exp3_grouped_cv.py      --model gpt-4o --seed 42
python experiments/exp4_leniency_anchor.py --model gpt-4o --seed 42
python experiments/round3_analyses.py   # no API calls; reads results/ only
```

Each writes per-sample CSVs (and audit records where applicable) to `results/`.

### benchmark_evalsbench.py / benchmark_evalsbench_v2.py (Stage 1, MAJ-only)

Runs the original MAJ-only experiments at 100 and 1000 samples used in the Stage 1 progress report. These predate the leakage-free protocol and should not be compared directly to Stage 2 numbers.

---

## Key Results

All headline numbers are from the leakage-free, audited protocol (GPT-4o, seed 42,
question-level split, frozen memory). Full details and all caveats are in the paper;
per-sample records and audit files are in `results/`.

### Clean-memory comparison (held-out n=80, question-clustered 95% CIs)

| Mode | Accuracy [95% CI] | Coverage |
|------|-------------------|----------|
| Stateless | 70.0 [62.5, 77.5] | 80/80 |
| MAJ (self-written memory) | 65.0 [57.5, 72.5] | 80/80 |
| MCTS-Judge (no memory) | 70.0 [60.0, 80.0] | 80/80 |
| MCTS-Judge + Memory | 71.8 [63.7, 79.7] | 78/80 |

No configuration significantly outperforms the stateless judge (paired McNemar +
bootstrap; the design is powered for the conventional 6-10 point claim, not for
few-point effects). An earlier claim that the combined mode beats stateless is
withdrawn; it predated the audited protocol.

### Context for absolute numbers

A response-length threshold alone scores **75.0%** on the held-out half (longer
responses pass), above every configuration tested. Absolute accuracies on this
benchmark partly reflect superficial cues; the paper's claims rest on paired
mode-versus-mode comparisons only.

### Memory-use controls (answer-injection positive control)

| Injected memory | Accuracy |
|-----------------|----------|
| Stateless (none) | 66.2 |
| **Answer injection (item's own verdict)** | **78.8** |
| Random context | 65.0 |
| Label-shuffled | 63.7 |
| Irrelevant (length-matched) | 68.8 |

The judge can use memory when it contains the answer; the benchmark's real
memory carries no exploitable signal for this pipeline.

### Poisoning

Under the corrected, audited harness, accuracy is flat across 10/20/50% label
poisoning (MAJ 62-64%, MCTS+Memory 65-70%). An earlier "collapse to 21.2% at
50% poisoning" was a harness artifact (network errors scored as wrong answers)
and is withdrawn.

### Grouped 5-fold cross-validation (all 160 items)

MAJ-self vs stateless +0.6pp [-3.8, +5.0]; oracle -3.8pp [-7.5, 0.0];
balanced-minus-self -0.6pp [-3.1, +1.3] (question-cluster bootstrap). The null
holds at double the sample size.

## Audit status

The frozen-memory audit is two-layer: a SHA-256 fingerprint over the full stored
graph state (v2: node labels + all properties + edge properties) and runtime
write-blocking at both the wrapper-method and raw-Cypher level. Violations abort
the run. `tests/test_audit_mutation.py` (7 tests) proves detection and prevention
against a live Neo4j. The June runs were audited with the earlier v1 topology
fingerprint; see the paper's Limitations for why v1 sufficed for the two modeled
channels. Do not describe the protocol as "provably leakage-free"; the audit
verifies state invariance and blocks in-process writes, no more.

## Provenance

Everything reported is reproducible from repository commit `4787e06f4c28`
(see `PROVENANCE.md`): per-sample predictions, splits (by seed), prompts,
audit records, retrieval logs, and analyses. Dataset: `data/benchmark_df.csv`,
SHA-256 `84886f7340df190b90d60777f22c7eedf1d3d29b9d18232e59d055d3e40d0c76`,
obtained 2026-02, Apache-2.0. Neo4j: server 5.x Community (June runs; 5.26.28
verification environment), Python driver `neo4j` 5.15.0.
