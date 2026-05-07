"""
Leakage-Free Benchmark for MCTS-MAJ

Addresses evaluation leakage concern: EvalsBench has paired pass/fail
versions of the same question. If memory is updated during evaluation,
the judge could learn patterns from one version and leak into the other.

Solution: Split by QUESTION (not by row).
- Train questions: build memory on these (both pass/fail versions)
- Test questions: evaluate on these (memory is frozen, no updates)

This ensures no question appears in both train and test sets.
"""

import sys
import time
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

sys.path.insert(0, 'src')

from models import Policy, Attempt, get_embedding
from judge import judge, judge_with_memory
from graph_manager import GraphManager
from mcts_judge import MCTSJudge, MCTSConfig
from mcts_pipeline import (
    run_stateless,
    run_maj,
    run_mcts_judge,
    run_mcts_judge_with_memory,
    run_mcts_retrieval_with_judge,
    run_full_mcts,
    store_mcts_result,
)
from mcts_retrieval import RetrievalConfig

# ============================================================
# CONFIG
# ============================================================
DATA_PATH = Path("data")
RESULTS_DIR = Path("results")

EVALSBENCH_GOAL = """Check if the response contains points mentioned from the grading notes and return 'pass' or 'fail'."""


def evalsbench_to_maj(row):
    return {
        'task': f"grading_notes: {row['grading_notes']}",
        'agent_output': row['response'],
        'expected': row['target'] == 'pass',
        'topic': row['topic']
    }


def split_by_question(df, train_ratio=0.5, seed=42):
    """
    Split dataset by unique questions so no question appears in both sets.
    Each question has a pass and fail version — both go to same split.
    """
    questions = df['question'].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(questions)

    split_idx = int(len(questions) * train_ratio)
    train_questions = set(questions[:split_idx])
    test_questions = set(questions[split_idx:])

    train_df = df[df['question'].isin(train_questions)]
    test_df = df[df['question'].isin(test_questions)]

    return train_df, test_df


def build_memory(train_df, gm, model):
    """Build memory from training set — store all evaluations."""
    print(f"\nBuilding memory from {len(train_df)} training samples...")

    for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Building memory"):
        sample = evalsbench_to_maj(row)
        try:
            result = judge_with_memory(
                task=sample['task'],
                agent_output=sample['agent_output'],
                graph_manager=gm,
                goal=EVALSBENCH_GOAL,
                model=model
            )

            # Store in memory
            gm.create_policy(result['policy'])
            gm.create_attempt(result['attempt'])
            for issue in result['issues']:
                gm.create_issue(issue)
            for fix in result['fixes']:
                gm.create_fix(fix)

            for rel in result['relationships']:
                if rel['type'] == 'SATISFIES':
                    gm.link_attempt_satisfies_policy(rel['from_id'], rel['to_id'])
                elif rel['type'] == 'CAUSES':
                    gm.link_attempt_causes_issue(rel['from_id'], rel['to_id'])
                elif rel['type'] == 'RESOLVES':
                    gm.link_fix_resolves_issue(rel['from_id'], rel['to_id'])

            for i, semantic in enumerate(result.get('semantics', [])):
                semantic_rels = result.get('semantic_relationships', [])
                if i < len(semantic_rels) and semantic_rels[i].get('is_new', True):
                    gm.get_or_create_semantic(semantic)
                if i < len(semantic_rels):
                    gm.link_issue_abstracts_to_semantic(
                        semantic_rels[i]['from_id'],
                        semantic_rels[i]['to_id']
                    )

        except Exception as e:
            print(f"  Error building memory for sample {idx}: {e}")

    # Count stored nodes
    counts = gm.driver.execute_query('MATCH (n) RETURN labels(n)[0] as label, count(n) as cnt')
    print("\nMemory graph contents:")
    for r in counts.records:
        print(f"  {r['label']}: {r['cnt']}")


def build_oracle_memory(train_df, gm):
    """Build memory using ground truth labels — perfect memory."""
    print(f"\nBuilding ORACLE memory from {len(train_df)} training samples...")

    for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Oracle memory"):
        sample = evalsbench_to_maj(row)
        try:
            policy = Policy(description=sample['task']).with_embedding()
            attempt = Attempt(
                agent_output=sample['agent_output'],
                is_successful=sample['expected'],  # Ground truth label
                reasoning=f"Oracle: ground truth label is {'pass' if sample['expected'] else 'fail'}"
            ).with_embedding()

            gm.create_policy(policy)
            gm.create_attempt(attempt)
            gm.link_attempt_satisfies_policy(attempt.id, policy.id)

        except Exception as e:
            print(f"  Error: {e}")

    counts = gm.driver.execute_query('MATCH (n) RETURN labels(n)[0] as label, count(n) as cnt')
    print("\nOracle memory contents:")
    for r in counts.records:
        print(f"  {r['label']}: {r['cnt']}")


def build_poisoned_memory(train_df, gm, poison_rate, seed=42):
    """Build memory with flipped labels to test robustness."""
    print(f"\nBuilding POISONED memory (flip rate={poison_rate:.0%}) from {len(train_df)} samples...")

    rng = np.random.RandomState(seed)
    flip_mask = rng.random(len(train_df)) < poison_rate
    flipped = 0

    for i, (idx, row) in enumerate(tqdm(train_df.iterrows(), total=len(train_df), desc=f"Poisoned ({poison_rate:.0%})")):
        sample = evalsbench_to_maj(row)
        try:
            label = sample['expected']
            if flip_mask[i]:
                label = not label
                flipped += 1

            policy = Policy(description=sample['task']).with_embedding()
            attempt = Attempt(
                agent_output=sample['agent_output'],
                is_successful=label,
                reasoning=f"Label: {'pass' if label else 'fail'}"
            ).with_embedding()

            gm.create_policy(policy)
            gm.create_attempt(attempt)
            gm.link_attempt_satisfies_policy(attempt.id, policy.id)

        except Exception as e:
            print(f"  Error: {e}")

    print(f"  Flipped {flipped}/{len(train_df)} labels ({flipped/len(train_df):.0%} actual)")

    counts = gm.driver.execute_query('MATCH (n) RETURN labels(n)[0] as label, count(n) as cnt')
    print("\nPoisoned memory contents:")
    for r in counts.records:
        print(f"  {r['label']}: {r['cnt']}")


def evaluate_test_set(test_df, mode, gm, model, audit_log_path=None):
    """
    Evaluate test set with frozen memory (no updates during eval).

    The frozen-memory protocol is enforced in two ways:
      1. A snapshot of the memory graph is taken before and after the run;
         a diff is computed and logged. Any non-zero delta fails the audit.
      2. All write methods on the graph are monkey-patched to raise
         FrozenMemoryViolation for the duration of the evaluation. This
         catches any leakage path immediately rather than after the fact.
    """
    snap_before = gm.snapshot()
    print(f"  [audit] before: {snap_before['total_nodes']} nodes, "
          f"{snap_before['total_edges']} edges, "
          f"fingerprint={snap_before['fingerprint'][:16]}...")

    results = []
    correct = 0
    total_time = 0

    with gm.freeze():
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"Eval [{mode}]"):
            sample = evalsbench_to_maj(row)
            start = time.time()

            try:
                if mode == "stateless":
                    result = run_stateless(
                        sample['task'], sample['agent_output'],
                        goal=EVALSBENCH_GOAL, model=model
                    )
                    predicted = result['attempt'].is_successful

                elif mode == "maj":
                    # Use memory but DO NOT store new results
                    result = judge_with_memory(
                        task=sample['task'],
                        agent_output=sample['agent_output'],
                        graph_manager=gm,
                        goal=EVALSBENCH_GOAL,
                        model=model
                    )
                    predicted = result['attempt'].is_successful

                elif mode == "mcts_judge":
                    config = MCTSConfig(model=model, num_rollouts=2, max_depth=3)
                    mcts = MCTSJudge(config)
                    result = mcts.evaluate(sample['task'], sample['agent_output'])
                    predicted = result['is_successful']

                elif mode == "mcts_judge_memory":
                    config = MCTSConfig(model=model, num_rollouts=2, max_depth=3)
                    result = run_mcts_judge_with_memory(
                        sample['task'], sample['agent_output'],
                        graph_manager=gm, goal=EVALSBENCH_GOAL,
                        mcts_config=config, model=model, store=False
                    )
                    predicted = result['is_successful']

                else:
                    raise ValueError(f"Unknown mode: {mode}")

                elapsed = time.time() - start
                total_time += elapsed
                expected = sample['expected']
                is_correct = predicted == expected

                if is_correct:
                    correct += 1

                results.append({
                    'idx': idx,
                    'topic': sample['topic'],
                    'expected': expected,
                    'predicted': predicted,
                    'correct': is_correct,
                    'latency_s': round(elapsed, 2),
                })

            except Exception as e:
                elapsed = time.time() - start
                total_time += elapsed
                print(f"  ERROR: {e}")
                results.append({
                    'idx': idx,
                    'topic': sample['topic'],
                    'expected': sample['expected'],
                    'predicted': None,
                    'correct': False,
                    'latency_s': round(elapsed, 2),
                })

    # Frozen-memory audit: snapshot after the run and diff against before.
    snap_after = gm.snapshot()
    diff = GraphManager.diff_snapshots(snap_before, snap_after)
    print(f"  [audit] after:  {snap_after['total_nodes']} nodes, "
          f"{snap_after['total_edges']} edges, "
          f"fingerprint={snap_after['fingerprint'][:16]}...")
    if diff['identical']:
        print("  [audit] PASS: memory unchanged during evaluation.")
    else:
        print(f"  [audit] FAIL: memory changed during evaluation. "
              f"node_delta={diff['node_delta']}, edge_delta={diff['edge_delta']}")

    if audit_log_path is not None:
        import json
        with open(audit_log_path, "w") as f:
            json.dump({
                "mode": mode,
                "before": snap_before,
                "after": snap_after,
                "diff": diff,
            }, f, indent=2)
        print(f"  [audit] log: {audit_log_path}")

    accuracy = correct / len(results) * 100 if results else 0
    avg_latency = total_time / len(results) if results else 0

    return pd.DataFrame(results), accuracy, avg_latency


def run_experiment(experiment, train_df, test_df, gm, model, poison_rate=0.0,
                   seed=42, tag=""):
    """
    Run a single experiment: build memory, then evaluate.

    ``tag`` is appended to output filenames so multi-seed runs do not
    overwrite one another (e.g. tag="_seed123").
    """
    gm.clear_all()
    print(f"\nNeo4j cleared for experiment: {experiment}")

    # Build memory based on experiment type
    if experiment == "self_written":
        build_memory(train_df, gm, model)
    elif experiment == "oracle":
        build_oracle_memory(train_df, gm)
    elif experiment.startswith("poisoned"):
        build_poisoned_memory(train_df, gm, poison_rate, seed)
    elif experiment == "no_memory":
        print("  No memory built (stateless baseline)")
    else:
        raise ValueError(f"Unknown experiment: {experiment}")

    # Evaluate — only run memory modes since stateless is constant (65.0%)
    if experiment == "no_memory":
        eval_modes = ['stateless']
    else:
        eval_modes = ['maj', 'mcts_judge_memory']

    results = {}
    for mode in eval_modes:
        print(f"\n--- Evaluating: {mode} ---")
        audit_log = RESULTS_DIR / f"lf_{experiment}_{mode}{tag}_audit.json"
        results_df, accuracy, avg_latency = evaluate_test_set(
            test_df, mode, gm, model, audit_log_path=audit_log
        )

        output_file = RESULTS_DIR / f"lf_{experiment}_{mode}{tag}.csv"
        results_df.to_csv(output_file, index=False)

        results[mode] = {'accuracy': accuracy, 'avg_latency': avg_latency}
        print(f"  {mode}: {accuracy:.1f}% ({avg_latency:.1f}s)")

    return results


def main():
    parser = argparse.ArgumentParser(description="Leakage-Free Benchmark")
    parser.add_argument('--model', type=str, default='gpt-4o')
    parser.add_argument('--experiment', type=str, default='all',
                       choices=['all', 'self_written', 'oracle', 'poisoned', 'compare'],
                       help='Which experiment to run')
    parser.add_argument('--train-ratio', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    benchmark_df = pd.read_csv(DATA_PATH / "benchmark_df.csv")

    print("=" * 60)
    print("LEAKAGE-FREE BENCHMARK")
    print("=" * 60)
    print(f"Total samples: {len(benchmark_df)}")
    print(f"Unique questions: {benchmark_df['question'].nunique()}")
    print(f"Model: {args.model}")
    print(f"Experiment: {args.experiment}")

    train_df, test_df = split_by_question(benchmark_df, args.train_ratio, args.seed)
    print(f"\nTrain: {len(train_df)} samples ({len(train_df)//2} questions)")
    print(f"Test:  {len(test_df)} samples ({len(test_df)//2} questions)")

    gm = GraphManager()
    all_experiments = {}

    if args.experiment in ['all', 'compare']:
        # For seed 42 we already have no_memory and self_written; for other
        # seeds we need the full set so the multi-seed CIs are complete.
        if args.seed == 42:
            experiments = [
                ("oracle", 0.0),
                ("poisoned_10", 0.10),
                ("poisoned_20", 0.20),
                ("poisoned_50", 0.50),
            ]
        else:
            experiments = [
                ("no_memory", 0.0),
                ("self_written", 0.0),
                ("oracle", 0.0),
                ("poisoned_10", 0.10),
                ("poisoned_20", 0.20),
                ("poisoned_50", 0.50),
            ]
    elif args.experiment == 'self_written':
        experiments = [("self_written", 0.0)]
    elif args.experiment == 'oracle':
        experiments = [("oracle", 0.0)]
    elif args.experiment == 'poisoned':
        experiments = [
            ("poisoned_10", 0.10),
            ("poisoned_20", 0.20),
            ("poisoned_50", 0.50),
        ]

    tag = "" if args.seed == 42 else f"_seed{args.seed}"

    for exp_name, poison_rate in experiments:
        print(f"\n{'='*60}")
        print(f"EXPERIMENT: {exp_name.upper()}  (seed={args.seed})")
        print(f"{'='*60}")

        results = run_experiment(
            exp_name, train_df, test_df, gm, args.model, poison_rate,
            args.seed, tag=tag
        )
        all_experiments[exp_name] = results

    # Final summary
    print(f"\n{'='*60}")
    print("FULL RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Train: {len(train_df)} samples | Test: {len(test_df)} samples")
    print(f"Model: {args.model}\n")

    print(f"{'Experiment':<20} {'Mode':<25} {'Accuracy':>10}")
    print("-" * 58)
    for exp_name, modes in all_experiments.items():
        for mode, stats in modes.items():
            print(f"{exp_name:<20} {mode:<25} {stats['accuracy']:>9.1f}%")
        print()
    print("=" * 60)


if __name__ == "__main__":
    main()
