"""
MCTS Benchmark on EvalsBench Dataset

Runs all 6 evaluation modes one by one:
1. Stateless (baseline)
2. MAJ (memory-assisted)
3. MCTS-Judge only (no memory)
4. MCTS-Judge + Memory
5. MCTS-Retrieval + Standard Judge
6. Full MCTS (retrieval + judge)

Usage:
    python benchmark_mcts.py --mode stateless
    python benchmark_mcts.py --mode maj
    python benchmark_mcts.py --mode mcts_judge
    python benchmark_mcts.py --mode mcts_judge_memory
    python benchmark_mcts.py --mode mcts_retrieval
    python benchmark_mcts.py --mode full_mcts
    python benchmark_mcts.py --mode all
"""

import sys
import time
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path

sys.path.insert(0, 'src')

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
)
from mcts_retrieval import RetrievalConfig

# ============================================================
# CONFIG
# ============================================================
SAMPLE_SIZE = 30
MODEL = "gpt-4o-mini"
DATA_PATH = Path("data")
RESULTS_DIR = Path("results")

EVALSBENCH_GOAL = """Check if the response contains points mentioned from the grading notes and return 'pass' or 'fail'."""


def evalsbench_to_maj(row):
    """Convert EvalsBench row to MAJ format."""
    return {
        'task': f"grading_notes: {row['grading_notes']}",
        'agent_output': row['response'],
        'expected': row['target'] == 'pass',
        'topic': row['topic']
    }


def run_benchmark_mode(mode, df, gm=None):
    """Run a single benchmark mode on the dataset."""
    results = []
    correct = 0
    total_time = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Mode: {mode}"):
        sample = evalsbench_to_maj(row)
        start = time.time()

        try:
            if mode == "stateless":
                result = run_stateless(
                    sample['task'], sample['agent_output'],
                    goal=EVALSBENCH_GOAL, model=MODEL
                )
                predicted = result['attempt'].is_successful
                reasoning = result['attempt'].reasoning[:300]
                extra = {}

            elif mode == "maj":
                result = run_maj(
                    sample['task'], sample['agent_output'],
                    graph_manager=gm, goal=EVALSBENCH_GOAL, model=MODEL
                )
                predicted = result['attempt'].is_successful
                reasoning = result['attempt'].reasoning[:300]
                extra = {"memory_used": result.get('memory_used', {})}

                # Store in memory for future samples
                _store_in_memory(gm, result)

            elif mode == "mcts_judge":
                config = MCTSConfig(model=MODEL, num_rollouts=4, max_depth=5)
                result = run_mcts_judge(
                    sample['task'], sample['agent_output'],
                    goal=EVALSBENCH_GOAL, config=config
                )
                predicted = result['is_successful']
                reasoning = result['reasoning'][:300]
                extra = {"stats": result.get('stats', {})}

            elif mode == "mcts_judge_memory":
                config = MCTSConfig(model=MODEL, num_rollouts=4, max_depth=5)
                result = run_mcts_judge_with_memory(
                    sample['task'], sample['agent_output'],
                    graph_manager=gm, goal=EVALSBENCH_GOAL,
                    mcts_config=config, model=MODEL
                )
                predicted = result['is_successful']
                reasoning = result['reasoning'][:300]
                extra = {
                    "stats": result.get('stats', {}),
                    "memory_used": result.get('memory_used', {})
                }

            elif mode == "mcts_retrieval":
                r_config = RetrievalConfig(num_rollouts=4, max_depth=3)
                result = run_mcts_retrieval_with_judge(
                    sample['task'], sample['agent_output'],
                    graph_manager=gm, goal=EVALSBENCH_GOAL,
                    retrieval_config=r_config, model=MODEL
                )
                predicted = result['is_successful']
                reasoning = result['reasoning'][:300]
                extra = {"retrieval_stats": result.get('retrieval_stats', {})}

            elif mode == "full_mcts":
                r_config = RetrievalConfig(num_rollouts=4, max_depth=3)
                j_config = MCTSConfig(model=MODEL, num_rollouts=4, max_depth=5)
                result = run_full_mcts(
                    sample['task'], sample['agent_output'],
                    graph_manager=gm, goal=EVALSBENCH_GOAL,
                    retrieval_config=r_config, judge_config=j_config, model=MODEL
                )
                predicted = result['is_successful']
                reasoning = result['reasoning'][:300]
                extra = {
                    "judge_stats": result.get('judge_stats', {}),
                    "retrieval_stats": result.get('retrieval_stats', {}),
                }

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
                'reasoning': reasoning,
                'latency_s': round(elapsed, 2),
                **{k: str(v) for k, v in extra.items()}
            })

            # Progress
            current_acc = correct / len(results) * 100
            print(f"  [{len(results)}/{len(df)}] correct={is_correct} | running acc: {current_acc:.1f}% | time: {elapsed:.1f}s")

        except Exception as e:
            elapsed = time.time() - start
            total_time += elapsed
            print(f"  ERROR on sample {idx}: {e}")
            results.append({
                'idx': idx,
                'topic': sample['topic'],
                'expected': sample['expected'],
                'predicted': None,
                'correct': False,
                'reasoning': str(e),
                'latency_s': round(elapsed, 2),
            })

    accuracy = correct / len(results) * 100 if results else 0
    avg_latency = total_time / len(results) if results else 0

    return pd.DataFrame(results), accuracy, avg_latency, total_time


def _store_in_memory(gm, result):
    """Store MAJ result in memory graph for future samples."""
    try:
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
        print(f"  Warning: Failed to store in memory: {e}")


def main():
    parser = argparse.ArgumentParser(description="MCTS Benchmark on EvalsBench")
    parser.add_argument('--mode', type=str, required=True,
                       choices=['stateless', 'maj', 'mcts_judge', 'mcts_judge_memory',
                               'mcts_retrieval', 'full_mcts', 'all'],
                       help='Evaluation mode to benchmark')
    parser.add_argument('--samples', type=int, default=SAMPLE_SIZE,
                       help=f'Number of samples (default: {SAMPLE_SIZE})')
    parser.add_argument('--model', type=str, default=MODEL,
                       help=f'Model to use (default: {MODEL})')
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    # Load data
    benchmark_df = pd.read_csv(DATA_PATH / "benchmark_df.csv")
    sample_df = benchmark_df.sample(n=args.samples, random_state=42)

    print("=" * 60)
    print("MCTS BENCHMARK ON EVALSBENCH")
    print("=" * 60)
    print(f"Samples: {args.samples}")
    print(f"Model: {args.model}")
    print(f"EvalsBench Baseline (Vanilla): 84.49%")
    print("=" * 60)

    modes = ['stateless', 'maj', 'mcts_judge', 'mcts_judge_memory',
             'mcts_retrieval', 'full_mcts'] if args.mode == 'all' else [args.mode]

    # Need GraphManager for memory modes
    needs_memory = {'maj', 'mcts_judge_memory', 'mcts_retrieval', 'full_mcts'}
    gm = None
    if needs_memory & set(modes):
        gm = GraphManager()
        gm.clear_all()
        print("Neo4j memory initialized and cleared.\n")

    all_results = {}

    for mode in modes:
        print(f"\n{'='*60}")
        print(f"RUNNING: {mode.upper()}")
        print(f"{'='*60}\n")

        # Reset memory for each mode that uses it
        if mode in needs_memory and gm:
            gm.clear_all()

        results_df, accuracy, avg_latency, total_time = run_benchmark_mode(
            mode, sample_df, gm
        )

        # Save results
        output_file = RESULTS_DIR / f"results_{mode}.csv"
        results_df.to_csv(output_file, index=False)

        all_results[mode] = {
            'accuracy': accuracy,
            'avg_latency': avg_latency,
            'total_time': total_time,
        }

        print(f"\n--- {mode.upper()} RESULTS ---")
        print(f"Accuracy:     {accuracy:.1f}%")
        print(f"Avg Latency:  {avg_latency:.1f}s per sample")
        print(f"Total Time:   {total_time:.0f}s")
        print(f"Saved to:     {output_file}")

    # Final summary
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("FINAL COMPARISON")
        print(f"{'='*60}")
        print(f"{'Mode':<25} {'Accuracy':>10} {'Avg Latency':>12} {'Total Time':>12}")
        print("-" * 60)
        for mode, stats in all_results.items():
            print(f"{mode:<25} {stats['accuracy']:>9.1f}% {stats['avg_latency']:>10.1f}s {stats['total_time']:>10.0f}s")
        print(f"\nEvalsBench Baseline:       84.49%")
        print("=" * 60)


if __name__ == "__main__":
    main()
