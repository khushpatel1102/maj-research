"""
MAJ Benchmark on EvalsBench - Using EXACT EvalsBench prompt

Compare:
1. Stateless (no memory)
2. With Memory (organic learning)

Using the exact same prompt as EvalsBench optimized.
"""

import sys
import pandas as pd
from tqdm import tqdm
from pathlib import Path

sys.path.insert(0, 'src')

from judge import judge, judge_with_memory
from graph_manager import GraphManager

# ============================================================
# CONFIG
# ============================================================
SAMPLE_SIZE = 30
MODEL = "gpt-4o-mini"
DATA_PATH = Path("data")

# ============================================================
# EXACT EVALSBENCH OPTIMIZED PROMPT
# ============================================================
EVALSBENCH_PROMPT = """Check if the response contains points mentioned from the grading notes and return 'pass' or 'fail'."""


def evalsbench_to_maj(row):
    """Use exact EvalsBench format."""
    # Task = grading_notes (what to check for)
    # Agent output = response (what to evaluate)
    task = f"grading_notes: {row['grading_notes']}"

    return {
        'task': task,
        'agent_output': row['response'],
        'expected': row['target'] == 'pass',
        'topic': row['topic']
    }


def run_stateless(df, sample_size=None, model=MODEL):
    """Run stateless benchmark."""
    if sample_size:
        df = df.sample(n=sample_size, random_state=42)

    results = []
    correct = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Stateless"):
        sample = evalsbench_to_maj(row)

        try:
            result = judge(
                task=sample['task'],
                agent_output=sample['agent_output'],
                goal=EVALSBENCH_PROMPT,
                model=model
            )

            predicted = result['attempt'].is_successful
            expected = sample['expected']
            is_correct = predicted == expected

            if is_correct:
                correct += 1

            results.append({
                'expected': expected,
                'predicted': predicted,
                'correct': is_correct
            })

        except Exception as e:
            print(f"Error: {e}")
            results.append({
                'expected': sample['expected'],
                'predicted': None,
                'correct': False
            })

    accuracy = correct / len(results) * 100
    return pd.DataFrame(results), accuracy


def run_with_memory(df, gm, sample_size=None, model=MODEL):
    """Run with memory benchmark."""
    if sample_size:
        df = df.sample(n=sample_size, random_state=42)

    results = []
    correct = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="With Memory"):
        sample = evalsbench_to_maj(row)

        try:
            result = judge_with_memory(
                task=sample['task'],
                agent_output=sample['agent_output'],
                graph_manager=gm,
                goal=EVALSBENCH_PROMPT,
                model=model
            )

            predicted = result['attempt'].is_successful
            expected = sample['expected']
            is_correct = predicted == expected

            if is_correct:
                correct += 1

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

            semantic_rels = result.get('semantic_relationships', [])
            for i, semantic in enumerate(result.get('semantics', [])):
                if i < len(semantic_rels) and semantic_rels[i].get('is_new', True):
                    gm.get_or_create_semantic(semantic)
                if i < len(semantic_rels):
                    gm.link_issue_abstracts_to_semantic(
                        semantic_rels[i]['from_id'],
                        semantic_rels[i]['to_id']
                    )

            results.append({
                'expected': expected,
                'predicted': predicted,
                'correct': is_correct
            })

        except Exception as e:
            print(f"Error: {e}")
            results.append({
                'expected': sample['expected'],
                'predicted': None,
                'correct': False
            })

    accuracy = correct / len(results) * 100
    return pd.DataFrame(results), accuracy


def main():
    print("=" * 60)
    print("MAJ vs MEMORY COMPARISON")
    print("Using EXACT EvalsBench optimized prompt")
    print("=" * 60)

    # Load data
    benchmark_df = pd.read_csv(DATA_PATH / "benchmark_df.csv")
    print(f"\nSamples: {SAMPLE_SIZE}")
    print(f"Model: {MODEL}")

    print("\n" + "-" * 60)
    print("EvalsBench Baseline (gpt-4o-mini Vanilla): 84.49%")
    print("-" * 60)

    # Run with memory only
    print("\nRunning WITH MEMORY...")
    gm = GraphManager()
    gm.clear_all()

    memory_results, memory_acc = run_with_memory(benchmark_df, gm, sample_size=SAMPLE_SIZE)

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"With Memory:  {memory_acc:.1f}%")
    print("=" * 60)

    # Save
    memory_results.to_csv('results_memory.csv', index=False)


if __name__ == "__main__":
    main()
