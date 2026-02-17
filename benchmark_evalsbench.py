"""
MAJ Benchmark on EvalsBench Dataset

Compares:
1. MAJ Stateless
2. MAJ With Memory (organic learning)

Against EvalsBench baselines (gpt-4o-mini):
- Vanilla: 84.49%
- Dynamic Few Shot: 76.65%
"""

import sys
import re
import pandas as pd
from tqdm import tqdm
from pathlib import Path

sys.path.insert(0, 'src')

from judge import judge, judge_with_memory
from graph_manager import GraphManager

# ============================================================
# CONFIG
# ============================================================
SAMPLE_SIZE = 100
MODEL = "gpt-4o-mini"
DATA_PATH = Path("data")

# ============================================================
# QA GOAL - Optimized for EvalsBench grading notation
# ============================================================
QA_GOAL = """You are evaluating whether a response comprehensively addresses all required points specified in the grading notes.

EVALUATION PROCESS:
1. First, carefully extract ALL individual points from the grading notes, including:
   - Points marked with asterisks (*) - these are critical
   - Points marked with exclamation marks (!) - these are essential
   - All other listed requirements, metrics, considerations, or criteria

2. For each point identified, systematically verify if the response:
   - Explicitly mentions or addresses the specific point
   - Provides relevant detail or explanation for that point
   - Uses similar terminology or clearly related concepts

3. Apply strict evaluation criteria:
   - ALL points must be present for a 'pass'
   - Missing even ONE required point results in 'fail'
   - Generic mentions without specific coverage = insufficient
   - Points must be substantively addressed, not just briefly mentioned

4. Final verification checklist:
   - Count total points in grading notes vs points covered in response
   - Verify critical points (*) and essential points (!) are all present
   - Ensure no abbreviated coverage of complex multi-part requirements

Return 'pass' only if the response comprehensively covers every single point in the grading notes. Return 'fail' if any point is missing, insufficiently covered, or only partially addressed."""


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def parse_grading_notes(notes):
    """Extract required (!) and important (*) points from grading notes."""
    required = re.findall(r'!([^!]+)!', notes)
    important = re.findall(r'\*([^*]+)\*', notes)
    return required, important


def evalsbench_to_maj(row):
    """Convert EvalsBench row to MAJ format - use EvalsBench format directly."""
    # Use same format as EvalsBench optimized prompt
    task = f"""grading_notes: {row['grading_notes']}"""

    return {
        'task': task,
        'agent_output': row['response'],
        'expected': row['target'] == 'pass',
        'topic': row['topic']
    }


def run_stateless_benchmark(df, sample_size=None, model=MODEL):
    """Run MAJ stateless on the benchmark."""
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
                goal=QA_GOAL,
                model=model
            )

            predicted = result['attempt'].is_successful
            expected = sample['expected']
            is_correct = predicted == expected

            if is_correct:
                correct += 1

            results.append({
                'topic': sample['topic'],
                'expected': expected,
                'predicted': predicted,
                'correct': is_correct,
                'reasoning': result['attempt'].reasoning[:200]
            })

        except Exception as e:
            print(f"Error on sample {idx}: {e}")
            results.append({
                'topic': sample['topic'],
                'expected': sample['expected'],
                'predicted': None,
                'correct': False,
                'reasoning': str(e)
            })

    accuracy = correct / len(results) * 100
    return pd.DataFrame(results), accuracy


def run_memory_benchmark(df, gm, sample_size=None, model=MODEL):
    """Run MAJ with organic memory on the benchmark."""
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
                goal=QA_GOAL,
                model=model
            )

            predicted = result['attempt'].is_successful
            expected = sample['expected']
            is_correct = predicted == expected

            if is_correct:
                correct += 1

            # Store in memory for future judgments
            gm.create_policy(result['policy'])
            gm.create_attempt(result['attempt'])
            for issue in result['issues']:
                gm.create_issue(issue)
            for fix in result['fixes']:
                gm.create_fix(fix)

            # Create relationships
            for rel in result['relationships']:
                if rel['type'] == 'SATISFIES':
                    gm.link_attempt_satisfies_policy(rel['from_id'], rel['to_id'])
                elif rel['type'] == 'CAUSES':
                    gm.link_attempt_causes_issue(rel['from_id'], rel['to_id'])
                elif rel['type'] == 'RESOLVES':
                    gm.link_fix_resolves_issue(rel['from_id'], rel['to_id'])

            # Store semantics
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
                'topic': sample['topic'],
                'expected': expected,
                'predicted': predicted,
                'correct': is_correct,
                'memory_used': result.get('memory_used', {}),
                'reasoning': result['attempt'].reasoning[:200]
            })

        except Exception as e:
            print(f"Error on sample {idx}: {e}")
            results.append({
                'topic': sample['topic'],
                'expected': sample['expected'],
                'predicted': None,
                'correct': False,
                'reasoning': str(e)
            })

    accuracy = correct / len(results) * 100
    return pd.DataFrame(results), accuracy


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("MAJ BENCHMARK ON EVALSBENCH")
    print("=" * 60)

    # Load data
    benchmark_df = pd.read_csv(DATA_PATH / "benchmark_df.csv")
    print(f"\nLoaded {len(benchmark_df)} samples")
    print(f"Running on: {SAMPLE_SIZE or 'ALL'} samples")
    print(f"Model: {MODEL}")

    # Baselines
    print("\n" + "-" * 60)
    print("EvalsBench Baselines (gpt-4o-mini):")
    print("-" * 60)
    print("Vanilla:          84.49%")
    print("Dynamic Few Shot: 76.65%")
    print("-" * 60)

    # Run with memory only
    print("\nRunning MAJ WITH MEMORY benchmark...")
    gm = GraphManager()
    gm.clear_all()
    print("Neo4j memory cleared\n")

    memory_results, memory_acc = run_memory_benchmark(
        benchmark_df,
        gm,
        sample_size=SAMPLE_SIZE
    )

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"MAJ With Memory:  {memory_acc:.2f}%")
    print(f"vs Vanilla (84.49%): {memory_acc - 84.49:+.2f}%")
    print(f"vs Dynamic (76.65%): {memory_acc - 76.65:+.2f}%")
    print("=" * 60)

    # Save results
    memory_results.to_csv('results_memory.csv', index=False)
    print("\nResults saved to results_memory.csv")


if __name__ == "__main__":
    main()
