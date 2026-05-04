"""
Harness Experiments: reliability stress tests for MCTS-MAJ.

Runs three experiments that back the LLM Judge Harness framing:

  B. Stochastic stability    -- run same input twice, measure verdict agreement
  C. Label flip              -- invert response label, measure verdict flip rate
  D. Defense mechanism       -- similarity-threshold filter on poisoned memory

All use the leakage-free protocol: question-level split, frozen memory.
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
    run_mcts_judge_with_memory,
    store_mcts_result,
)
from openai import OpenAI
import os
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DATA_PATH = Path("data")
RESULTS_DIR = Path("results")

EVALSBENCH_GOAL = """Check if the response contains points mentioned from the grading notes and return 'pass' or 'fail'."""


def evalsbench_to_maj(row):
    return {
        'task': f"grading_notes: {row['grading_notes']}",
        'agent_output': row['response'],
        'expected': row['target'] == 'pass',
        'topic': row['topic'],
        'question': row['question'],
    }


def split_by_question(df, train_ratio=0.5, seed=42):
    questions = df['question'].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(questions)
    split_idx = int(len(questions) * train_ratio)
    train_q = set(questions[:split_idx])
    test_q = set(questions[split_idx:])
    return df[df['question'].isin(train_q)], df[df['question'].isin(test_q)]


# ============================================================
# JUDGE EVAL (with optional memory)
# ============================================================
def evaluate_one(mode, sample, gm, model, mcts_config=None):
    if mode == "stateless":
        result = run_stateless(sample['task'], sample['agent_output'],
                               goal=EVALSBENCH_GOAL, model=model)
        return result['attempt'].is_successful

    if mode == "maj":
        result = judge_with_memory(
            task=sample['task'],
            agent_output=sample['agent_output'],
            graph_manager=gm,
            goal=EVALSBENCH_GOAL,
            model=model
        )
        return result['attempt'].is_successful

    if mode == "mcts_judge_memory":
        config = mcts_config or MCTSConfig(model=model, num_rollouts=2, max_depth=3)
        result = run_mcts_judge_with_memory(
            sample['task'], sample['agent_output'],
            graph_manager=gm, goal=EVALSBENCH_GOAL,
            mcts_config=config, model=model
        )
        return result['is_successful']

    raise ValueError(f"unknown mode: {mode}")


# ============================================================
# EXPERIMENT B: STOCHASTIC STABILITY
# ============================================================
def stochastic_stability(test_df, gm, model, modes, n_subset=20):
    """
    Run each test sample twice; measure how often verdicts agree.
    Subsample to keep cost bounded.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT B: STOCHASTIC STABILITY")
    print("=" * 60)
    sub = test_df.sample(n=min(n_subset, len(test_df)), random_state=42)

    rows = []
    for mode in modes:
        agreements = 0
        total = 0
        for _, row in tqdm(sub.iterrows(), total=len(sub), desc=f"stability/{mode}"):
            sample = evalsbench_to_maj(row)
            try:
                v1 = evaluate_one(mode, sample, gm, model)
                v2 = evaluate_one(mode, sample, gm, model)
                rows.append({"mode": mode, "idx": row.name, "v1": v1, "v2": v2,
                             "agree": v1 == v2})
                if v1 == v2:
                    agreements += 1
                total += 1
            except Exception as e:
                rows.append({"mode": mode, "idx": row.name, "v1": None,
                             "v2": None, "agree": False, "error": str(e)})
        rate = agreements / total * 100 if total else 0
        print(f"  {mode}: {agreements}/{total} agree = {rate:.1f}%")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "harness_stability.csv", index=False)
    print(f"Saved: {RESULTS_DIR / 'harness_stability.csv'}")
    return df


# ============================================================
# EXPERIMENT C: LABEL FLIP DISCRIMINATIVE TEST
# ============================================================
class LabelFlippedResponse(BaseModel):
    flipped_response: str

def flip_response_label(task, response, target_label, model="gpt-4o"):
    """Generate a label-inverted version of the response."""
    direction = ("Rewrite this response so it now CLEARLY FAILS the rubric "
                 "(omits required points, gets facts wrong, or contradicts "
                 "the grading notes), while keeping the same topic and overall "
                 "structure.") if not target_label else \
                ("Rewrite this response so it now CLEARLY PASSES the rubric "
                 "(addresses every required point with sufficient depth and "
                 "accuracy), while keeping the same topic and overall structure.")

    prompt = f"""You are generating a label-flipped version of a response for evaluation harness testing.

GRADING CRITERIA:
{task[:500]}

ORIGINAL RESPONSE:
{response[:1500]}

TASK:
{direction}
Return only the rewritten response."""

    try:
        r = client.responses.parse(
            model=model,
            input=[{"role": "user", "content": prompt}],
            text_format=LabelFlippedResponse,
            temperature=0.4,
        )
        return r.output_parsed.flipped_response
    except Exception as e:
        return None


def label_flip(test_df, gm, model, modes, n_subset=20):
    """
    For each sample, generate an inverted version and check whether the
    judge correctly flips its verdict.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT C: LABEL FLIP DISCRIMINATIVE TEST")
    print("=" * 60)
    sub = test_df.sample(n=min(n_subset, len(test_df)), random_state=42)

    rows = []
    for mode in modes:
        flips_correct = 0
        total = 0
        for _, row in tqdm(sub.iterrows(), total=len(sub), desc=f"flip/{mode}"):
            sample = evalsbench_to_maj(row)
            try:
                # Original verdict
                orig_verdict = evaluate_one(mode, sample, gm, model)
                # Generate flipped response (target = not original label)
                target_label = not sample['expected']
                flipped = flip_response_label(sample['task'],
                                              sample['agent_output'],
                                              target_label, model=model)
                if flipped is None:
                    continue

                flipped_sample = dict(sample)
                flipped_sample['agent_output'] = flipped
                flipped_verdict = evaluate_one(mode, flipped_sample, gm, model)

                # A reliable judge flips its verdict
                correct_flip = (orig_verdict != flipped_verdict)
                rows.append({
                    "mode": mode, "idx": row.name,
                    "original_label": sample['expected'],
                    "original_verdict": orig_verdict,
                    "flipped_verdict": flipped_verdict,
                    "correctly_flipped": correct_flip,
                })
                if correct_flip:
                    flips_correct += 1
                total += 1
            except Exception as e:
                rows.append({"mode": mode, "idx": row.name, "error": str(e)})
        rate = flips_correct / total * 100 if total else 0
        print(f"  {mode}: {flips_correct}/{total} correctly flipped = {rate:.1f}%")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "harness_label_flip.csv", index=False)
    print(f"Saved: {RESULTS_DIR / 'harness_label_flip.csv'}")
    return df


# ============================================================
# EXPERIMENT D: DEFENSE MECHANISM
# ============================================================
def build_poisoned_memory(train_df, gm, poison_rate, seed=42):
    """Same as benchmark_leakage_free.py but inline for self-containment."""
    rng = np.random.RandomState(seed)
    flip_mask = rng.random(len(train_df)) < poison_rate

    for i, (_, row) in enumerate(tqdm(train_df.iterrows(), total=len(train_df),
                                       desc=f"poison@{poison_rate:.0%}")):
        sample = evalsbench_to_maj(row)
        label = sample['expected']
        if flip_mask[i]:
            label = not label

        policy = Policy(description=sample['task']).with_embedding()
        attempt = Attempt(
            agent_output=sample['agent_output'],
            is_successful=label,
            reasoning=f"Label: {'pass' if label else 'fail'}",
        ).with_embedding()
        gm.create_policy(policy)
        gm.create_attempt(attempt)
        gm.link_attempt_satisfies_policy(attempt.id, policy.id)


def defense_filter_eval(test_df, gm, model, similarity_floor):
    """
    Evaluate MCTS-Judge + Memory under a similarity-threshold filter.
    Memory is only used when retrieved similarity is above the floor.
    Implementation: monkey-patch graph_manager retrieval methods.
    """
    # Save originals
    orig_contrastive = gm.find_contrastive_attempts
    orig_issues = gm.find_similar_issues
    orig_patterns = gm.find_semantic_patterns

    def filt(items, key='score'):
        return [it for it in items if it.get(key, 0) >= similarity_floor]

    def filtered_contrastive(emb, top_k=3):
        r = orig_contrastive(emb, top_k=top_k)
        return {'positive': filt(r.get('positive', [])),
                'negative': filt(r.get('negative', []))}

    def filtered_issues(emb, top_k=5):
        return filt(orig_issues(emb, top_k=top_k))

    def filtered_patterns(emb, top_k=3):
        return filt(orig_patterns(emb, top_k=top_k), key='avg_similarity')

    gm.find_contrastive_attempts = filtered_contrastive
    gm.find_similar_issues = filtered_issues
    gm.find_semantic_patterns = filtered_patterns

    rows = []
    correct = 0
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="defense"):
        sample = evalsbench_to_maj(row)
        try:
            verdict = evaluate_one("mcts_judge_memory", sample, gm, model)
            is_correct = verdict == sample['expected']
            if is_correct:
                correct += 1
            rows.append({
                "idx": row.name, "topic": sample['topic'],
                "expected": sample['expected'],
                "predicted": verdict, "correct": is_correct,
            })
        except Exception as e:
            rows.append({"idx": row.name, "error": str(e), "correct": False})

    # Restore originals
    gm.find_contrastive_attempts = orig_contrastive
    gm.find_similar_issues = orig_issues
    gm.find_semantic_patterns = orig_patterns

    return pd.DataFrame(rows), correct / len(rows) * 100 if rows else 0


def defense_experiment(train_df, test_df, gm, model, similarity_floor=0.92):
    """
    Run MCTS-Judge + Memory under poisoned-50% memory:
    (1) without filter (baseline collapse)
    (2) with similarity filter (defense)
    """
    print("\n" + "=" * 60)
    print(f"EXPERIMENT D: DEFENSE MECHANISM (similarity_floor={similarity_floor})")
    print("=" * 60)

    # Build poisoned-50% memory
    gm.clear_all()
    build_poisoned_memory(train_df, gm, poison_rate=0.50, seed=42)

    # No defense
    print("\n[1/2] No defense (baseline poisoned)")
    df_nodef, acc_nodef = defense_filter_eval(test_df, gm, model,
                                              similarity_floor=0.0)
    df_nodef.to_csv(RESULTS_DIR / "harness_defense_off.csv", index=False)
    print(f"  No defense: {acc_nodef:.1f}%")

    # With defense
    print(f"\n[2/2] With similarity filter ({similarity_floor})")
    df_def, acc_def = defense_filter_eval(test_df, gm, model,
                                          similarity_floor=similarity_floor)
    df_def.to_csv(RESULTS_DIR / "harness_defense_on.csv", index=False)
    print(f"  With defense: {acc_def:.1f}%")

    print(f"\n  Recovery: {acc_def - acc_nodef:+.1f} points")


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='gpt-4o')
    parser.add_argument('--experiment', default='all',
                        choices=['stability', 'flip', 'defense', 'all'])
    parser.add_argument('--n_subset', type=int, default=20,
                        help='subsample size for stability and flip')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip_build', action='store_true',
                        help='reuse existing memory in Neo4j')
    parser.add_argument('--modes', default='stateless,maj,mcts_judge_memory',
                        help='comma-separated modes to run')
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    benchmark_df = pd.read_csv(DATA_PATH / "benchmark_df.csv")
    train_df, test_df = split_by_question(benchmark_df, 0.5, args.seed)
    print(f"Train: {len(train_df)}  Test: {len(test_df)}  Model: {args.model}")

    gm = GraphManager()
    modes = [m.strip() for m in args.modes.split(',')]

    # Build self-written memory once for stability + flip experiments
    if args.experiment in ('stability', 'flip', 'all') and not args.skip_build:
        gm.clear_all()
        print("\nBuilding self-written memory from training set...")
        for _, row in tqdm(train_df.iterrows(), total=len(train_df),
                            desc="building memory"):
            sample = evalsbench_to_maj(row)
            try:
                result = judge_with_memory(
                    task=sample['task'], agent_output=sample['agent_output'],
                    graph_manager=gm, goal=EVALSBENCH_GOAL, model=args.model
                )
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
                for i, sem in enumerate(result.get('semantics', [])):
                    semantic_rels = result.get('semantic_relationships', [])
                    if i < len(semantic_rels) and semantic_rels[i].get('is_new', True):
                        gm.get_or_create_semantic(sem)
                    if i < len(semantic_rels):
                        gm.link_issue_abstracts_to_semantic(
                            semantic_rels[i]['from_id'],
                            semantic_rels[i]['to_id']
                        )
            except Exception as e:
                print(f"  build error: {e}")

    if args.experiment in ('stability', 'all'):
        stochastic_stability(test_df, gm, args.model, modes, args.n_subset)

    if args.experiment in ('flip', 'all'):
        label_flip(test_df, gm, args.model, modes, args.n_subset)

    if args.experiment in ('defense', 'all'):
        defense_experiment(train_df, test_df, gm, args.model)


if __name__ == "__main__":
    main()
