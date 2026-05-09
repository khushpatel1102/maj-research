"""
run_bare_ablation.py -- structure ablation runner.

Builds a "bare" self-written memory containing only Policy and Attempt
nodes (no Issue, Fix, or Semantic extraction) and evaluates MAJ on the
same held-out test split used in the rest of the leakage-free protocol.

This directly answers Bader's structure-ablation question: does the
elaborate 5-node typed graph beat a plain contrastive memory of past
attempts? If MAJ accuracy on the bare condition is comparable to MAJ on
the full self-written schema, the extra schema is not earning its
complexity.

The runner uses the same frozen-memory audit and the same evaluate_test_set
machinery as benchmark_leakage_free.py, so output CSVs and audit JSONs
plug into analyze_stats.py and reproduce_all.py automatically.

Usage
-----
    python run_bare_ablation.py --model gpt-4o --seed 42
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, "src")

from models import Policy, Attempt
from graph_manager import GraphManager
from benchmark_leakage_free import (
    DATA_PATH, RESULTS_DIR, EVALSBENCH_GOAL,
    evalsbench_to_maj, split_by_question, evaluate_test_set,
)
from judge import judge as run_judge  # not used here but keeps imports honest


def build_bare_memory(train_df, gm):
    """
    Build memory with Policy and Attempt nodes only.

    Labels are the judge's own verdicts on the training set, exactly
    like self_written. The difference is that no Issue, Fix, or Semantic
    extraction is performed — the graph carries only the contrastive
    attempt signal.
    """
    from judge import judge_with_memory  # imported here to avoid circulars

    print(f"\nBuilding BARE memory (Policy + Attempt only) "
          f"from {len(train_df)} training samples...")
    for _, row in tqdm(train_df.iterrows(), total=len(train_df),
                       desc="bare memory"):
        sample = evalsbench_to_maj(row)
        try:
            result = judge_with_memory(
                task=sample["task"], agent_output=sample["agent_output"],
                graph_manager=gm, goal=EVALSBENCH_GOAL, model="gpt-4o"
            )
            # Only commit the Policy and Attempt; ignore issues, fixes, semantics.
            gm.create_policy(result["policy"])
            gm.create_attempt(result["attempt"])
            for rel in result["relationships"]:
                if rel["type"] == "SATISFIES":
                    gm.link_attempt_satisfies_policy(
                        rel["from_id"], rel["to_id"])
        except Exception as e:
            print(f"  build error: {e}")

    counts = gm.driver.execute_query(
        "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS cnt")
    print("Bare memory contents:")
    for r in counts.records:
        print(f"  {r['label']}: {r['cnt']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(DATA_PATH / "benchmark_df.csv")
    train_df, test_df = split_by_question(df, 0.5, args.seed)

    print("=" * 60)
    print(f"STRUCTURE ABLATION: BARE MEMORY  (seed {args.seed})")
    print("=" * 60)
    print(f"Train: {len(train_df)} samples ({len(train_df)//2} questions)")
    print(f"Test:  {len(test_df)} samples ({len(test_df)//2} questions)")

    gm = GraphManager()
    gm.clear_all()
    build_bare_memory(train_df, gm)

    tag = "" if args.seed == 42 else f"_seed{args.seed}"
    out_csv = RESULTS_DIR / f"lf_bare_maj{tag}.csv"
    out_audit = RESULTS_DIR / f"lf_bare_maj{tag}_audit.json"

    print(f"\nEvaluating MAJ on the bare memory graph...")
    results_df, acc, lat = evaluate_test_set(
        test_df, "maj", gm, args.model, audit_log_path=out_audit
    )
    results_df.to_csv(out_csv, index=False)

    print("\n" + "=" * 60)
    print("STRUCTURE-ABLATION RESULTS")
    print("=" * 60)
    print(f"  bare maj : {acc:.1f}%  ({lat:.1f}s/sample)")
    print(f"  CSV      : {out_csv}")
    print(f"  audit    : {out_audit}")
    print()
    print("Compare against (same seed):")
    print(f"  full self-written maj  -> results/leakage_free_maj.csv  "
          f"(seed 42 = 63.7%)")
    print(f"  full self-written maj  -> results/lf_self_written_maj_seed123.csv "
          f"(seed 123 = 67.5%)")
    print(f"  stateless              -> results/leakage_free_stateless.csv "
          f"(seed 42 = 65.0%)")


if __name__ == "__main__":
    main()
