"""
multiseed_fast.py -- fast, network-robust multi-seed runner.

Runs only the cheap, write-free evaluation modes (stateless, MAJ) across
several question-level splits, so the cross-seed table can be completed
without the ~40-minute MCTS evals that proved fragile to mid-run network
drops. The frozen-memory audit (snapshot + freeze) is applied exactly as
in benchmark_leakage_free.py.

Conditions covered per seed:
  - no_memory  -> stateless
  - self_written -> maj
  - oracle      -> maj

Output: results/lf_{condition}_{mode}_seed{seed}.csv plus _audit.json,
matching the naming used by benchmark_leakage_free.py so analyze_stats.py
and reproduce_all.py pick them up automatically.

Usage
-----
    python multiseed_fast.py --seeds 123 7 --model gpt-4o
"""

from __future__ import annotations

import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, "src")

from models import Policy, Attempt
from judge import judge, judge_with_memory
from graph_manager import GraphManager

DATA = Path("data")
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

GOAL = "Check if the response contains points mentioned from the grading notes and return 'pass' or 'fail'."


def to_maj(row):
    return {
        "task": f"grading_notes: {row['grading_notes']}",
        "agent_output": row["response"],
        "expected": row["target"] == "pass",
        "topic": row["topic"],
        "question": row["question"],
    }


def split_by_question(df, ratio=0.5, seed=42):
    qs = df["question"].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(qs)
    cut = int(len(qs) * ratio)
    train_q, test_q = set(qs[:cut]), set(qs[cut:])
    return df[df["question"].isin(train_q)], df[df["question"].isin(test_q)]


def build_self_written(train_df, gm, model):
    print("  building self-written memory...")
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="  mem"):
        s = to_maj(row)
        try:
            r = judge_with_memory(task=s["task"], agent_output=s["agent_output"],
                                  graph_manager=gm, goal=GOAL, model=model)
            gm.create_policy(r["policy"])
            gm.create_attempt(r["attempt"])
            for issue in r["issues"]:
                gm.create_issue(issue)
            for fix in r["fixes"]:
                gm.create_fix(fix)
            for rel in r["relationships"]:
                if rel["type"] == "SATISFIES":
                    gm.link_attempt_satisfies_policy(rel["from_id"], rel["to_id"])
                elif rel["type"] == "CAUSES":
                    gm.link_attempt_causes_issue(rel["from_id"], rel["to_id"])
                elif rel["type"] == "RESOLVES":
                    gm.link_fix_resolves_issue(rel["from_id"], rel["to_id"])
            for i, sem in enumerate(r.get("semantics", [])):
                srels = r.get("semantic_relationships", [])
                if i < len(srels) and srels[i].get("is_new", True):
                    gm.get_or_create_semantic(sem)
                if i < len(srels):
                    gm.link_issue_abstracts_to_semantic(srels[i]["from_id"], srels[i]["to_id"])
        except Exception as e:
            print(f"    build error: {e}")


def build_oracle(train_df, gm):
    print("  building oracle memory...")
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="  mem"):
        s = to_maj(row)
        try:
            pol = Policy(description=s["task"]).with_embedding()
            att = Attempt(agent_output=s["agent_output"],
                          is_successful=s["expected"],
                          reasoning=f"Oracle: ground truth {'pass' if s['expected'] else 'fail'}").with_embedding()
            gm.create_policy(pol)
            gm.create_attempt(att)
            gm.link_attempt_satisfies_policy(att.id, pol.id)
        except Exception as e:
            print(f"    build error: {e}")


def evaluate(test_df, mode, gm, model, audit_path):
    snap_before = gm.snapshot()
    print(f"  [audit] before: {snap_before['total_nodes']} nodes, "
          f"{snap_before['total_edges']} edges, "
          f"fingerprint={snap_before['fingerprint'][:16]}...")
    rows, correct, total_time = [], 0, 0.0
    with gm.freeze():
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"  eval/{mode}"):
            s = to_maj(row)
            t0 = time.time()
            try:
                if mode == "stateless":
                    r = judge(s["task"], s["agent_output"], goal=GOAL, model=model)
                    pred = r["attempt"].is_successful
                elif mode == "maj":
                    r = judge_with_memory(task=s["task"], agent_output=s["agent_output"],
                                          graph_manager=gm, goal=GOAL, model=model)
                    pred = r["attempt"].is_successful
                else:
                    raise ValueError(mode)
                dt = time.time() - t0
                total_time += dt
                ok = pred == s["expected"]
                correct += int(ok)
                rows.append({"idx": row.name, "topic": s["topic"],
                             "expected": s["expected"], "predicted": pred,
                             "correct": ok, "latency_s": round(dt, 2)})
            except Exception as e:
                dt = time.time() - t0
                total_time += dt
                print(f"    ERROR: {e}")
                rows.append({"idx": row.name, "topic": s["topic"],
                             "expected": s["expected"], "predicted": None,
                             "correct": False, "latency_s": round(dt, 2)})
    snap_after = gm.snapshot()
    diff = GraphManager.diff_snapshots(snap_before, snap_after)
    print(f"  [audit] after:  {snap_after['total_nodes']} nodes, "
          f"{snap_after['total_edges']} edges, "
          f"fingerprint={snap_after['fingerprint'][:16]}...")
    print(f"  [audit] {'PASS' if diff['identical'] else 'FAIL'}")
    with open(audit_path, "w") as f:
        json.dump({"mode": mode, "before": snap_before, "after": snap_after,
                   "diff": diff}, f, indent=2)
    acc = correct / len(rows) * 100 if rows else 0
    lat = total_time / len(rows) if rows else 0
    return pd.DataFrame(rows), acc, lat


def run_seed(seed, model):
    print("\n" + "=" * 60)
    print(f"SEED {seed}  (fast: stateless + maj only)")
    print("=" * 60)
    df = pd.read_csv(DATA / "benchmark_df.csv")
    train_df, test_df = split_by_question(df, 0.5, seed)
    print(f"Train {len(train_df)} samples | Test {len(test_df)} samples")
    gm = GraphManager()
    tag = f"_seed{seed}"

    # 1. no_memory / stateless
    print("\n[no_memory / stateless]")
    gm.clear_all()
    res, acc, lat = evaluate(test_df, "stateless", gm, model,
                             RESULTS / f"lf_no_memory_stateless{tag}_audit.json")
    res.to_csv(RESULTS / f"lf_no_memory_stateless{tag}.csv", index=False)
    print(f"  stateless: {acc:.1f}% ({lat:.1f}s)")

    # 2. self_written / maj
    print("\n[self_written / maj]")
    gm.clear_all()
    build_self_written(train_df, gm, model)
    res, acc, lat = evaluate(test_df, "maj", gm, model,
                             RESULTS / f"lf_self_written_maj{tag}_audit.json")
    res.to_csv(RESULTS / f"lf_self_written_maj{tag}.csv", index=False)
    print(f"  maj: {acc:.1f}% ({lat:.1f}s)")

    # 3. oracle / maj
    print("\n[oracle / maj]")
    gm.clear_all()
    build_oracle(train_df, gm)
    res, acc, lat = evaluate(test_df, "maj", gm, model,
                             RESULTS / f"lf_oracle_maj{tag}_audit.json")
    res.to_csv(RESULTS / f"lf_oracle_maj{tag}.csv", index=False)
    print(f"  maj (oracle): {acc:.1f}% ({lat:.1f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[123, 7])
    ap.add_argument("--model", default="gpt-4o")
    args = ap.parse_args()
    for seed in args.seeds:
        run_seed(seed, args.model)
    print("\nDone. Run analyze_stats.py / reproduce_all.py to refresh tables.")


if __name__ == "__main__":
    main()
