"""
Revision experiments (supervisor revision round, 2026-06).

Implements four task groups in one harness. Every memory-mode evaluation runs
under the frozen-memory audit unless the design explicitly enables write-back.

  Task 1  2x2 leakage design        --task split2x2
      rows {row, question} x writeback {on, frozen}. This is the experiment
      that turns "the gain disappeared under the audited protocol" from a
      before/after anecdote into a controlled 2x2: the conventional gain
      should appear in (row-split, writeback-on) and vanish in
      (question-split, frozen).

  Task 2  memory-use controls       --task controls
      positive control "cheating" (paired twin / oracle exemplar injected)
      and negative controls "random", "shuffled", "irrelevant". Proves the
      judge CAN use memory when it is informative, and is not merely ignoring
      the context block.

  Task 3  grouped 5-fold CV          --task cv
      grouped (by question) 5-fold cross-validation for primary modes only
      {stateless, maj_asymmetric, maj_oracle, maj_balanced}. No MCTS. Writes
      per-fold, per-item CSVs for the question-cluster bootstrap / mixed-effects
      analysis in analyze_grouped.py.

  Task 4  leniency-anchor logging    --task anchor
      runs maj_asymmetric and maj_balanced with full per-item retrieval logs
      (pos/neg counts, similarities, top-1 label, memory tokens, flip vs
      stateless). If balanced removes the false-pass tilt, remedy (a) is
      confirmed.

All four can be run with --task all. Requires Neo4j running and OPENAI_API_KEY.
"""

import sys
import time
import json
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, "src")

from models import Policy, Attempt, get_embedding
from judge import judge_with_memory
from judge_instrumented import judge_with_memory_instrumented
from graph_manager import GraphManager
from mcts_pipeline import run_stateless

DATA_PATH = Path("data")
OUT = Path("results/revision")
EVALSBENCH_GOAL = "Check if the response contains points mentioned from the grading notes and return 'pass' or 'fail'."


def to_sample(row):
    return {
        "task": f"grading_notes: {row['grading_notes']}",
        "agent_output": row["response"],
        "expected": row["target"] == "pass",
        "topic": row["topic"],
        "question": row["question"],
    }


# ----------------------------------------------------------------------
# retry wrapper (same policy as benchmark_leakage_free.py)
# ----------------------------------------------------------------------
def _is_transient(exc):
    m = (str(exc) + type(exc).__name__).lower()
    return any(k in m for k in (
        "connection", "timeout", "timed out", "rate limit", "ratelimit",
        "503", "502", "504", "apiconnection", "remoteprotocol",
        "remotedisconnected", "incomplete read"))


def with_retries(fn, max_attempts=5):
    last = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:
            last = exc
            if not _is_transient(exc) or attempt == max_attempts:
                raise
            time.sleep(min(30.0, 2.0 ** attempt) + random.uniform(0, 0.5))
    raise last


# ----------------------------------------------------------------------
# splits
# ----------------------------------------------------------------------
def split_by_question(df, seed=42, train_ratio=0.5):
    q = df["question"].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(q)
    cut = int(len(q) * train_ratio)
    train_q, test_q = set(q[:cut]), set(q[cut:])
    return df[df["question"].isin(train_q)], df[df["question"].isin(test_q)]


def split_by_row(df, seed=42, train_ratio=0.5):
    """Row-level split: paired twins CAN straddle the split (the leaky design)."""
    rng = np.random.RandomState(seed)
    idx = df.index.to_numpy().copy()
    rng.shuffle(idx)
    cut = int(len(idx) * train_ratio)
    return df.loc[idx[:cut]], df.loc[idx[cut:]]


def grouped_kfold(df, k=5, seed=42):
    """Yield (train_df, test_df) folds split by question group."""
    q = df["question"].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(q)
    folds = np.array_split(q, k)
    for i in range(k):
        test_q = set(folds[i])
        train_q = set(np.concatenate([folds[j] for j in range(k) if j != i]))
        yield i, df[df["question"].isin(train_q)], df[df["question"].isin(test_q)]


# ----------------------------------------------------------------------
# memory builders
# ----------------------------------------------------------------------
def build_self_written(train_df, gm, model):
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="mem:self"):
        s = to_sample(row)
        try:
            r = judge_with_memory(task=s["task"], agent_output=s["agent_output"],
                                  graph_manager=gm, goal=EVALSBENCH_GOAL, model=model)
            gm.create_policy(r["policy"]); gm.create_attempt(r["attempt"])
            for issue in r["issues"]: gm.create_issue(issue)
            for fix in r["fixes"]: gm.create_fix(fix)
            for rel in r["relationships"]:
                if rel["type"] == "SATISFIES": gm.link_attempt_satisfies_policy(rel["from_id"], rel["to_id"])
                elif rel["type"] == "CAUSES": gm.link_attempt_causes_issue(rel["from_id"], rel["to_id"])
                elif rel["type"] == "RESOLVES": gm.link_fix_resolves_issue(rel["from_id"], rel["to_id"])
        except Exception as e:
            print(f"  build err: {e}")


def build_oracle(train_df, gm):
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="mem:oracle"):
        s = to_sample(row)
        p = Policy(description=s["task"]).with_embedding()
        a = Attempt(agent_output=s["agent_output"], is_successful=s["expected"],
                    reasoning=f"Oracle: {'pass' if s['expected'] else 'fail'}").with_embedding()
        gm.create_policy(p); gm.create_attempt(a)
        gm.link_attempt_satisfies_policy(a.id, p.id)


def pool_attempts(gm):
    """Flat list of stored attempts for the random/irrelevant controls."""
    rows = gm.driver.execute_query(
        "MATCH (a:Attempt) RETURN a.id AS id, a.agent_output AS agent_output, "
        "a.is_successful AS is_successful, a.reasoning AS reasoning"
    )
    return [{"id": r["id"], "agent_output": r["agent_output"],
             "is_successful": r["is_successful"], "reasoning": r["reasoning"],
             "score": 0.0} for r in rows.records]


# ----------------------------------------------------------------------
# evaluation core
# ----------------------------------------------------------------------
def eval_rows(test_df, fn, gm, frozen=True, desc="eval", audit_path=None):
    """Run fn(sample)->predicted over test_df. Audit unless frozen=False."""
    snap_before = gm.snapshot() if frozen else None
    rows = []
    ctx = gm.freeze() if frozen else _nullctx()
    with ctx:
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc=desc):
            s = to_sample(row); t0 = time.time()
            try:
                pred, log = fn(s)
                rec = {"idx": idx, "question": s["question"], "topic": s["topic"],
                       "expected": s["expected"], "predicted": pred,
                       "correct": (pred == s["expected"]), "latency_s": round(time.time() - t0, 2)}
                if log: rec.update({f"log_{k}": v for k, v in log.items()})
            except Exception as e:
                print(f"  ERROR {idx}: {e}")
                rec = {"idx": idx, "question": s["question"], "topic": s["topic"],
                       "expected": s["expected"], "predicted": np.nan,
                       "correct": np.nan, "latency_s": round(time.time() - t0, 2)}
            rows.append(rec)
    df = pd.DataFrame(rows)
    audit = None
    if frozen:
        snap_after = gm.snapshot()
        diff = GraphManager.diff_snapshots(snap_before, snap_after)
        audit = {"before": snap_before, "after": snap_after, "diff": diff}
        print(f"  [audit] {'PASS' if diff['identical'] else 'FAIL'} "
              f"({snap_before['total_nodes']}->{snap_after['total_nodes']} nodes)")
        if audit_path:
            Path(audit_path).write_text(json.dumps(audit, indent=2))
    comp = df.dropna(subset=["correct"])
    acc = comp["correct"].mean() * 100 if len(comp) else 0.0
    return df, acc, audit


class _nullctx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


# ----------------------------------------------------------------------
# fn factories
# ----------------------------------------------------------------------
def fn_stateless(model):
    def f(s):
        r = with_retries(lambda: run_stateless(s["task"], s["agent_output"],
                                               goal=EVALSBENCH_GOAL, model=model))
        return r["attempt"].is_successful, None
    return f


def fn_maj_writeback(model, gm):
    """MAJ that WRITES its verdict back to memory (the leaky conventional path)."""
    def f(s):
        r = with_retries(lambda: judge_with_memory(
            task=s["task"], agent_output=s["agent_output"],
            graph_manager=gm, goal=EVALSBENCH_GOAL, model=model))
        # write-back: store the just-judged item
        gm.create_policy(r["policy"]); gm.create_attempt(r["attempt"])
        gm.link_attempt_satisfies_policy(r["attempt"].id, r["policy"].id)
        return r["attempt"].is_successful, None
    return f


def fn_maj(model, gm, mode, k=3, all_attempts=None, twin_lookup=None):
    def f(s):
        cheat = None
        if mode == "cheating" and twin_lookup is not None:
            cheat = twin_lookup.get(s["question"])
        r = with_retries(lambda: judge_with_memory_instrumented(
            s["task"], s["agent_output"], gm, goal=EVALSBENCH_GOAL, model=model,
            retrieval_mode=mode, k=k, all_attempts=all_attempts, cheat_exemplar=cheat))
        return r["attempt"].is_successful, r["retrieval_log"]
    return f


# ----------------------------------------------------------------------
# tasks
# ----------------------------------------------------------------------
def task_split2x2(df, gm, model, seed):
    print("\n=== TASK 1: 2x2 leakage design ===")
    OUT.mkdir(parents=True, exist_ok=True)
    for split_name, splitter in (("question", split_by_question), ("row", split_by_row)):
        train, test = splitter(df, seed=seed)
        for wb_name, frozen in (("frozen", True), ("writeback", False)):
            gm.clear_all()
            build_self_written(train, gm, model)
            fn = fn_maj_writeback(model, gm) if not frozen else fn_maj(model, gm, "asymmetric")
            tag = f"2x2_{split_name}_{wb_name}_seed{seed}"
            d, acc, _ = eval_rows(test, fn, gm, frozen=frozen, desc=tag,
                                  audit_path=OUT / f"{tag}_audit.json" if frozen else None)
            d.to_csv(OUT / f"{tag}.csv", index=False)
            print(f"  {split_name:9s} / {wb_name:9s}: {acc:.1f}%  (n={d['correct'].notna().sum()})")


def task_controls(df, gm, model, seed):
    print("\n=== TASK 2: memory-use controls ===")
    OUT.mkdir(parents=True, exist_ok=True)
    train, test = split_by_question(df, seed=seed)
    # twin lookup: the paired (opposite-label) row for each TEST question,
    # taken from the FULL df so it is the genuine oracle twin.
    twin = {}
    for q, grp in df.groupby("question"):
        for _, row in grp.iterrows():
            s = to_sample(row)
            twin[q] = {"agent_output": s["agent_output"], "is_successful": s["expected"],
                       "reasoning": f"ORACLE twin: ground truth {'pass' if s['expected'] else 'fail'}",
                       "score": 0.99}
    gm.clear_all(); build_oracle(train, gm)        # oracle memory base
    pool = pool_attempts(gm)
    # stateless baseline for reference
    sd, sacc, _ = eval_rows(test, fn_stateless(model), gm, desc="ctrl:stateless")
    sd.to_csv(OUT / f"controls_stateless_seed{seed}.csv", index=False)
    print(f"  stateless           : {sacc:.1f}%")
    for mode in ("cheating", "random", "shuffled", "irrelevant"):
        # cheating must NOT freeze writes? No: injection is in-context only,
        # memory graph is untouched, so the frozen audit still applies.
        fn = fn_maj(model, gm, mode, all_attempts=pool, twin_lookup=twin)
        d, acc, _ = eval_rows(test, fn, gm, desc=f"ctrl:{mode}",
                              audit_path=OUT / f"controls_{mode}_seed{seed}_audit.json")
        d.to_csv(OUT / f"controls_{mode}_seed{seed}.csv", index=False)
        print(f"  {mode:20s}: {acc:.1f}%")


def task_cv(df, gm, model, seed, k=5):
    print(f"\n=== TASK 3: grouped {k}-fold CV (primary modes only) ===")
    OUT.mkdir(parents=True, exist_ok=True)
    modes = ["stateless", "maj_asymmetric", "maj_oracle", "maj_balanced"]
    for i, train, test in grouped_kfold(df, k=k, seed=seed):
        # build both memories once per fold
        for mode in modes:
            if mode == "stateless":
                gm.clear_all()
                fn = fn_stateless(model)
            elif mode == "maj_asymmetric":
                gm.clear_all(); build_self_written(train, gm, model)
                fn = fn_maj(model, gm, "asymmetric")
            elif mode == "maj_balanced":
                gm.clear_all(); build_self_written(train, gm, model)
                fn = fn_maj(model, gm, "balanced")
            elif mode == "maj_oracle":
                gm.clear_all(); build_oracle(train, gm)
                fn = fn_maj(model, gm, "asymmetric")
            tag = f"cv_{mode}_fold{i}_seed{seed}"
            d, acc, _ = eval_rows(test, fn, gm, desc=tag,
                                  audit_path=OUT / f"{tag}_audit.json" if mode != "stateless" else None)
            d["mode"] = mode; d["fold"] = i
            d.to_csv(OUT / f"{tag}.csv", index=False)
            print(f"  fold {i} {mode:16s}: {acc:.1f}%")


def task_anchor(df, gm, model, seed):
    print("\n=== TASK 4: leniency-anchor logging ===")
    OUT.mkdir(parents=True, exist_ok=True)
    train, test = split_by_question(df, seed=seed)
    gm.clear_all(); build_self_written(train, gm, model)
    # stateless reference to compute flips
    sd, _, _ = eval_rows(test, fn_stateless(model), gm, desc="anchor:stateless")
    base = {r.idx: r.predicted for r in sd.itertuples()}
    for mode in ("asymmetric", "balanced"):
        fn = fn_maj(model, gm, mode)
        d, acc, _ = eval_rows(test, fn, gm, desc=f"anchor:{mode}",
                              audit_path=OUT / f"anchor_{mode}_seed{seed}_audit.json")
        d["flip_vs_stateless"] = d.apply(
            lambda r: (r["predicted"] != base.get(r["idx"]))
            if pd.notna(r["predicted"]) and pd.notna(base.get(r["idx"])) else np.nan, axis=1)
        d.to_csv(OUT / f"anchor_{mode}_seed{seed}.csv", index=False)
        # false-pass tilt: among flips, fraction toward pass
        flips = d[d["flip_vs_stateless"] == True]
        tilt = (flips["predicted"] == True).mean() if len(flips) else float("nan")
        print(f"  {mode:11s}: {acc:.1f}%  flips={len(flips)}  toward_pass={tilt:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="all",
                    choices=["all", "split2x2", "controls", "cv", "anchor"])
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    df = pd.read_csv(DATA_PATH / "benchmark_df.csv")
    gm = GraphManager()
    tasks = ["split2x2", "controls", "cv", "anchor"] if args.task == "all" else [args.task]
    for t in tasks:
        {"split2x2": lambda: task_split2x2(df, gm, args.model, args.seed),
         "controls": lambda: task_controls(df, gm, args.model, args.seed),
         "cv": lambda: task_cv(df, gm, args.model, args.seed, args.folds),
         "anchor": lambda: task_anchor(df, gm, args.model, args.seed)}[t]()
    print("\nDone. Results in", OUT)


if __name__ == "__main__":
    main()
