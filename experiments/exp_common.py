"""
Shared infrastructure for the revision experiments.

Everything in this module is intentionally small, deterministic, and reused by
every experiment script so the four experiments cannot drift apart in how they
split data, build memory, run the audited eval loop, or write per-sample CSVs.

Design notes
------------
* All randomness flows through an explicit ``seed`` so any run is reproducible.
* Memory writes happen ONLY during build_*; evaluation always runs inside
  ``gm.freeze()`` and is bracketed by snapshot/diff, exactly like the published
  ``benchmark_leakage_free.py``. The ONE exception is the 2x2 write-back arm,
  which deliberately re-enables writes to *measure* write-back leakage; that arm
  is clearly flagged and never used for any "clean" number.
* Transient API errors are retried; persistent failures are recorded with a NaN
  verdict and excluded from accuracy (never scored as wrong) -- the correction
  that turned the spurious "poisoning collapse" into a flat curve.
"""

import os
import sys
import time
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# --- make src importable and load credentials exactly once -----------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

DATA_PATH = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

EVALSBENCH_GOAL = (
    "Check if the response contains points mentioned from the grading notes "
    "and return 'pass' or 'fail'."
)

# ---------------------------------------------------------------------------
# Retry wrapper (copied semantics from benchmark_leakage_free.py)
# ---------------------------------------------------------------------------

def _is_transient(exc: Exception) -> bool:
    msg = str(exc).lower()
    name = type(exc).__name__.lower()
    keys = ("connection", "timeout", "timed out", "rate limit", "ratelimit",
            "503", "502", "504", "apiconnection", "remoteprotocol",
            "remotedisconnected", "incomplete read", "overloaded")
    return any(k in msg or k in name for k in keys)


def call_with_retries(fn, *, max_attempts=5, base=2.0, max_sleep=30.0):
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if not _is_transient(exc) or attempt == max_attempts:
                raise
            sleep_s = min(max_sleep, base ** attempt) + random.uniform(0, 0.5)
            print(f"  [retry {attempt}/{max_attempts-1}] {type(exc).__name__}: "
                  f"{exc}; sleeping {sleep_s:.1f}s")
            time.sleep(sleep_s)
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Data + splits
# ---------------------------------------------------------------------------

def load_benchmark() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH / "benchmark_df.csv")


def evalsbench_to_maj(row) -> dict:
    return {
        "task": f"grading_notes: {row['grading_notes']}",
        "agent_output": row["response"],
        "expected": row["target"] == "pass",
        "topic": row["topic"],
        "question": row["question"],
    }


def split_by_question(df, train_ratio=0.5, seed=42):
    """Question-level split: both pass/fail versions of a question stay together.
    This is the leakage-FREE split."""
    questions = df["question"].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(questions)
    k = int(len(questions) * train_ratio)
    train_q, test_q = set(questions[:k]), set(questions[k:])
    return df[df["question"].isin(train_q)].copy(), df[df["question"].isin(test_q)].copy()


def split_by_row(df, train_ratio=0.5, seed=42):
    """Row-level split: pass/fail twins of a question can land on opposite sides.
    This is the LEAKY split -- used only by the 2x2 experiment to quantify the
    paired-item leakage channel."""
    idx = np.array(df.index)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    k = int(len(idx) * train_ratio)
    train_idx, test_idx = set(idx[:k]), set(idx[k:])
    return df.loc[df.index.isin(train_idx)].copy(), df.loc[df.index.isin(test_idx)].copy()


# ---------------------------------------------------------------------------
# Memory builders (thin wrappers around the project code)
# ---------------------------------------------------------------------------

def build_self_written_memory(train_df, gm, model):
    from judge import judge_with_memory
    from tqdm import tqdm
    for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="build self"):
        s = evalsbench_to_maj(row)
        try:
            r = call_with_retries(lambda: judge_with_memory(
                task=s["task"], agent_output=s["agent_output"],
                graph_manager=gm, goal=EVALSBENCH_GOAL, model=model))
            _commit_full_result(gm, r)
        except Exception as e:  # noqa: BLE001
            print(f"  build error idx={idx}: {e}")


def build_oracle_memory(train_df, gm, flip_rate=0.0, seed=42):
    """Oracle (ground-truth) memory, optionally poisoned by ``flip_rate``."""
    from models import Policy, Attempt
    from tqdm import tqdm
    rng = np.random.RandomState(seed)
    flip = rng.random(len(train_df)) < flip_rate
    for i, (idx, row) in enumerate(tqdm(list(train_df.iterrows()),
                                        total=len(train_df), desc="build oracle")):
        s = evalsbench_to_maj(row)
        label = s["expected"]
        if flip[i]:
            label = not label
        policy = Policy(description=s["task"]).with_embedding()
        attempt = Attempt(agent_output=s["agent_output"], is_successful=label,
                          reasoning=f"Oracle: ground truth label is "
                                    f"{'pass' if label else 'fail'}").with_embedding()
        gm.create_policy(policy)
        gm.create_attempt(attempt)
        gm.link_attempt_satisfies_policy(attempt.id, policy.id)


def _commit_full_result(gm, r):
    """Commit a full judge_with_memory() result to the graph (5-node schema)."""
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
    for i, semantic in enumerate(r.get("semantics", [])):
        rels = r.get("semantic_relationships", [])
        if i < len(rels) and rels[i].get("is_new", True):
            gm.get_or_create_semantic(semantic)
        if i < len(rels):
            gm.link_issue_abstracts_to_semantic(rels[i]["from_id"], rels[i]["to_id"])


# ---------------------------------------------------------------------------
# Audited evaluation loop (one row -> one prediction)
# ---------------------------------------------------------------------------

def audited_eval(test_df, predict_fn, gm, *, allow_writes=False, audit_path=None,
                 desc="eval"):
    """
    Run ``predict_fn(sample) -> (predicted_bool, extra_dict)`` over every test row.

    Unless ``allow_writes`` is True the run executes inside ``gm.freeze()`` and is
    bracketed by snapshot/diff; the audit result is written to ``audit_path``.

    ``allow_writes=True`` is ONLY for the write-back-leakage arm of the 2x2
    experiment, where predict_fn deliberately writes the just-judged item back.

    Returns (results_df, audit_dict).
    """
    from graph_manager import GraphManager
    from tqdm import tqdm

    snap_before = gm.snapshot()
    rows = []

    from graph_manager import FrozenMemoryViolation

    def _loop():
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc=desc):
            s = evalsbench_to_maj(row)
            s["idx"] = idx  # predictors may key flip/dose logs on the row index
            t0 = time.time()
            try:
                predicted, extra = call_with_retries(lambda: predict_fn(s))
                dt = time.time() - t0
                rows.append({"idx": idx, "topic": s["topic"], "question": s["question"],
                             "expected": s["expected"], "predicted": predicted,
                             "correct": (predicted == s["expected"]),
                             "latency_s": round(dt, 2), **(extra or {})})
            except FrozenMemoryViolation:
                # An attempted write during a frozen run is a protocol
                # violation, not an item error: abort the entire run.
                raise
            except Exception as e:  # noqa: BLE001
                dt = time.time() - t0
                print(f"  ERROR idx={idx}: {e}")
                rows.append({"idx": idx, "topic": s["topic"], "question": s["question"],
                             "expected": s["expected"], "predicted": np.nan,
                             "correct": np.nan, "latency_s": round(dt, 2)})

    if allow_writes:
        _loop()
    else:
        with gm.freeze():
            _loop()

    snap_after = gm.snapshot()
    diff = GraphManager.diff_snapshots(snap_before, snap_after)
    audit = {"before": snap_before, "after": snap_after, "diff": diff,
             "allow_writes": allow_writes}
    status = "PASS" if diff["identical"] else "FAIL"
    if allow_writes:
        status = "WRITE-BACK ARM (writes expected)"
    print(f"  [audit] {status}: {snap_before['total_nodes']}->{snap_after['total_nodes']} nodes, "
          f"fp {snap_before['fingerprint'][:10]} -> {snap_after['fingerprint'][:10]}")
    if audit_path is not None:
        Path(audit_path).write_text(json.dumps(audit, indent=2))

    df = pd.DataFrame(rows)
    return df, audit


def accuracy_of(df) -> tuple[float, int]:
    completed = df.dropna(subset=["correct"])
    n = len(completed)
    acc = (completed["correct"].mean() * 100) if n else 0.0
    return acc, n


# ---------------------------------------------------------------------------
# Predictors
# ---------------------------------------------------------------------------

def make_stateless_predictor(model):
    from mcts_pipeline import run_stateless
    def predict(s):
        r = run_stateless(s["task"], s["agent_output"], goal=EVALSBENCH_GOAL, model=model)
        return r["attempt"].is_successful, {}
    return predict


def make_maj_predictor(model, gm):
    from judge import judge_with_memory
    def predict(s):
        r = judge_with_memory(task=s["task"], agent_output=s["agent_output"],
                              graph_manager=gm, goal=EVALSBENCH_GOAL, model=model)
        mu = r.get("memory_used", {})
        return r["attempt"].is_successful, {
            "pos_retrieved": mu.get("positive_examples"),
            "neg_retrieved": mu.get("negative_examples"),
        }
    return predict


# ---------------------------------------------------------------------------
# Class-conditional metrics: the row split is 44/36, not 40/40, so raw accuracy
# across split types is confounded -- report balanced accuracy and per-class recall
# ---------------------------------------------------------------------------

def class_metrics(df) -> dict:
    """Return overall acc, balanced acc, and per-class recall over completed rows."""
    c = df.dropna(subset=["correct"])
    out = {"n": len(c), "acc": (c["correct"].mean() * 100) if len(c) else 0.0}
    for label, key in ((True, "pass"), (False, "fail")):
        sub = c[c["expected"] == label]
        out[f"{key}_n"] = len(sub)
        out[f"{key}_recall"] = (sub["correct"].mean() * 100) if len(sub) else float("nan")
    pr, fr = out["pass_recall"], out["fail_recall"]
    out["balanced_acc"] = (pr + fr) / 2 if (pr == pr and fr == fr) else float("nan")
    return out


# ---------------------------------------------------------------------------
# Token counting (log the realized memory token count, same encoder
# as the judge model, on the EXACT memory_context string the verdict used)
# ---------------------------------------------------------------------------

def count_tokens(text, model="gpt-4o"):
    try:
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("o200k_base")
        return len(enc.encode(text or ""))
    except Exception:
        return len((text or "").split())  # fallback: whitespace tokens


# ---------------------------------------------------------------------------
# Custom-context predictor: feeds an ARBITRARY memory_context string straight
# into the published memory prompt. This is the interface-faithful way to run
# the exp2 controls (cheating-oracle / random / label-shuffled / irrelevant)
# and the exp4 symmetric-threshold variant, WITHOUT touching judge.py.
# ---------------------------------------------------------------------------

def make_custom_context_predictor(model, build_context_fn):
    """build_context_fn(sample) -> (memory_context_str, extra_log_dict).

    Routes through build_judge_with_memory_prompt so the prompt TEMPLATE is
    identical to MAJ; only the memory block content changes, so a control's only
    difference from MAJ is the injected block.
    """
    import os
    from openai import OpenAI
    from prompts import build_judge_with_memory_prompt
    from models import JudgeResult
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def predict(s):
        memory_context, extra = build_context_fn(s)
        prompt = build_judge_with_memory_prompt(
            task=s["task"], agent_output=s["agent_output"],
            goal=EVALSBENCH_GOAL, memory_context=memory_context)
        resp = client.responses.parse(
            model=model, input=[{"role": "user", "content": prompt}],
            text_format=JudgeResult, temperature=0)  # determinism for flip analysis
        verdict = resp.output_parsed.is_successful
        log = {"mem_tokens": count_tokens(memory_context, model), **(extra or {})}
        return verdict, log
    return predict

