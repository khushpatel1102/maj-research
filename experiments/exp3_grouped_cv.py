"""
Experiment 3 (Bader review, task 3): grouped cross-validation for reliability.

5-fold grouped CV at the QUESTION level (80 questions -> 5 folds of 16 test
questions; memory built on the other 64 questions = 128 rows). Both pass/fail
twins of a test question stay together in the test fold and never enter memory,
so each fold is leakage-free.

Primary modes only (NO MCTS): stateless, MAJ(self/asymmetric), MAJ(oracle),
MAJ(balanced/symmetric-thresholds).

Reviewer fixes applied:
* Self-written memory is built ONCE per fold and reused for asymmetric AND
  symmetric retrieval, so the asymmetric-vs-balanced contrast is a pure
  retrieval-policy effect, not LLM-build noise.
* Every fold/mode runs inside the frozen-memory audit; a fold whose audit
  fails is asserted out rather than silently pooled.
* Aggregation is a QUESTION-CLUSTER bootstrap (resample questions with
  replacement) so CIs respect the pass/fail pairing.

Usage:
  python experiments/exp3_grouped_cv.py --model gpt-4o --seed 42
  python experiments/exp3_grouped_cv.py --model gpt-4o-mini --seed 42 --folds 2 --limit-q 8
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp_common as E
from graph_manager import GraphManager
from exp4_leniency_anchor import format_context  # reuse threshold-parametrized formatter


def grouped_folds(questions, n_folds, seed):
    qs = np.array(sorted(questions))
    rng = np.random.RandomState(seed)
    rng.shuffle(qs)
    return [set(f) for f in np.array_split(qs, n_folds)]


def make_threshold_maj_predictor(model, gm, pos_thr, neg_thr):
    import os
    from openai import OpenAI
    from prompts import build_judge_with_memory_prompt
    from models import JudgeResult, get_embedding
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def predict(s):
        emb = get_embedding(s["agent_output"])
        contr = gm.find_contrastive_attempts(emb, top_k=3)
        sim_issues = gm.find_similar_issues(emb, top_k=5)
        sem = gm.find_semantic_patterns(emb, top_k=3)
        ctx, counts = format_context(contr, sim_issues, sem,
                                     pos_thr=pos_thr, neg_thr=neg_thr)
        prompt = build_judge_with_memory_prompt(
            task=s["task"], agent_output=s["agent_output"],
            goal=E.EVALSBENCH_GOAL, memory_context=ctx)
        resp = client.responses.parse(model=model,
                                      input=[{"role": "user", "content": prompt}],
                                      text_format=JudgeResult, temperature=0)
        return resp.output_parsed.is_successful, {"n_pos_post": counts["n_pos"],
                                                  "n_neg_post": counts["n_neg"]}
    return predict


def cluster_bootstrap_delta(df_a, df_b, n_boot=10000, seed=0):
    """Paired question-cluster bootstrap on mode B minus mode A accuracy.
    Resamples QUESTIONS with replacement; both twins move together."""
    m = df_a.merge(df_b, on="idx", suffixes=("_a", "_b")).dropna(
        subset=["correct_a", "correct_b"])
    if not len(m):
        return float("nan"), (float("nan"), float("nan"))
    by_q = {q: g for q, g in m.groupby("question_a")}
    questions = list(by_q.keys())
    rng = np.random.RandomState(seed)
    point = m["correct_b"].mean() - m["correct_a"].mean()
    deltas = []
    for _ in range(n_boot):
        pick = rng.choice(len(questions), size=len(questions), replace=True)
        rows = pd.concat([by_q[questions[i]] for i in pick])
        deltas.append(rows["correct_b"].mean() - rows["correct_a"].mean())
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return point * 100, (lo * 100, hi * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--limit-q", type=int, default=None,
                    help="cap total questions for a cheap smoke run")
    args = ap.parse_args()

    df = E.load_benchmark()
    questions = list(dict.fromkeys(df["question"]))
    if args.limit_q:
        questions = questions[: args.limit_q]
        df = df[df["question"].isin(questions)].copy()
    folds = grouped_folds(questions, args.folds, args.seed)

    gm = GraphManager()
    # collect per-item rows per mode across folds
    rows_by_mode = {m: [] for m in ["stateless", "maj_self", "maj_oracle", "maj_balanced"]}

    for fi, test_q in enumerate(folds):
        train_df = df[~df["question"].isin(test_q)].copy()
        test_df = df[df["question"].isin(test_q)].copy()
        print(f"\n########## FOLD {fi+1}/{len(folds)}: "
              f"train={len(train_df)} test={len(test_df)} ##########")

        # --- stateless (no memory needed) ---
        gm.clear_all()
        sl = E.make_stateless_predictor(args.model)
        r, _ = E.audited_eval(test_df, sl, gm, desc=f"f{fi}-stateless")
        r["fold"] = fi; rows_by_mode["stateless"].append(r)

        # --- self-written memory, reused for asymmetric + symmetric ---
        gm.clear_all()
        E.build_self_written_memory(train_df, gm, args.model)
        asym = make_threshold_maj_predictor(args.model, gm, 0.85, 0.92)
        r, a = E.audited_eval(test_df, asym, gm, desc=f"f{fi}-maj_self")
        assert a["diff"]["identical"], f"fold {fi} maj_self mutated memory"
        r["fold"] = fi; rows_by_mode["maj_self"].append(r)

        sym = make_threshold_maj_predictor(args.model, gm, 0.85, 0.85)
        r, a = E.audited_eval(test_df, sym, gm, desc=f"f{fi}-maj_balanced")
        assert a["diff"]["identical"], f"fold {fi} maj_balanced mutated memory"
        r["fold"] = fi; rows_by_mode["maj_balanced"].append(r)

        # --- oracle memory ---
        gm.clear_all()
        E.build_oracle_memory(train_df, gm, flip_rate=0.0, seed=args.seed)
        orc = make_threshold_maj_predictor(args.model, gm, 0.85, 0.92)
        r, a = E.audited_eval(test_df, orc, gm, desc=f"f{fi}-maj_oracle")
        assert a["diff"]["identical"], f"fold {fi} maj_oracle mutated memory"
        r["fold"] = fi; rows_by_mode["maj_oracle"].append(r)

    # pool folds, write per-mode CSVs, compute pooled metrics
    pooled = {}
    for mode, parts in rows_by_mode.items():
        d = pd.concat(parts, ignore_index=True)
        d.to_csv(E.RESULTS_DIR / f"exp3_cv_{mode}_seed{args.seed}.csv", index=False)
        pooled[mode] = d

    print("\n================ GROUPED CV SUMMARY (pooled over folds) ================")
    print(f"{'mode':<14}{'acc':>7}{'bal':>7}{'passR':>8}{'failR':>8}{'n':>5}")
    out = {}
    for mode, d in pooled.items():
        cm = E.class_metrics(d)
        out[mode] = cm
        print(f"{mode:<14}{cm['acc']:>6.1f}%{cm['balanced_acc']:>6.1f}%"
              f"{cm['pass_recall']:>7.1f}%{cm['fail_recall']:>7.1f}%{cm['n']:>5}")

    print("\nPaired question-cluster bootstrap deltas vs stateless (95% CI):")
    base = pooled["stateless"]
    deltas = {}
    for mode in ["maj_self", "maj_oracle", "maj_balanced"]:
        pt, (lo, hi) = cluster_bootstrap_delta(base, pooled[mode], seed=args.seed)
        deltas[mode] = {"delta_pp": pt, "ci_lo": lo, "ci_hi": hi}
        sig = "" if (lo <= 0 <= hi) else "  *significant*"
        print(f"  {mode:<14} {pt:+5.1f}pp  [{lo:+5.1f}, {hi:+5.1f}]{sig}")
    # the key leniency contrast: balanced vs self
    pt, (lo, hi) = cluster_bootstrap_delta(pooled["maj_self"], pooled["maj_balanced"],
                                           seed=args.seed)
    print(f"  balanced - self: {pt:+5.1f}pp  [{lo:+5.1f}, {hi:+5.1f}]")
    deltas["balanced_minus_self"] = {"delta_pp": pt, "ci_lo": lo, "ci_hi": hi}

    (E.RESULTS_DIR / f"exp3_cv_summary_seed{args.seed}.json").write_text(
        json.dumps({"metrics": out, "deltas": deltas}, indent=2))


if __name__ == "__main__":
    main()
