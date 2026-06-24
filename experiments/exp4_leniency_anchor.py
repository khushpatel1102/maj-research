"""
Experiment 4 (Bader review, task 4): test the leniency-anchor explanation.

The paper claims memory acts as a *leniency anchor*: the asymmetric retrieval
thresholds (positive >= 0.85, negative >= 0.92) admit more passing than failing
exemplars, tilting the judge toward "pass" and degrading fail-class accuracy.

This script does two things the review demanded:

1. PER-ITEM INSTRUMENTATION on the published (asymmetric) MAJ path. For every
   test item it logs, from the SAME retrieval the verdict used: pre/post
   threshold pos & neg counts, the similarity of each retrieved exemplar, the
   top-1 retrieved label, the realized memory token count, the stateless
   verdict, the MAJ verdict, whether it flipped, and the flip direction.
   (Reviewer fix: logs must come from the same objects the verdict saw.)

2. THE CAUSAL MANIPULATION. The asymmetry lives in the THRESHOLDS, not the
   counts (negatives are already capped to len(pos)+1 in the published code).
   So the scientifically meaningful balanced condition is SYMMETRIC THRESHOLDS
   (both 0.85), holding retrieval fixed -- a clean one-variable change. We run
   asymmetric MAJ and symmetric MAJ on the identical test set and compare
   class-conditional accuracy. If symmetric lifts fail-class recall toward
   stateless, the leniency-anchor explanation is causally confirmed.

We do NOT hard-code 32.5/42.5 as a target: the stateless and asymmetric
class-conditional baselines are re-established here, and "no degradation to
remove" is allowed as a valid outcome.

Usage:
  python experiments/exp4_leniency_anchor.py --model gpt-4o --seed 42
  python experiments/exp4_leniency_anchor.py --model gpt-4o-mini --seed 42 --limit 6
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp_common as E
from graph_manager import GraphManager


# --- threshold-parametrized memory-context formatter -----------------------
# This mirrors judge.py:_format_memory_context exactly, but exposes the two
# thresholds so we can run asymmetric (0.85/0.92, published) vs symmetric
# (0.85/0.85) while holding the retrieved candidates fixed.

def format_context(contrastive, similar_issues, semantic_patterns,
                   pos_thr=0.85, neg_thr=0.92, issue_thr=0.90, balance_cap=True):
    parts = []
    positive = [a for a in contrastive["positive"] if a.get("score", 0) >= pos_thr]
    negative = [a for a in contrastive["negative"] if a.get("score", 0) >= neg_thr]
    issues = [i for i in similar_issues if i.get("score", 0) >= issue_thr]
    if balance_cap and len(negative) > len(positive) + 1:
        negative = negative[: len(positive) + 1]
    patterns = [p for p in (semantic_patterns or []) if p.get("avg_similarity", 0) >= 0.90]

    parts.append("IMPORTANT: These are REFERENCE examples only. Each case is UNIQUE.")
    parts.append("Do NOT assume this case will have the same outcome as similar past cases.")
    parts.append("Judge THIS response on its OWN merits against the grading criteria.\n")
    if positive:
        parts.append("SUCCESSFUL EXAMPLES (similar responses that passed):")
        for i, a in enumerate(positive, 1):
            parts.append(f"  {i}. [similarity: {a.get('score',0):.0%}] Response excerpt: {a['agent_output'][:150]}...")
            parts.append(f"     Why it passed: {a['reasoning'][:100]}...")
    if negative:
        parts.append("\nFAILED EXAMPLES (similar responses that failed - check if same issue applies):")
        for i, a in enumerate(negative, 1):
            parts.append(f"  {i}. [similarity: {a.get('score',0):.0%}] Response excerpt: {a['agent_output'][:150]}...")
            parts.append(f"     Why it failed: {a['reasoning'][:100]}...")
    if issues:
        parts.append("\nPAST ISSUES (only flag if SPECIFICALLY present in this response):")
        for i, issue in enumerate(issues, 1):
            parts.append(f"  {i}. [similarity: {issue.get('score',0):.0%}] {issue['description'][:100]}...")
    if patterns:
        parts.append("\nPATTERNS TO CHECK (warnings only - NOT automatic failures):")
        for i, p in enumerate(patterns, 1):
            parts.append(f"  {i}. {p['name']} [similarity: {p.get('avg_similarity',0):.0%}]")
    if not positive and not negative and not issues and not patterns:
        return "No highly similar past experiences found. Judge based on the criteria alone.", \
               dict(n_pos=0, n_neg=0)
    counts = dict(n_pos=len(positive), n_neg=len(negative))
    return "\n".join(parts), counts


def make_instrumented_predictor(model, gm, *, pos_thr, neg_thr, stateless_map):
    """One predictor that retrieves once, logs everything, and judges via the
    threshold-parametrized context. stateless_map[idx]=bool gives the paired
    stateless verdict for flip analysis."""
    import os as _os
    from openai import OpenAI
    from prompts import build_judge_with_memory_prompt
    from models import JudgeResult, get_embedding
    client = OpenAI(api_key=_os.getenv("OPENAI_API_KEY"))

    def predict(s):
        emb = get_embedding(s["agent_output"])
        contr = gm.find_contrastive_attempts(emb, top_k=3)
        sim_issues = gm.find_similar_issues(emb, top_k=5)
        sem = gm.find_semantic_patterns(emb, top_k=3)

        pos_all = contr.get("positive", [])
        neg_all = contr.get("negative", [])
        ctx, counts = format_context(contr, sim_issues, sem,
                                     pos_thr=pos_thr, neg_thr=neg_thr)
        prompt = build_judge_with_memory_prompt(
            task=s["task"], agent_output=s["agent_output"],
            goal=E.EVALSBENCH_GOAL, memory_context=ctx)
        resp = client.responses.parse(
            model=model, input=[{"role": "user", "content": prompt}],
            text_format=JudgeResult, temperature=0)  # determinism for flip analysis
        verdict = resp.output_parsed.is_successful

        retrieved = pos_all + neg_all
        top1 = max(retrieved, key=lambda a: a.get("score", 0)) if retrieved else None
        sv = stateless_map.get(s["idx"]) if stateless_map else None
        flipped = (sv is not None and sv != verdict)
        log = {
            "n_pos_pre": len(pos_all), "n_neg_pre": len(neg_all),
            "n_pos_post": counts["n_pos"], "n_neg_post": counts["n_neg"],
            "pos_sims": ";".join(f"{a.get('score',0):.3f}" for a in pos_all),
            "neg_sims": ";".join(f"{a.get('score',0):.3f}" for a in neg_all),
            "top1_label": (top1["is_successful"] if top1 else None),
            "top1_sim": (top1.get("score") if top1 else float("nan")),
            "mem_tokens": E.count_tokens(ctx, model),
            "stateless_verdict": sv,
            "flipped": flipped,
            "flip_dir": ("to_pass" if (flipped and verdict) else
                         "to_fail" if flipped else "none"),
        }
        return verdict, log
    return predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    df = E.load_benchmark()
    train_df, test_df = E.split_by_question(df, seed=args.seed)
    if args.limit:
        # Smoke mode: shrink BOTH the test set and the memory-build set, else the
        # build runs on the full 80-row train half regardless of --limit.
        qs = list(dict.fromkeys(test_df["question"]))[: args.limit]
        test_df = test_df[test_df["question"].isin(qs)].copy()
        tq = list(dict.fromkeys(train_df["question"]))[: args.limit]
        train_df = train_df[train_df["question"].isin(tq)].copy()

    gm = GraphManager()
    gm.clear_all()
    # ONE self-written memory, reused by asymmetric AND symmetric so the only
    # difference between the two conditions is the threshold (reviewer fix:
    # avoid comparing on two different LLM-built graphs).
    E.build_self_written_memory(train_df, gm, args.model)

    # 1) stateless baseline first, to anchor flip analysis + class-conditional gap
    print("\n--- stateless baseline ---")
    sl_pred = E.make_stateless_predictor(args.model)
    sl_res, _ = E.audited_eval(test_df, sl_pred, gm, desc="stateless")
    stateless_map = {r["idx"]: r["predicted"] for r in sl_res.to_dict("records")
                     if r["predicted"] in (True, False)}
    sl_res.to_csv(E.RESULTS_DIR / f"exp4_stateless_seed{args.seed}.csv", index=False)

    conditions = [
        ("asymmetric", 0.85, 0.92),   # published MAJ
        ("symmetric",  0.85, 0.85),   # the causal manipulation
    ]
    summary = {"stateless": E.class_metrics(sl_res)}
    for name, pt, nt in conditions:
        print(f"\n--- MAJ {name} (pos>={pt}, neg>={nt}) ---")
        pred = make_instrumented_predictor(args.model, gm, pos_thr=pt, neg_thr=nt,
                                           stateless_map=stateless_map)
        res, audit = E.audited_eval(test_df, pred, gm,
                                    audit_path=E.RESULTS_DIR / f"exp4_{name}_seed{args.seed}_audit.json",
                                    desc=f"maj-{name}")
        assert audit["diff"]["identical"], f"{name} mutated memory!"
        res.to_csv(E.RESULTS_DIR / f"exp4_{name}_seed{args.seed}.csv", index=False)
        summary[name] = E.class_metrics(res)
        # leniency signature: how many more passing than failing exemplars cleared threshold
        post = res.dropna(subset=["correct"])
        if "n_pos_post" in post:
            print(f"    avg post-threshold pos={post['n_pos_post'].mean():.2f} "
                  f"neg={post['n_neg_post'].mean():.2f}  "
                  f"flips_to_pass={int((res['flip_dir']=='to_pass').sum())} "
                  f"flips_to_fail={int((res['flip_dir']=='to_fail').sum())}")

    print("\n================ LENIENCY-ANCHOR SUMMARY ================")
    print(f"{'condition':<14}{'acc':>7}{'bal':>7}{'passR':>8}{'failR':>8}{'n':>5}")
    for k, m in summary.items():
        print(f"{k:<14}{m['acc']:>6.1f}%{m['balanced_acc']:>6.1f}%"
              f"{m['pass_recall']:>7.1f}%{m['fail_recall']:>7.1f}%{m['n']:>5}")
    a, s = summary.get("asymmetric"), summary.get("symmetric")
    if a and s:
        print(f"\nFail-class recall: asymmetric {a['fail_recall']:.1f}% -> "
              f"symmetric {s['fail_recall']:.1f}%  (Δ {s['fail_recall']-a['fail_recall']:+.1f}pp)")
        print("If Δ is positive and meaningful, symmetric thresholds remove the "
              "false-pass tilt -> leniency-anchor explanation causally supported.")
    import json
    (E.RESULTS_DIR / f"exp4_summary_seed{args.seed}.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
