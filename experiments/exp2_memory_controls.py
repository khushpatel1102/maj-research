"""
Memory-use controls: an answer-injection positive control plus negative controls.

Bracket the null from both sides. The audited result says memory does not help;
this experiment proves that is a property of the BENCHMARK (memory is not
informative here), not of the JUDGE being unable to use memory at all.

Positive control (upper bound):
  * cheating_oracle  -- inject the test item's OWN ground-truth-correct verdict
    as an exemplar. If the judge can use memory, accuracy must jump toward 100%.

Negative controls (must match the no-memory / stateless baseline):
  * random_context     -- inject a random unrelated past attempt.
  * label_shuffled     -- inject real similar exemplars but with labels permuted
                          (so the displayed pass/fail is uninformative).
  * irrelevant_samelen -- inject a topically irrelevant block padded to match the
                          cheating-oracle block's token length.

All arms route through build_judge_with_memory_prompt with the SAME template, so
each arm differs from MAJ only in the injected block (otherwise the negative
arms would measure an empty-context memory judge, not the stateless baseline).

Manipulation check: per item we log whether the cheating arm flips the verdict
relative to stateless, so a high cheating accuracy is attributable to memory use
rather than inherited from an already-correct stateless verdict.

Usage:
  python experiments/exp2_memory_controls.py --model gpt-4o --seed 42
  python experiments/exp2_memory_controls.py --model gpt-4o-mini --seed 42 --limit 6
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


def _exemplar_block(excerpt, passed, reasoning):
    """Format a single exemplar the same way the real memory context does."""
    head = "SUCCESSFUL EXAMPLES (similar responses that passed):" if passed \
        else "FAILED EXAMPLES (similar responses that failed - check if same issue applies):"
    why = "Why it passed:" if passed else "Why it failed:"
    return (
        "IMPORTANT: These are REFERENCE examples only. Each case is UNIQUE.\n"
        "Do NOT assume this case will have the same outcome as similar past cases.\n"
        "Judge THIS response on its OWN merits against the grading criteria.\n\n"
        f"{head}\n  1. [similarity: 99%] Response excerpt: {excerpt[:150]}...\n"
        f"     {why} {reasoning[:100]}..."
    )


def build_controls(df, model, seed):
    """Build the four control context-builders over the question-split test set."""
    train_df, test_df = E.split_by_question(df, seed=seed)
    rng = np.random.RandomState(seed)

    # pool of real attempts (from train) for random/irrelevant controls
    pool = [eval_row for _, row in train_df.iterrows()
            for eval_row in [E.evalsbench_to_maj(row)]]

    # precompute a long irrelevant filler from an unrelated train response
    filler_src = pool[0]["agent_output"] if pool else "Unrelated content. " * 50

    def cheating_oracle(s):
        # inject THIS item's own ground-truth verdict as a near-identical exemplar
        truth = s["expected"]
        block = _exemplar_block(s["agent_output"], truth,
                                f"Ground-truth verdict for this exact case is "
                                f"{'pass' if truth else 'fail'}.")
        return block, {"control": "cheating_oracle", "injected_label": truth}

    def random_context(s):
        pick = pool[rng.randint(len(pool))]
        block = _exemplar_block(pick["agent_output"], pick["expected"], "From an unrelated case.")
        return block, {"control": "random_context"}

    def label_shuffled(s):
        pick = pool[rng.randint(len(pool))]
        flipped = not pick["expected"]  # show the WRONG label -> uninformative
        block = _exemplar_block(pick["agent_output"], flipped, "Label intentionally permuted.")
        return block, {"control": "label_shuffled"}

    def irrelevant_samelen(s):
        # match cheating-oracle token length with topically irrelevant filler
        target = E.count_tokens(cheating_oracle(s)[0], model)
        words = filler_src.split()
        block, n = [], 0
        while E.count_tokens(" ".join(block), model) < target and words:
            block.append(words[n % len(words)]); n += 1
        return " ".join(block), {"control": "irrelevant_samelen",
                                  "target_tokens": target}

    return test_df, {
        "cheating_oracle": cheating_oracle,
        "random_context": random_context,
        "label_shuffled": label_shuffled,
        "irrelevant_samelen": irrelevant_samelen,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    df = E.load_benchmark()
    test_df, controls = build_controls(df, args.model, args.seed)
    if args.limit:
        qs = list(dict.fromkeys(test_df["question"]))[: args.limit]
        test_df = test_df[test_df["question"].isin(qs)].copy()

    gm = GraphManager()
    gm.clear_all()  # controls inject context directly; no graph memory needed

    # stateless baseline for the equivalence comparison + flip manipulation check
    print("\n--- stateless baseline ---")
    sl = E.make_stateless_predictor(args.model)
    sl_res, _ = E.audited_eval(test_df, sl, gm, desc="stateless")
    stateless_map = {r["idx"]: r["predicted"] for r in sl_res.to_dict("records")
                     if r["predicted"] in (True, False)}
    sl_res.to_csv(E.RESULTS_DIR / f"exp2_stateless_seed{args.seed}.csv", index=False)
    summary = {"stateless": E.class_metrics(sl_res)}

    for name, ctx_fn in controls.items():
        print(f"\n--- control: {name} ---")
        pred = E.make_custom_context_predictor(args.model, ctx_fn)
        res, _ = E.audited_eval(test_df, pred, gm, desc=name)
        # manipulation check: did this arm flip the verdict vs stateless?
        res["stateless_verdict"] = res["idx"].map(stateless_map)
        res["flipped"] = res.apply(
            lambda r: (r["stateless_verdict"] in (True, False)
                       and r["predicted"] in (True, False)
                       and r["stateless_verdict"] != r["predicted"]), axis=1)
        res.to_csv(E.RESULTS_DIR / f"exp2_{name}_seed{args.seed}.csv", index=False)
        cm = E.class_metrics(res)
        cm["flip_rate"] = float(res["flipped"].mean() * 100)
        summary[name] = cm

    print("\n================ MEMORY-USE CONTROLS SUMMARY ================")
    print(f"{'arm':<20}{'acc':>7}{'bal':>7}{'failR':>8}{'flip%':>7}{'n':>5}")
    base = summary["stateless"]["acc"]
    for k, m in summary.items():
        flip = m.get("flip_rate", float("nan"))
        print(f"{k:<20}{m['acc']:>6.1f}%{m['balanced_acc']:>6.1f}%"
              f"{m['fail_recall']:>7.1f}%{flip:>6.1f}%{m['n']:>5}")
    ch = summary.get("cheating_oracle", {}).get("acc", float("nan"))
    print(f"\nUpper bound (cheating_oracle) acc = {ch:.1f}%  vs stateless {base:.1f}%")
    print("Negative controls should sit near stateless; the cheating-vs-control gap "
          "proves the judge CAN use memory when it is informative.")
    (E.RESULTS_DIR / f"exp2_controls_summary_seed{args.seed}.json").write_text(
        json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
