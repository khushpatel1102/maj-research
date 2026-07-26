"""
The 2x2 leakage design: {row, question} split x {write-back, frozen} memory.

Crosses two binary factors that each correspond to ONE leakage channel:

                       | frozen memory        | write-back enabled
  ---------------------+----------------------+-----------------------
  question-level split | (A) clean / published| (C) write-back only
  row-level split      | (B) paired-item only | (D) both channels

Cell (A) is the leakage-free number from the paper (MAJ ~65.0%). Each other
cell re-opens exactly one or both channels, so the accuracy difference from (A)
attributes the conventional protocol's apparent +6-10pp gain to a specific
channel. This is the experiment that turns "the gain was leakage" from an
assertion into a measured decomposition.

Memory is self-written MAJ memory built on the train half of each split.

Usage:
  python experiments/exp1_2x2_leakage.py --model gpt-4o --seed 42
  python experiments/exp1_2x2_leakage.py --model gpt-4o-mini --seed 42   # cheap smoke
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp_common as E
from graph_manager import GraphManager


def make_maj_writeback_predictor(model, gm):
    """MAJ predictor that ALSO writes the just-judged item back into memory.

    This is the write-back leakage channel, reproduced deliberately. It runs
    OUTSIDE gm.freeze() (allow_writes=True) so the writes succeed; the audit on
    that arm is expected to show growth, which is the point.
    """
    from judge import judge_with_memory
    from exp_common import _commit_full_result

    def predict(s):
        r = judge_with_memory(task=s["task"], agent_output=s["agent_output"],
                              graph_manager=gm, goal=E.EVALSBENCH_GOAL, model=model)
        verdict = r["attempt"].is_successful
        # Write-back: commit this evaluation so later, related test items can see it.
        _commit_full_result(gm, r)
        return verdict, {"wrote_back": True}
    return predict


def make_maj_dose_predictor(model, gm, twin_outputs):
    """MAJ predictor that also logs leakage DOSE: was the test item's paired
    twin retrieved among the contrastive attempts, and at what similarity?

    ``twin_outputs`` maps a test item's agent_output -> its twin's agent_output,
    so we can check whether the retrieved attempts include the twin. This makes
    a null row-split effect interpretable (no effect vs no exposure)."""
    from judge import judge_with_memory
    from models import get_embedding

    def predict(s):
        r = judge_with_memory(task=s["task"], agent_output=s["agent_output"],
                              graph_manager=gm, goal=E.EVALSBENCH_GOAL, model=model)
        mu = r.get("memory_used", {})
        # dose: re-run the same retrieval the judge used and look for the twin
        emb = get_embedding(s["agent_output"])
        contr = gm.find_contrastive_attempts(emb, top_k=3)
        retrieved = (contr.get("positive", []) + contr.get("negative", []))
        twin = twin_outputs.get(s["agent_output"])
        twin_sim, twin_hit = float("nan"), False
        if twin is not None:
            for a in retrieved:
                if a.get("agent_output", "")[:80] == twin[:80]:
                    twin_hit = True
                    twin_sim = a.get("score", float("nan"))
                    break
        top1_label = retrieved[0]["is_successful"] if retrieved else None
        return r["attempt"].is_successful, {
            "pos_retrieved": mu.get("positive_examples"),
            "neg_retrieved": mu.get("negative_examples"),
            "twin_in_memory": twin is not None,
            "twin_retrieved": twin_hit,
            "twin_similarity": twin_sim,
            "top1_retrieved_label": top1_label,
        }
    return predict


def _build_twin_map(df):
    """Map each row's agent_output to its paired-twin row's agent_output."""
    out = {}
    for _q, grp in df.groupby("question"):
        rows = list(grp.itertuples())
        if len(rows) == 2:
            a, b = rows
            out[a.response] = b.response
            out[b.response] = a.response
    return out


def run_cell(name, split_fn, writeback, df, model, seed, results_dir):
    train_df, test_df = split_fn(df, train_ratio=0.5, seed=seed)
    gm = GraphManager()
    gm.clear_all()
    print(f"\n=== CELL {name}: split={split_fn.__name__} writeback={writeback} ===")
    print(f"    train={len(train_df)} test={len(test_df)}")

    # Memory must be built from train rows only. For the write-back arm the
    # memory still STARTS as the train-only graph; write-back then grows it
    # during evaluation.
    E.build_self_written_memory(train_df, gm, model)

    tag = f"exp1_{name}"
    audit_path = results_dir / f"{tag}_seed{seed}_audit.json"
    twin_map = _build_twin_map(df)  # full-df twins so we can flag in-memory twins
    if writeback:
        predict = make_maj_writeback_predictor(model, gm)
        res, audit = E.audited_eval(test_df, predict, gm, allow_writes=True,
                                    audit_path=audit_path, desc=f"[{name}]")
    else:
        predict = make_maj_dose_predictor(model, gm, twin_map)
        res, audit = E.audited_eval(test_df, predict, gm, allow_writes=False,
                                    audit_path=audit_path, desc=f"[{name}]")

    cm = E.class_metrics(res)
    res.to_csv(results_dir / f"{tag}_seed{seed}.csv", index=False)
    grew = not audit["diff"]["identical"]
    # Frozen cells must be byte-identical before/after; flag loudly otherwise.
    if not writeback and grew:
        print(f"    !! AUDIT VIOLATION: frozen cell {name} mutated memory; "
              f"node_delta={audit['diff']['node_delta']}")
    print(f"    -> acc={cm['acc']:.1f}%  bal_acc={cm['balanced_acc']:.1f}%  "
          f"pass_recall={cm['pass_recall']:.1f}%  fail_recall={cm['fail_recall']:.1f}%  "
          f"n={cm['n']}  memory_grew={grew}")
    return {"cell": name, "split": split_fn.__name__, "writeback": writeback,
            "memory_grew": grew, **cm}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap test rows per cell for a cheap smoke run")
    args = ap.parse_args()

    df = E.load_benchmark()
    if args.limit:
        # keep it paired: cap by questions, not rows, so smoke runs stay valid
        pass

    cells = [
        ("A_qsplit_frozen",    E.split_by_question, False),
        ("B_rowsplit_frozen",  E.split_by_row,      False),
        ("C_qsplit_writeback", E.split_by_question, True),
        ("D_rowsplit_writeback", E.split_by_row,    True),
    ]

    summary = []
    for name, split_fn, wb in cells:
        sub = df
        if args.limit:
            # deterministic small slice for smoke: first `limit` questions
            qs = list(dict.fromkeys(df["question"]))[: max(4, args.limit)]
            sub = df[df["question"].isin(qs)].copy()
        summary.append(run_cell(name, split_fn, wb, sub, args.model, args.seed, E.RESULTS_DIR))

    print("\n================ 2x2 SUMMARY ================")
    print(f"{'cell':<22}{'acc':>7}{'bal':>7}{'passR':>7}{'failR':>7}{'n':>5}{'grew':>7}")
    base = next((r for r in summary if r["cell"] == "A_qsplit_frozen"), None)
    for r in summary:
        print(f"{r['cell']:<22}{r['acc']:>6.1f}%{r['balanced_acc']:>6.1f}%"
              f"{r['pass_recall']:>6.1f}%{r['fail_recall']:>6.1f}%{r['n']:>5}"
              f"{str(r['memory_grew']):>7}")
    if base:
        # Self-test: the clean cell (A) should reproduce the published MAJ number
        # (only meaningful on a full, non-limited gpt-4o run).
        if not args.limit and args.model.startswith("gpt-4o") and "mini" not in args.model:
            if abs(base["acc"] - 65.0) > 6.0:
                print(f"\n[warn] clean cell A acc={base['acc']:.1f}% deviates from "
                      f"published 65.0% by >6pp; check reproduction.")
        print("\nDelta vs clean cell (A) -- both raw and balanced acc "
              "(balanced neutralizes the row-split 44/36 base-rate shift):")
        for r in summary:
            if r["cell"] != "A_qsplit_frozen":
                print(f"  {r['cell']:<22} raw {r['acc']-base['acc']:+5.1f}pp   "
                      f"bal {r['balanced_acc']-base['balanced_acc']:+5.1f}pp")
    import json
    (E.RESULTS_DIR / f"exp1_2x2_summary_seed{args.seed}.json").write_text(
        json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
