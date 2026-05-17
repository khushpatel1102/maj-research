"""
reproduce_all.py -- one-command reproduction of every table and figure in
the MCTS-MAJ thesis from the raw per-sample CSVs in ``results/``.

This script does NOT re-run any LLM evaluation. It only re-derives the
aggregate numbers (accuracy, Wilson CIs, paired bootstrap CIs, McNemar
tests), the audit summary, and the figures from the per-sample CSVs that
are committed to the repository.

Usage
-----
    python reproduce_all.py

What it does
------------
1. Verifies every expected per-sample CSV exists.
2. Re-runs ``analyze_stats.py`` to regenerate the statistics tables.
3. Re-runs ``make_figures.py`` to regenerate all figures.
4. Summarizes the frozen-memory audit logs (PASS/FAIL per run).
5. Reprints every headline table used in Chapter 4.

Exit code is non-zero if any expected artifact is missing or any audit
log reports FAIL.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
FIGDIR = RESULTS / "figures"


# ---------------------------------------------------------------------------
# Expected artifacts
# ---------------------------------------------------------------------------
# Stage 2 leakage-free, seed 42 (the primary set).
STAGE2_SEED42 = {
    "stateless":            RESULTS / "leakage_free_stateless.csv",
    "maj":                  RESULTS / "leakage_free_maj.csv",
    "mcts_judge":           RESULTS / "leakage_free_mcts_judge.csv",
    "mcts_judge_memory":    RESULTS / "leakage_free_mcts_judge_memory.csv",
    "maj_oracle":           RESULTS / "lf_oracle_maj.csv",
    "mcts_mem_oracle":      RESULTS / "lf_oracle_mcts_judge_memory.csv",
    "maj_poison_10":        RESULTS / "lf_poisoned_10_maj.csv",
    "maj_poison_20":        RESULTS / "lf_poisoned_20_maj.csv",
    "maj_poison_50":        RESULTS / "lf_poisoned_50_maj.csv",
    "mcts_mem_poison_10":   RESULTS / "lf_poisoned_10_mcts_judge_memory.csv",
    "mcts_mem_poison_20":   RESULTS / "lf_poisoned_20_mcts_judge_memory.csv",
    "mcts_mem_poison_50":   RESULTS / "lf_poisoned_50_mcts_judge_memory.csv",
}

HARNESS = {
    "stability":      RESULTS / "harness_stability.csv",
    "label_flip":     RESULTS / "harness_label_flip.csv",
    "defense_off":    RESULTS / "harness_defense_off.csv",
    "defense_on":     RESULTS / "harness_defense_on.csv",
}

# Stage 1 (earlier, pre-leakage-fix) — retained for context only.
STAGE1 = {
    "results_stateless":          RESULTS / "results_stateless.csv",
    "results_maj":                RESULTS / "results_maj.csv",
    "results_mcts_judge":         RESULTS / "results_mcts_judge.csv",
    "results_mcts_judge_memory":  RESULTS / "results_mcts_judge_memory.csv",
    "results_mcts_retrieval":     RESULTS / "results_mcts_retrieval.csv",
    "results_full_mcts":          RESULTS / "results_full_mcts.csv",
}


# ---------------------------------------------------------------------------
# Statistical primitives (kept self-contained so this script is standalone)
# ---------------------------------------------------------------------------
def wilson_ci(k: int, n: int, alpha: float = 0.05):
    if n == 0:
        return 0.0, 1.0
    z = sps.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return max(0.0, (centre - margin) / denom), min(1.0, (centre + margin) / denom)


def paired_bootstrap_ci(a, b, n_boot=10_000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    d = a.astype(float) - b.astype(float)
    n = len(d)
    if n == 0:
        return 0.0, 0.0, 0.0
    idx = rng.integers(0, n, size=(n_boot, n))
    bm = d[idx].mean(axis=1)
    lo, hi = np.percentile(bm, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(d.mean()), float(lo), float(hi)


def mcnemar_p(a, b):
    a = a.astype(bool); b = b.astype(bool)
    b01 = int(((~a) & b).sum())
    b10 = int((a & (~b)).sum())
    n = b01 + b10
    if n == 0:
        return 1.0
    return float(sps.binomtest(min(b01, b10), n, 0.5).pvalue)


def acc(df):
    df = df.dropna(subset=["correct"])
    n = len(df)
    k = int(df["correct"].astype(bool).sum())
    lo, hi = wilson_ci(k, n)
    return k, n, (k / n if n else 0.0), lo, hi


def load(path):
    return pd.read_csv(path) if path.exists() else None


def align(a, b):
    m = a[["idx", "correct"]].merge(b[["idx", "correct"]], on="idx",
                                    suffixes=("_a", "_b")).dropna()
    return m["correct_a"].astype(int).values, m["correct_b"].astype(int).values


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------
def step_verify_artifacts() -> bool:
    print("=" * 70)
    print("STEP 1 / 5  Verifying per-sample CSVs")
    print("=" * 70)
    ok = True
    for group, files in (("Stage 2 (seed 42)", STAGE2_SEED42),
                         ("Reliability harness", HARNESS),
                         ("Stage 1 (legacy)", STAGE1)):
        print(f"\n[{group}]")
        for name, path in files.items():
            exists = path.exists()
            ok &= exists or group == "Stage 1 (legacy)"  # Stage 1 optional
            print(f"  {'OK ' if exists else 'MISSING'}  {name:<28} {path.name}")
    return ok


def step_run_subprocess(script: str, label: str):
    print("\n" + "=" * 70)
    print(label)
    print("=" * 70)
    r = subprocess.run([sys.executable, str(ROOT / script)],
                       capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        raise RuntimeError(f"{script} exited with code {r.returncode}")


def step_audit_summary() -> bool:
    print("\n" + "=" * 70)
    print("STEP 4 / 5  Frozen-memory audit summary")
    print("=" * 70)
    logs = sorted(RESULTS.glob("*_audit.json"))
    if not logs:
        print("  FAIL: no audit logs found in results/.")
        return False
    all_pass = True
    for log in logs:
        with open(log) as f:
            d = json.load(f)
        diff = d.get("diff", {})
        verdict = "PASS" if diff.get("identical") else "FAIL"
        all_pass &= diff.get("identical", False)
        print(f"  {verdict}  {log.name:<48} "
              f"nodes {diff.get('before_total_nodes', '?')}->"
              f"{diff.get('after_total_nodes', '?')}  "
              f"edges {diff.get('before_total_edges', '?')}->"
              f"{diff.get('after_total_edges', '?')}")

    # Cross-check: every Stage-2 CSV should have a matching audit JSON.
    print("\n  Audit coverage of STAGE2_SEED42 CSVs:")
    missing = []
    for name, csv in STAGE2_SEED42.items():
        audit = csv.with_name(csv.stem + "_audit.json")
        if audit.exists():
            print(f"    OK       {csv.name:<48} -> {audit.name}")
        else:
            print(f"    MISSING  {csv.name:<48} (no {audit.name})")
            missing.append(name)
    if missing:
        print(f"  WARNING: {len(missing)} CSV(s) have no audit log: {missing}")
        # Coverage gap is a warning, not a hard fail — flag but don't crash.
    return all_pass


def step_reprint_tables():
    print("\n" + "=" * 70)
    print("STEP 5 / 5  Headline tables (re-derived from CSVs)")
    print("=" * 70)

    # --- Table: Stage 2 per-mode accuracy with Wilson CIs ---
    print("\n[Table] Stage 2 leakage-free per-mode accuracy (seed 42, 80 samples)")
    print(f"{'Mode':<24} {'Acc':>7}  {'Wilson 95% CI':>20}  {'n':>4}")
    print("-" * 60)
    for name, path in STAGE2_SEED42.items():
        df = load(path)
        if df is None:
            continue
        k, n, p, lo, hi = acc(df)
        print(f"{name:<24} {p*100:>6.1f}% [{lo*100:>5.1f}%, {hi*100:>5.1f}%]  {n:>4}")

    # --- Table: key paired comparisons ---
    pairs = [
        ("maj", "stateless"),
        ("mcts_judge", "stateless"),
        ("mcts_judge_memory", "stateless"),
        ("mcts_judge_memory", "maj"),
        ("maj_oracle", "maj"),
        ("mcts_mem_oracle", "mcts_judge_memory"),
        ("mcts_mem_poison_10", "mcts_mem_oracle"),
        ("mcts_mem_poison_50", "mcts_mem_oracle"),
        ("defense_on", "defense_off"),
    ]
    # add the defense pair from HARNESS
    csv_for = dict(STAGE2_SEED42)
    csv_for["defense_off"] = HARNESS["defense_off"]
    csv_for["defense_on"] = HARNESS["defense_on"]

    print("\n[Table] Paired comparisons (same items only)")
    print(f"{'A':<22} {'B':<22} {'Δ (bootstrap 95% CI)':>26}  {'McNemar p':>10}  Sig")
    print("-" * 92)
    for a_name, b_name in pairs:
        da = load(csv_for.get(a_name))
        db = load(csv_for.get(b_name))
        if da is None or db is None:
            continue
        x, y = align(da, db)
        d, lo, hi = paired_bootstrap_ci(x, y)
        p = mcnemar_p(x, y)
        sig = "yes" if (p < 0.05 and (lo > 0 or hi < 0)) else "no"
        print(f"{a_name:<22} {b_name:<22} "
              f"{d*100:>+7.1f}pp [{lo*100:>+6.1f}, {hi*100:>+6.1f}]  "
              f"{p:>10.3f}  {sig}")

    # --- Table: reliability harness ---
    print("\n[Table] Reliability harness")
    stab = load(HARNESS["stability"])
    flip = load(HARNESS["label_flip"])
    doff = load(HARNESS["defense_off"])
    don = load(HARNESS["defense_on"])
    if stab is not None and "mode" in stab:
        print("  Stochastic stability (verdict agreement on repeated calls):")
        for m in stab["mode"].unique():
            v = stab[stab["mode"] == m]["agree"].mean() * 100
            print(f"    {m:<22} {v:>5.1f}%")
    if flip is not None and "mode" in flip:
        col = "correctly_flipped" if "correctly_flipped" in flip else "correct"
        print("  Label-flip discrimination (verdict correctly flips on inversion):")
        for m in flip["mode"].unique():
            v = flip[flip["mode"] == m][col].mean() * 100
            print(f"    {m:<22} {v:>5.1f}%")
    if doff is not None and don is not None:
        _, _, p_off, _, _ = acc(doff)
        _, _, p_on, _, _ = acc(don)
        print("  Defense filter on 50% poisoned memory:")
        print(f"    no defense             {p_off*100:>5.1f}%")
        print(f"    similarity filter      {p_on*100:>5.1f}%")
        print(f"    recovery               {(p_on - p_off)*100:>+5.1f}pp")

    # --- Multi-seed summary: any lf_*_seed*.csv files ---
    seed_files = sorted(RESULTS.glob("lf_*_seed*.csv"))
    if seed_files:
        print("\n[Table] Multi-seed leakage-free results (cross-seed comparison)")
        print(f"{'File':<48} {'Acc':>7}  {'Wilson 95% CI':>20}")
        print("-" * 80)
        for path in seed_files:
            df = load(path)
            if df is None:
                continue
            k, n, p, lo, hi = acc(df)
            print(f"{path.name:<48} {p*100:>6.1f}% [{lo*100:>5.1f}%, {hi*100:>5.1f}%]")
        # Compact cross-seed comparison for the modes that were run across seeds.
        print("\n  Cross-seed comparison (stateless + MAJ):")
        for cond, label in (("no_memory_stateless", "stateless (no memory)"),
                            ("self_written_maj", "maj (self-written)"),
                            ("oracle_maj", "maj (oracle)")):
            seed42_path = {
                "no_memory_stateless": RESULTS / "leakage_free_stateless.csv",
                "self_written_maj": RESULTS / "leakage_free_maj.csv",
                "oracle_maj": RESULTS / "lf_oracle_maj.csv",
            }[cond]
            cells = []
            for seed_label, p in [("42", seed42_path)] + [
                (str(s), RESULTS / f"lf_{cond}_seed{s}.csv") for s in (123, 7)
            ]:
                df = load(p)
                if df is None:
                    cells.append(f"seed{seed_label}=--")
                    continue
                _, _, acc_v, _, _ = acc(df)
                cells.append(f"seed{seed_label}={acc_v*100:.1f}%")
            print(f"    {label:<24} " + "  ".join(cells))
    else:
        print("\n(multi-seed results not committed yet — table will appear here)")


# ---------------------------------------------------------------------------
def main():
    artifacts_ok = step_verify_artifacts()
    step_run_subprocess("analyze_stats.py",
                        "STEP 2 / 5  Regenerating statistics (analyze_stats.py)")
    step_run_subprocess("make_figures.py",
                        "STEP 3 / 5  Regenerating figures (make_figures.py)")
    audit_ok = step_audit_summary()
    step_reprint_tables()

    print("\n" + "=" * 70)
    print("REPRODUCTION COMPLETE")
    print("=" * 70)
    print(f"  Statistics : results/stats_table.csv, stats_pairs.csv")
    print(f"  Figures    : {FIGDIR}/ ({len(list(FIGDIR.glob('*.png')))} PNGs)")
    print(f"  Artifacts  : {'all present' if artifacts_ok else 'SOME MISSING (see step 1)'}")
    print(f"  Audit      : {'all PASS' if audit_ok else 'SOME FAIL (see step 4)'}")

    if not artifacts_ok or not audit_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
