"""
Statistical analysis of MCTS-MAJ benchmark results.

Computes the rigorous statistics requested by the supervisor for every
mode and condition:

  * Wilson 95 percent confidence intervals on accuracy.
  * Paired bootstrap 95 percent confidence intervals on mode-vs-mode deltas.
  * McNemar's exact test on paired binary outcomes.

Operates on per-sample CSVs in ``results/``. No new evaluations are run.

Usage
-----
    python analyze_stats.py
    python analyze_stats.py --output results/stats_summary.md

Outputs
-------
* ``results/stats_summary.md`` -- human-readable Markdown report.
* ``results/stats_table.csv`` -- machine-readable per-mode statistics.
* ``results/stats_pairs.csv`` -- machine-readable pairwise comparisons.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats as sps

RESULTS = Path("results")


# ---------------------------------------------------------------------------
# Statistical primitives
# ---------------------------------------------------------------------------
def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Two-sided Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 1.0
    z = sps.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    lo = (centre - margin) / denom
    hi = (centre + margin) / denom
    return max(0.0, lo), min(1.0, hi)


def paired_bootstrap_ci(
    a: np.ndarray, b: np.ndarray, n_boot: int = 10_000,
    alpha: float = 0.05, seed: int = 42,
) -> tuple[float, float, float]:
    """
    Bootstrap 95 percent CI on the mean paired difference (a - b).

    Both arrays must be aligned on the same items (same length, same order).
    Returns (delta, lo, hi) as proportion differences.
    """
    assert a.shape == b.shape, "paired arrays must align"
    rng = np.random.default_rng(seed)
    diffs = a.astype(float) - b.astype(float)
    n = len(diffs)
    if n == 0:
        return 0.0, 0.0, 0.0
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(diffs.mean()), float(lo), float(hi)


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> dict:
    """
    Exact McNemar test on paired binary outcomes.

    a, b are 0/1 correctness arrays for two judges on the same items.
    Returns the contingency cells and the two-sided exact p-value.
    """
    a = a.astype(bool)
    b = b.astype(bool)
    b01 = int(((~a) & b).sum())  # b correct, a wrong
    b10 = int((a & (~b)).sum())  # a correct, b wrong
    n_disc = b01 + b10
    if n_disc == 0:
        p = 1.0
    else:
        # Exact two-sided binomial test, p = 0.5 under H0.
        p = float(sps.binomtest(min(b01, b10), n_disc, 0.5).pvalue)
    return {"b01": b01, "b10": b10, "n_disc": n_disc, "p": p}


# ---------------------------------------------------------------------------
# Per-mode summary
# ---------------------------------------------------------------------------
def summarize(name: str, df: pd.DataFrame) -> dict:
    df = df.dropna(subset=["correct"]).copy()
    n = len(df)
    k = int(df["correct"].astype(bool).sum())
    p = k / n if n else 0.0
    lo, hi = wilson_ci(k, n)
    mean_lat = float(df["latency_s"].mean()) if "latency_s" in df else float("nan")
    return {
        "name": name,
        "n": n,
        "correct": k,
        "accuracy": p,
        "wilson_lo": lo,
        "wilson_hi": hi,
        "wilson_str": f"{p:.3f} [{lo:.3f}, {hi:.3f}]",
        "mean_latency_s": mean_lat,
    }


# ---------------------------------------------------------------------------
# Pairwise comparisons (only on aligned items)
# ---------------------------------------------------------------------------
def align(a: pd.DataFrame, b: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, int]:
    """Inner-join on idx and return correctness arrays in matching order."""
    cols = ["idx", "correct"]
    merged = a[cols].merge(b[cols], on="idx", suffixes=("_a", "_b")).dropna()
    return (
        merged["correct_a"].astype(int).values,
        merged["correct_b"].astype(int).values,
        len(merged),
    )


def pairwise(name_a: str, df_a: pd.DataFrame, name_b: str, df_b: pd.DataFrame) -> dict:
    a, b, n = align(df_a, df_b)
    if n == 0:
        return {"a": name_a, "b": name_b, "n": 0}
    delta, lo, hi = paired_bootstrap_ci(a, b)
    mc = mcnemar_exact(a, b)
    return {
        "a": name_a,
        "b": name_b,
        "n": n,
        "acc_a": float(a.mean()),
        "acc_b": float(b.mean()),
        "delta": delta,
        "delta_lo": lo,
        "delta_hi": hi,
        "delta_str": f"{delta:+.3f} [{lo:+.3f}, {hi:+.3f}]",
        "mcnemar_b01": mc["b01"],
        "mcnemar_b10": mc["b10"],
        "mcnemar_p": mc["p"],
        "significant_at_0.05": mc["p"] < 0.05 and (lo > 0 or hi < 0),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


# Map a logical name to its CSV file. Order is the order we report in.
MODE_FILES = {
    # Stage 2 leakage-free, self-written memory by default
    "stateless":             RESULTS / "leakage_free_stateless.csv",
    "maj":                   RESULTS / "leakage_free_maj.csv",
    "mcts_judge":            RESULTS / "leakage_free_mcts_judge.csv",
    "mcts_judge_memory":     RESULTS / "leakage_free_mcts_judge_memory.csv",
    # Oracle memory
    "maj_oracle":            RESULTS / "lf_oracle_maj.csv",
    "mcts_judge_memory_oracle": RESULTS / "lf_oracle_mcts_judge_memory.csv",
    # Poisoned memory
    "maj_poison_10":         RESULTS / "lf_poisoned_10_maj.csv",
    "maj_poison_20":         RESULTS / "lf_poisoned_20_maj.csv",
    "maj_poison_50":         RESULTS / "lf_poisoned_50_maj.csv",
    "mcts_mem_poison_10":    RESULTS / "lf_poisoned_10_mcts_judge_memory.csv",
    "mcts_mem_poison_20":    RESULTS / "lf_poisoned_20_mcts_judge_memory.csv",
    "mcts_mem_poison_50":    RESULTS / "lf_poisoned_50_mcts_judge_memory.csv",
    # Defense
    "defense_off":           RESULTS / "harness_defense_off.csv",
    "defense_on":            RESULTS / "harness_defense_on.csv",
}

# Aligned pairs to test (same items only).
PAIRS = [
    ("maj", "stateless"),
    ("mcts_judge", "stateless"),
    ("mcts_judge_memory", "stateless"),
    ("mcts_judge_memory", "maj"),
    ("mcts_judge_memory", "mcts_judge"),
    ("maj_oracle", "maj"),
    ("mcts_judge_memory_oracle", "mcts_judge_memory"),
    ("maj_poison_10", "maj_oracle"),
    ("maj_poison_50", "maj_oracle"),
    ("mcts_mem_poison_10", "mcts_judge_memory_oracle"),
    ("mcts_mem_poison_50", "mcts_judge_memory_oracle"),
    ("defense_on", "defense_off"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="results/stats_summary.md")
    args = ap.parse_args()

    summaries = []
    dfs = {}
    for name, path in MODE_FILES.items():
        df = load_csv(path)
        if df is None:
            continue
        dfs[name] = df
        summaries.append(summarize(name, df))
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(RESULTS / "stats_table.csv", index=False)

    pair_rows = []
    for a, b in PAIRS:
        if a in dfs and b in dfs:
            pair_rows.append(pairwise(a, dfs[a], b, dfs[b]))
    pairs_df = pd.DataFrame(pair_rows)
    pairs_df.to_csv(RESULTS / "stats_pairs.csv", index=False)

    # ---- Markdown report ----
    lines = []
    lines.append("# Statistical Analysis of MCTS-MAJ Benchmarks")
    lines.append("")
    lines.append(
        "All accuracies are reported with two-sided Wilson 95 percent "
        "confidence intervals. Mode-vs-mode deltas are paired (computed "
        "on the same items only) with bootstrap 95 percent confidence "
        "intervals (10,000 resamples) and an exact McNemar two-sided "
        "test. A pair is flagged ``significant`` only when the McNemar "
        "p-value is below 0.05 AND the bootstrap CI excludes zero."
    )
    lines.append("")

    lines.append("## Per-mode accuracy")
    lines.append("")
    lines.append("| Mode | n | Correct | Accuracy [Wilson 95% CI] | Mean latency (s) |")
    lines.append("|------|---|---------|--------------------------|------------------|")
    for s in summaries:
        lat = "" if pd.isna(s["mean_latency_s"]) else f"{s['mean_latency_s']:.2f}"
        lines.append(
            f"| {s['name']} | {s['n']} | {s['correct']} | "
            f"{s['accuracy']*100:.1f}% [{s['wilson_lo']*100:.1f}%, "
            f"{s['wilson_hi']*100:.1f}%] | {lat} |"
        )
    lines.append("")

    lines.append("## Pairwise comparisons (paired on same items)")
    lines.append("")
    lines.append(
        "| A | B | n | Acc(A) | Acc(B) | Δ (paired bootstrap 95% CI) | "
        "McNemar p | Significant @ 0.05 |"
    )
    lines.append(
        "|---|---|---|--------|--------|----------------------------|-----------|--------------------|"
    )
    for r in pair_rows:
        if "delta" not in r:
            continue
        lines.append(
            f"| {r['a']} | {r['b']} | {r['n']} | "
            f"{r['acc_a']*100:.1f}% | {r['acc_b']*100:.1f}% | "
            f"{r['delta']*100:+.1f}pp [{r['delta_lo']*100:+.1f}, "
            f"{r['delta_hi']*100:+.1f}] | {r['mcnemar_p']:.3f} | "
            f"{'**yes**' if r['significant_at_0.05'] else 'no'} |"
        )
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- Wilson intervals are exact under the binomial model; they are "
        "preferred over normal-approximation intervals for moderate n."
    )
    lines.append(
        "- McNemar's exact test conditions on the discordant pairs and is "
        "appropriate for paired binary outcomes on identical items."
    )
    lines.append(
        "- The ``Significant`` flag is conservative: it requires both "
        "the bootstrap CI on the paired delta to exclude zero AND the "
        "McNemar p-value to be below 0.05. A trend that flips with seed "
        "should not earn this flag."
    )

    out = Path(args.output)
    out.write_text("\n".join(lines))
    print(f"Wrote {out}")
    print(f"Wrote {RESULTS / 'stats_table.csv'}")
    print(f"Wrote {RESULTS / 'stats_pairs.csv'}")

    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
