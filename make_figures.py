"""
All figures are built from the per-sample CSVs in ``results/``. Nothing is
re-evaluated. Output PNGs go to ``results/figures/``.

Figures produced
----------------
1. accuracy_ci.png       -- accuracy with Wilson 95% CI per mode
2. paired_deltas.png     -- forest plot of paired mode-vs-mode deltas
3. confusion_matrices.png-- per-mode 2x2 confusion matrices
4. passfail_bias.png     -- per-class (pass vs fail) accuracy per mode
5. poisoning_curve.png   -- accuracy vs poison rate with CI bands
6. harness_results.png   -- stability / label-flip / defense bars
7. latency_accuracy.png  -- cost-reliability scatter (accuracy vs latency)

Usage
-----
    python make_figures.py
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sps

RESULTS = Path("results")
FIGDIR = RESULTS / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
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


def load(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def acc_and_ci(df: pd.DataFrame) -> tuple[float, float, float, int, int]:
    df = df.dropna(subset=["correct"])
    n = len(df)
    k = int(df["correct"].astype(bool).sum())
    lo, hi = wilson_ci(k, n)
    return (k / n if n else 0.0), lo, hi, k, n


def align(a: pd.DataFrame, b: pd.DataFrame):
    m = a[["idx", "correct"]].merge(b[["idx", "correct"]], on="idx",
                                    suffixes=("_a", "_b")).dropna()
    return m["correct_a"].astype(int).values, m["correct_b"].astype(int).values


# ---------------------------------------------------------------------------
# Source files
# ---------------------------------------------------------------------------
MODES = {
    "Stateless":            RESULTS / "leakage_free_stateless.csv",
    "MAJ (bare)":           RESULTS / "lf_bare_maj.csv",
    "MAJ":                  RESULTS / "leakage_free_maj.csv",
    "MCTS-Judge":           RESULTS / "leakage_free_mcts_judge.csv",
    "MCTS-Judge+Memory":    RESULTS / "leakage_free_mcts_judge_memory.csv",
    "MAJ (oracle)":         RESULTS / "lf_oracle_maj.csv",
    "MCTS+Mem (oracle)":    RESULTS / "lf_oracle_mcts_judge_memory.csv",
}

# Multi-seed runs (stateless + MAJ across 3 seeds).
MULTISEED = {
    "stateless (no memory)": {
        42:  RESULTS / "leakage_free_stateless.csv",
        123: RESULTS / "lf_no_memory_stateless_seed123.csv",
        7:   RESULTS / "lf_no_memory_stateless_seed7.csv",
    },
    "maj (self-written)": {
        42:  RESULTS / "leakage_free_maj.csv",
        123: RESULTS / "lf_self_written_maj_seed123.csv",
        7:   RESULTS / "lf_self_written_maj_seed7.csv",
    },
    "maj (oracle)": {
        42:  RESULTS / "lf_oracle_maj.csv",
        123: RESULTS / "lf_oracle_maj_seed123.csv",
        7:   RESULTS / "lf_oracle_maj_seed7.csv",
    },
}

POISON = {
    "MAJ":              {0.0: RESULTS / "lf_oracle_maj.csv",
                         0.10: RESULTS / "lf_poisoned_10_maj.csv",
                         0.20: RESULTS / "lf_poisoned_20_maj.csv",
                         0.50: RESULTS / "lf_poisoned_50_maj.csv"},
    "MCTS-Judge+Memory": {0.0: RESULTS / "lf_oracle_mcts_judge_memory.csv",
                          0.10: RESULTS / "lf_poisoned_10_mcts_judge_memory.csv",
                          0.20: RESULTS / "lf_poisoned_20_mcts_judge_memory.csv",
                          0.50: RESULTS / "lf_poisoned_50_mcts_judge_memory.csv"},
}

LATENCY_MODES = ["Stateless", "MAJ", "MCTS-Judge", "MCTS-Judge+Memory"]

PAIRS = [
    ("MAJ", "Stateless"),
    ("MAJ (bare)", "Stateless"),
    ("MAJ (bare)", "MAJ"),
    ("MCTS-Judge", "Stateless"),
    ("MCTS-Judge+Memory", "Stateless"),
    ("MCTS-Judge+Memory", "MAJ"),
    ("MCTS-Judge+Memory", "MCTS-Judge"),
    ("MAJ (oracle)", "MAJ"),
    ("MCTS+Mem (oracle)", "MCTS-Judge+Memory"),
]


# ---------------------------------------------------------------------------
# Figure 1: accuracy with Wilson CIs
# ---------------------------------------------------------------------------
def fig_accuracy_ci(dfs):
    names, accs, los, his = [], [], [], []
    for name in MODES:
        df = dfs.get(name)
        if df is None:
            continue
        a, lo, hi, _, _ = acc_and_ci(df)
        names.append(name)
        accs.append(a * 100)
        los.append((a - lo) * 100)
        his.append((hi - a) * 100)
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(names))
    ax.bar(x, accs, color="#4C72B0", alpha=0.85)
    ax.errorbar(x, accs, yerr=[los, his], fmt="none", ecolor="black", capsize=4, lw=1.2)
    ax.axhline(50, ls="--", c="grey", lw=1, label="chance (50%)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy with Wilson 95% confidence intervals (80 test samples, GPT-4o)")
    ax.set_ylim(0, 100)
    ax.legend()
    for xi, a in zip(x, accs):
        ax.text(xi, a + 2, f"{a:.1f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGDIR / "accuracy_ci.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: paired deltas forest plot
# ---------------------------------------------------------------------------
def fig_paired_deltas(dfs):
    rows = []
    for a, b in PAIRS:
        da, db = dfs.get(a), dfs.get(b)
        if da is None or db is None:
            continue
        x, y = align(da, db)
        d, lo, hi = paired_bootstrap_ci(x, y)
        rows.append((f"{a}\n  vs {b}", d * 100, lo * 100, hi * 100))
    if not rows:
        return
    labels = [r[0] for r in rows]
    deltas = [r[1] for r in rows]
    lows = [r[1] - r[2] for r in rows]
    highs = [r[3] - r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 0.7 * len(rows) + 1.5))
    y = np.arange(len(rows))
    colors = ["#55A868" if d > 0 else "#C44E52" for d in deltas]
    ax.errorbar(deltas, y, xerr=[lows, highs], fmt="o", color="black",
                ecolor="grey", capsize=4)
    for yi, d, c in zip(y, deltas, colors):
        ax.plot(d, yi, "o", color=c, ms=8)
    ax.axvline(0, ls="--", c="grey", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Paired accuracy delta A − B (percentage points), bootstrap 95% CI")
    ax.set_title("Paired mode-vs-mode deltas (computed on identical items)")
    fig.tight_layout()
    fig.savefig(FIGDIR / "paired_deltas.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: per-mode confusion matrices
# ---------------------------------------------------------------------------
def fig_confusion_matrices(dfs):
    targets = ["Stateless", "MAJ", "MCTS-Judge", "MCTS-Judge+Memory"]
    avail = [t for t in targets if dfs.get(t) is not None]
    if not avail:
        return
    fig, axes = plt.subplots(1, len(avail), figsize=(3.2 * len(avail), 3.4))
    if len(avail) == 1:
        axes = [axes]
    for ax, name in zip(axes, avail):
        df = dfs[name].dropna(subset=["predicted"])
        exp = df["expected"].astype(bool)
        pred = df["predicted"].astype(bool)
        cm = np.array([
            [int(((~exp) & (~pred)).sum()), int(((~exp) & pred).sum())],
            [int((exp & (~pred)).sum()),    int((exp & pred).sum())],
        ])
        im = ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                        fontsize=12)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["pred fail", "pred pass"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["true fail", "true pass"])
        ax.set_title(name, fontsize=10)
    fig.suptitle("Confusion matrices (80 test samples)", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGDIR / "confusion_matrices.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: pass/fail bias (per-class accuracy)
# ---------------------------------------------------------------------------
def fig_passfail_bias(dfs):
    targets = ["Stateless", "MAJ", "MCTS-Judge", "MCTS-Judge+Memory"]
    avail = [t for t in targets if dfs.get(t) is not None]
    if not avail:
        return
    pass_acc, fail_acc = [], []
    for name in avail:
        df = dfs[name].dropna(subset=["predicted"])
        p = df[df["expected"] == True]
        f = df[df["expected"] == False]
        pass_acc.append(p["correct"].mean() * 100 if len(p) else 0)
        fail_acc.append(f["correct"].mean() * 100 if len(f) else 0)
    x = np.arange(len(avail))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w/2, pass_acc, w, label="accuracy on PASS items", color="#55A868")
    ax.bar(x + w/2, fail_acc, w, label="accuracy on FAIL items", color="#C44E52")
    ax.axhline(50, ls="--", c="grey", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(avail, rotation=20, ha="right")
    ax.set_ylabel("Per-class accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Pass/fail bias: accuracy split by ground-truth label")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGDIR / "passfail_bias.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5: poisoning curves with CI bands
# ---------------------------------------------------------------------------
def fig_poisoning_curve():
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colours = {"MAJ": "#4C72B0", "MCTS-Judge+Memory": "#C44E52"}
    for mode, files in POISON.items():
        rates, accs, los, his = [], [], [], []
        for r, path in sorted(files.items()):
            df = load(path)
            if df is None:
                continue
            a, lo, hi, _, _ = acc_and_ci(df)
            rates.append(r * 100); accs.append(a * 100)
            los.append(lo * 100); his.append(hi * 100)
        if not rates:
            continue
        ax.plot(rates, accs, "o-", color=colours.get(mode, None), label=mode)
        ax.fill_between(rates, los, his, color=colours.get(mode, None), alpha=0.18)
    ax.axhline(50, ls="--", c="grey", lw=1, label="chance (50%)")
    ax.set_xlabel("Memory poisoning rate (% of training labels flipped)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Accuracy under memory poisoning (Wilson 95% CI bands)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGDIR / "poisoning_curve.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6: harness results
# ---------------------------------------------------------------------------
def fig_harness_results():
    stab = load(RESULTS / "harness_stability.csv")
    flip = load(RESULTS / "harness_label_flip.csv")
    doff = load(RESULTS / "harness_defense_off.csv")
    don = load(RESULTS / "harness_defense_on.csv")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

    # Stability
    if stab is not None and "mode" in stab:
        modes = stab["mode"].unique()
        vals = [stab[(stab["mode"] == m)]["agree"].mean() * 100 for m in modes]
        axes[0].bar(range(len(modes)), vals, color="#4C72B0")
        axes[0].set_xticks(range(len(modes)))
        axes[0].set_xticklabels(modes, rotation=20, ha="right", fontsize=8)
        axes[0].set_ylim(0, 100); axes[0].set_ylabel("Agreement (%)")
        axes[0].set_title("Stochastic stability")
        for i, v in enumerate(vals):
            axes[0].text(i, v + 2, f"{v:.0f}", ha="center", fontsize=8)
    else:
        axes[0].set_visible(False)

    # Label flip
    if flip is not None and "mode" in flip:
        modes = flip["mode"].unique()
        col = "correctly_flipped" if "correctly_flipped" in flip else "correct"
        vals = [flip[(flip["mode"] == m)][col].mean() * 100 for m in modes]
        axes[1].bar(range(len(modes)), vals, color="#DD8452")
        axes[1].axhline(50, ls="--", c="grey", lw=1)
        axes[1].set_xticks(range(len(modes)))
        axes[1].set_xticklabels(modes, rotation=20, ha="right", fontsize=8)
        axes[1].set_ylim(0, 100); axes[1].set_ylabel("Correct flip (%)")
        axes[1].set_title("Label-flip discrimination")
        for i, v in enumerate(vals):
            axes[1].text(i, v + 2, f"{v:.0f}", ha="center", fontsize=8)
    else:
        axes[1].set_visible(False)

    # Defense
    if doff is not None and don is not None:
        a_off, _, _, _, _ = acc_and_ci(doff)
        a_on, _, _, _, _ = acc_and_ci(don)
        axes[2].bar([0, 1], [a_off * 100, a_on * 100],
                    color=["#C44E52", "#55A868"])
        axes[2].set_xticks([0, 1])
        axes[2].set_xticklabels(["no defense", "similarity filter"], fontsize=9)
        axes[2].set_ylim(0, 100); axes[2].set_ylabel("Accuracy (%)")
        axes[2].set_title("Defense filter (50% poisoned memory)")
        for i, v in enumerate([a_off * 100, a_on * 100]):
            axes[2].text(i, v + 2, f"{v:.1f}", ha="center", fontsize=8)
    else:
        axes[2].set_visible(False)

    fig.suptitle("Reliability harness results", y=1.04)
    fig.tight_layout()
    fig.savefig(FIGDIR / "harness_results.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 7: latency vs accuracy
# ---------------------------------------------------------------------------
def fig_latency_accuracy(dfs):
    pts = []
    for name in LATENCY_MODES:
        df = dfs.get(name)
        if df is None or "latency_s" not in df:
            continue
        a, lo, hi, _, _ = acc_and_ci(df)
        lat = df["latency_s"].mean()
        pts.append((name, lat, a * 100, (a - lo) * 100, (hi - a) * 100))
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, lat, a, lo, hi in pts:
        ax.errorbar(lat, a, yerr=[[lo], [hi]], fmt="o", capsize=4, ms=9)
        ax.annotate(name, (lat, a), textcoords="offset points",
                    xytext=(8, 6), fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Mean latency per evaluation (s, log scale)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Cost-reliability frontier (accuracy vs latency, Wilson 95% CI)")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(FIGDIR / "latency_accuracy.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 8: multi-seed cross-seed comparison
# ---------------------------------------------------------------------------
def fig_multiseed():
    rows = []
    for cond, files in MULTISEED.items():
        for seed, path in files.items():
            df = load(path)
            if df is None:
                continue
            a, lo, hi, _, _ = acc_and_ci(df)
            rows.append((cond, seed, a * 100, (a - lo) * 100, (hi - a) * 100))
    if not rows:
        return
    df = pd.DataFrame(rows, columns=["cond", "seed", "acc", "lo", "hi"])
    conds = list(MULTISEED.keys())
    seeds = sorted({r[1] for r in rows})
    x = np.arange(len(conds))
    w = 0.8 / len(seeds)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for i, s in enumerate(seeds):
        ys = [df[(df["cond"] == c) & (df["seed"] == s)]["acc"].iloc[0]
              if not df[(df["cond"] == c) & (df["seed"] == s)].empty else 0
              for c in conds]
        los = [df[(df["cond"] == c) & (df["seed"] == s)]["lo"].iloc[0]
               if not df[(df["cond"] == c) & (df["seed"] == s)].empty else 0
               for c in conds]
        his = [df[(df["cond"] == c) & (df["seed"] == s)]["hi"].iloc[0]
               if not df[(df["cond"] == c) & (df["seed"] == s)].empty else 0
               for c in conds]
        offset = (i - (len(seeds) - 1) / 2) * w
        ax.bar(x + offset, ys, w, label=f"seed {s}",
               yerr=[los, his], capsize=3, alpha=0.9)
    ax.axhline(50, ls="--", c="grey", lw=1, label="chance (50%)")
    ax.set_xticks(x)
    ax.set_xticklabels(conds, rotation=15, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Multi-seed comparison (stateless + MAJ, 80 samples per seed, Wilson 95% CI)")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGDIR / "multiseed.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 9: structure ablation (bare vs full vs stateless)
# ---------------------------------------------------------------------------
def fig_structure_ablation(dfs):
    targets = [("Stateless (no memory)", "Stateless"),
               ("Bare memory\n(Policy + Attempt only)", "MAJ (bare)"),
               ("Full self-written\n(5-node typed graph)", "MAJ")]
    rows = []
    for label, key in targets:
        df = dfs.get(key)
        if df is None:
            continue
        a, lo, hi, _, _ = acc_and_ci(df)
        rows.append((label, a * 100, (a - lo) * 100, (hi - a) * 100))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(rows))
    accs = [r[1] for r in rows]
    los = [r[2] for r in rows]
    his = [r[3] for r in rows]
    ax.bar(x, accs, color=["#999999", "#4C72B0", "#C44E52"], alpha=0.9)
    ax.errorbar(x, accs, yerr=[los, his], fmt="none", ecolor="black",
                capsize=4, lw=1.2)
    ax.axhline(50, ls="--", c="grey", lw=1, label="chance (50%)")
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], fontsize=9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Structure ablation: does the typed graph earn its complexity? (seed 42, n=80)")
    ax.legend(loc="lower right", fontsize=9)
    for xi, a in zip(x, accs):
        ax.text(xi, a + 2, f"{a:.1f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGDIR / "structure_ablation.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    dfs = {name: load(path) for name, path in MODES.items()}

    fig_accuracy_ci(dfs)
    fig_paired_deltas(dfs)
    fig_confusion_matrices(dfs)
    fig_passfail_bias(dfs)
    fig_poisoning_curve()
    fig_harness_results()
    fig_latency_accuracy(dfs)
    fig_multiseed()
    fig_structure_ablation(dfs)

    produced = sorted(p.name for p in FIGDIR.glob("*.png"))
    print(f"Wrote {len(produced)} figures to {FIGDIR}/")
    for p in produced:
        print(f"  {p}")


if __name__ == "__main__":
    main()
