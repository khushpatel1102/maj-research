"""
Figures for the round-2 revision experiments, built from the per-sample CSVs in
results/ (nothing re-evaluated). Matches the visual style of make_figures.py.

Outputs (to paper/figures/):
  memory_controls.png  -- exp2: accuracy per injected-memory arm, Wilson 95% CI,
                          with the stateless baseline drawn as a reference line.
  cv_deltas.png        -- exp3: forest plot of paired question-cluster-bootstrap
                          deltas vs stateless (grouped 5-fold CV, n=160).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sps

RESULTS = Path("results")
FIGDIR = Path("paper/figures")
FIGDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 10,
    "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
})

ACCENT = "#8b1538"   # match the deck/paper accent
BASE = "#1a3d6e"
GOOD = "#1f7a3a"
MUTE = "#6a6a66"


def wilson_ci(k, n, alpha=0.05):
    if n == 0:
        return 0.0, 1.0
    z = sps.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return max(0.0, (centre - margin) / denom), min(1.0, (centre + margin) / denom)


def acc_and_ci(df):
    c = df.dropna(subset=["correct"])
    k = int(c["correct"].astype(str).isin(["True", "1", "1.0"]).sum())
    n = len(c)
    p = 100 * k / n
    lo, hi = wilson_ci(k, n)
    return p, 100 * lo, 100 * hi, n


def load(name):
    p = RESULTS / name
    return pd.read_csv(p) if p.exists() else None


# ---------------------------------------------------------------------------
# Figure 1: memory-use controls (exp2)
# ---------------------------------------------------------------------------
def fig_controls():
    arms = [
        ("Cheating\noracle", "exp2_cheating_oracle_seed42.csv", ACCENT),
        ("Irrelevant\n(len-matched)", "exp2_irrelevant_samelen_seed42.csv", MUTE),
        ("Random\ncontext", "exp2_random_context_seed42.csv", MUTE),
        ("Label\nshuffled", "exp2_label_shuffled_seed42.csv", MUTE),
    ]
    sl = load("exp2_stateless_seed42.csv")
    sl_acc, sl_lo, sl_hi, _ = acc_and_ci(sl)

    labels, accs, los, his, colors = [], [], [], [], []
    for lab, f, col in arms:
        df = load(f)
        if df is None:
            continue
        p, lo, hi, n = acc_and_ci(df)
        labels.append(lab); accs.append(p); los.append(p - lo); his.append(hi - p)
        colors.append(col)

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    x = np.arange(len(labels))
    ax.bar(x, accs, width=0.62, color=colors, zorder=3,
           yerr=[los, his], capsize=4, ecolor="#333", error_kw={"lw": 1.2})
    # stateless baseline band
    ax.axhspan(sl_lo, sl_hi, color=BASE, alpha=0.12, zorder=1)
    ax.axhline(sl_acc, color=BASE, lw=1.6, ls="--", zorder=2,
               label=f"Stateless baseline ({sl_acc:.1f}%)")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(40, 90)
    ax.set_title("Memory-use controls: only informative memory helps")
    for xi, a in zip(x, accs):
        ax.text(xi, a + max(his) + 1.5, f"{a:.1f}", ha="center", fontsize=9,
                fontweight="bold")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()
    out = FIGDIR / "memory_controls.png"
    fig.savefig(out); plt.close(fig)
    print("wrote", out)


# ---------------------------------------------------------------------------
# Figure 2: grouped-CV paired deltas (exp3)
# ---------------------------------------------------------------------------
def cluster_bootstrap_delta(base, other, n_boot=10000, seed=42):
    m = base.merge(other, on="idx", suffixes=("_a", "_b")).dropna(
        subset=["correct_a", "correct_b"])
    def tb(s):  # correctness to float
        return s.astype(str).isin(["True", "1", "1.0"]).astype(float)
    m["ca"] = tb(m["correct_a"]); m["cb"] = tb(m["correct_b"])
    qcol = "question_a" if "question_a" in m else "question"
    by_q = {q: g for q, g in m.groupby(qcol)}
    qs = list(by_q.keys())
    rng = np.random.RandomState(seed)
    point = 100 * (m["cb"].mean() - m["ca"].mean())
    ds = []
    for _ in range(n_boot):
        pick = rng.choice(len(qs), len(qs), replace=True)
        rows = pd.concat([by_q[qs[i]] for i in pick])
        ds.append(100 * (rows["cb"].mean() - rows["ca"].mean()))
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return point, lo, hi


def fig_cv_deltas():
    sl = load("exp3_cv_stateless_seed42.csv")
    modes = [
        ("MAJ (self)", "exp3_cv_maj_self_seed42.csv"),
        ("MAJ (oracle)", "exp3_cv_maj_oracle_seed42.csv"),
        ("MAJ (balanced)", "exp3_cv_maj_balanced_seed42.csv"),
    ]
    rows = []
    for lab, f in modes:
        df = load(f)
        if df is None:
            continue
        pt, lo, hi = cluster_bootstrap_delta(sl, df)
        rows.append((lab, pt, lo, hi))

    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, pt, lo, hi) in zip(y, rows):
        sig = not (lo <= 0 <= hi)
        col = ACCENT if sig else MUTE
        ax.plot([lo, hi], [yi, yi], color=col, lw=2.2, zorder=2)
        ax.plot(pt, yi, "o", color=col, ms=7, zorder=3)
        ax.text(hi + 0.4, yi, f"{pt:+.1f} [{lo:+.1f}, {hi:+.1f}]",
                va="center", fontsize=8.5, color="#222")
    ax.axvline(0, color="#333", lw=1.1, ls="--", zorder=1)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("Accuracy delta vs. stateless (pp), 95% cluster-bootstrap CI")
    ax.set_title("Grouped 5-fold CV (n=160): no mode beats stateless")
    ax.set_xlim(-10, 8)
    fig.tight_layout()
    out = FIGDIR / "cv_deltas.png"
    fig.savefig(out); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    fig_controls()
    fig_cv_deltas()
