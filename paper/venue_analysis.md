# Venue Analysis — "When Does Memory Help an LLM Judge?"

Researched 2026-07-09. Deadlines change; verify each before committing.

## Bottom line
The paper's true nature is an **evaluation-methodology / corrective-null-result** paper.
The best-scoped venues (NeurIPS Datasets & Benchmarks, EMNLP theme track, GEM,
Eval4NLP, ICBINB) had **spring 2026 deadlines that already passed** — so the realistic
in-window options are narrower than the five suggested. Of the five, **ELMKE @ ISWC**
is the tightest scope + full-length fit. The single most on-scope venue is one not on
the list: the **Insights from Negative Results** workshop at EMNLP.

## Ranked recommendation (deadlines in our window)

| Rank | Venue | Deadline | Archival | Length | Scope fit | Note |
|------|-------|----------|----------|--------|-----------|------|
| 1 | **ELMKE @ ISWC 2026** | **Jul 24** | CEUR (Scopus) | full 8pp OK | ★★★★ | best of Bader's five; "standardizing evaluation" = our methodology; KG angle native |
| 2 | **Insights (Negative Results) @ EMNLP 2026** | ~Aug (call pending) | ACL Anthology | **≤4pp only** | ★★★★★ | CfP literally names "evaluation metrics that prevent fair comparison"; needs condensing |
| 3 | **TMLR** (journal, rolling) | none | Yes (indexed) | no limit | ★★★★★ criteria | reviewers told NOT to reject for lack of novelty → best odds for an honest null |
| 4 | **GLOW @ ISWC 2026** | Jul 24 | CEUR (Scopus) | ≤12pp | ★★★½ | same event/deadline as ELMKE; graph+LLM+trustworthiness framing |
| 5 | **REALM @ EMNLP 2026** | Aug 5 | ACL Anthology | ≤8pp | ★★★ | frame judge as memory-augmented agent; "agent quality evaluation" in scope |
| 6 | **BlackboxNLP @ EMNLP 2026** | Jul 17 | ACL Anthology | ≤8pp | ★★★ | lead with the "why memory doesn't help" analysis; reproducibility special track fits |
| 7 | **INLG 2026** | Jul 15 | ACL Anthology | ≤8pp | ★★½ | highest brand, weakest fit (generation conf); off-core for a judge-eval paper |
| — | **ARR Aug 3 → EACL 2027** | Aug 3 | ACL venue | 8pp | ★★★★ | EMNLP/AACL 2026 already unreachable; this feeds 2027 |

## What I'd actually do
- **If we want to submit this month:** **ELMKE (Jul 24)** with the full 8-page paper. It's
  archival, in-window, and its mission maps onto the leakage-free audited methodology.
  GLOW is the same-day backup at the same event.
- **In parallel (allowed, different manuscript status):** **TMLR** — lowest-risk home for a
  full-length null result; rolling, so no deadline pressure. Do NOT dual-submit the same
  manuscript to two archival venues at once.
- **Prepare a ≤4-page condensed version** (lead with protocol + two-layer crypto audit) for
  **Insights @ EMNLP** once its call posts (~Aug). Best pure-scope match.
- **If an ACL-brand conference is the goal:** ARR Aug 3 → EACL 2027, framed as *corrective
  methodology* ("prior practice overstates memory gains; here is the audited correction"),
  not "we found nothing."

## Honesty flags (verify before committing)
- Insights 2026 and a possible NeurIPS 2026 LLM-eval workshop calls are **not yet posted**;
  Aug/Sep estimates are inferred from sibling EMNLP/NeurIPS 2026 workshops.
- ELMKE's 2026 page doesn't restate CEUR archival status or 2026 page limits — confirm with
  organizers.
- Passed for 2026 (target 2027): NeurIPS D&B (May 6), EMNLP theme track (May 25), GEM
  (Mar 19), Eval4NLP (Sep 2025), ICBINB@ICLR (Jan 31), TrustNLP (Mar 5), CONDA (dormant).
