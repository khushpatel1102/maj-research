# CEUR / ELMKE submission — Overleaf setup

The CEUR single-column version is `paper_ceur.tex`. Compile it on Overleaf
(local TeX Live 2025 is missing several ceurart dependencies; Overleaf's 2026
distribution has them all).

## Upload to the Overleaf project
- `paper_ceur.tex`   — the CEUR single-column source (main document)
- `references.bib`   — bibliography (includes the 4 ELMKE citations)
- `figures/`         — all result PNGs, including the 2 NEW ones:
                       `memory_controls.png` (exp2) and `cv_deltas.png` (exp3)
- **Do NOT upload `ceurart.cls`** — Overleaf already provides it. (There's a
  local copy here only for reference; uploading it can shadow Overleaf's newer
  version.) If Overleaf can't find the class, start from its built-in template
  "Template for submissions to CEUR Workshop Proceedings" and paste our body in.
- **Do NOT upload `acl.sty`** — that's for the ACL version only.

## Overleaf settings
- Compiler: **pdfLaTeX**
- Main document: `paper_ceur.tex`
- The class loads `pdfx` (PDF/A) only in *final* mode; for review it's fine.

## What was converted from the ACL version
- Document class: `acl` → `ceurart` (single-column; `twocolumn` is forbidden by CEUR).
- Title/author block → CEUR `\author[1]{...}[email=...]` + `\address[1]{...}` +
  `\cormark`/`\cortext`.
- Added `\conference{ELMKE 2026 ...}`, `\copyrightyear`, `\copyrightclause`.
- Abstract kept verbatim; added a `\begin{keywords}...\sep...\end{keywords}` block.
- Bibliography: `\bibliographystyle{elsarticle-num-names}` (CEUR default) + natbib
  `\citep`/`\citet` (ceurart uses natbib, so these work unchanged).
- Figures rescaled `\columnwidth` → `0.72\linewidth` (columnwidth = full width in
  single-column, which would make them span the page).
- All content, tables, TikZ diagrams, and the 4 new experiment sections carry over
  unchanged.

## Before submitting — TODO
1. **Author block**: currently shows the real names (Khush Patel, Bader Rasheed) +
   `@innopolis.university` emails. Fix the email addresses to the real ones.
2. **Anonymity**: confirm whether ELMKE 2026 review is anonymous. If double-blind,
   add `\documentclass[review,anonymous]{ceurart}` or strip the author block per
   their instructions. (The CfP didn't state a policy — email elmke@googlegroups.com.)
3. **Page limit**: ELMKE 2024 was full 10–15pp incl. refs; the paper is ~10-11pp in
   single column, which fits. Verify the 2026 limit hasn't changed.
4. **Conference line**: I set ISWC 2026, Bari, Oct 25–29 — double-check the exact
   ELMKE 2026 dates/location string.
5. **Deadline: July 24, 2026.**
