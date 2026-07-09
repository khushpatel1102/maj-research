# Statistical Analysis of MCTS-MAJ Benchmarks

All accuracies are reported with two-sided Wilson 95 percent confidence intervals. Mode-vs-mode deltas are paired (computed on the same items only) with bootstrap 95 percent confidence intervals (10,000 resamples) and an exact McNemar two-sided test. A pair is flagged ``significant`` only when the McNemar p-value is below 0.05 AND the bootstrap CI excludes zero.

## Per-mode accuracy

| Mode | n | Correct | Accuracy [Wilson 95% CI] | Mean latency (s) |
|------|---|---------|--------------------------|------------------|
| stateless | 80 | 56 | 70.0% [59.2%, 78.9%] | 3.58 |
| maj | 80 | 52 | 65.0% [54.1%, 74.5%] | 4.51 |
| mcts_judge | 80 | 56 | 70.0% [59.2%, 78.9%] | 31.82 |
| mcts_judge_memory | 78 | 56 | 71.8% [61.0%, 80.6%] | 33.03 |
| maj_oracle | 80 | 48 | 60.0% [49.0%, 70.0%] | 4.53 |
| mcts_judge_memory_oracle | 78 | 51 | 65.4% [54.3%, 75.0%] | 34.06 |
| maj_poison_10 | 80 | 50 | 62.5% [51.5%, 72.3%] | 4.19 |
| maj_poison_20 | 80 | 51 | 63.7% [52.8%, 73.4%] | 4.38 |
| maj_poison_50 | 80 | 50 | 62.5% [51.5%, 72.3%] | 4.36 |
| mcts_mem_poison_10 | 77 | 52 | 67.5% [56.5%, 76.9%] | 37.11 |
| mcts_mem_poison_20 | 76 | 53 | 69.7% [58.7%, 78.9%] | 121.55 |
| mcts_mem_poison_50 | 78 | 51 | 65.4% [54.3%, 75.0%] | 49.00 |
| defense_off | 80 | 55 | 68.8% [57.9%, 77.8%] |  |
| defense_on | 80 | 52 | 65.0% [54.1%, 74.5%] |  |

## Pairwise comparisons (paired on same items)

| A | B | n | Acc(A) | Acc(B) | Δ (paired bootstrap 95% CI) | McNemar p | Significant @ 0.05 |
|---|---|---|--------|--------|----------------------------|-----------|--------------------|
| maj | stateless | 80 | 65.0% | 70.0% | -5.0pp [-12.5, +1.2] | 0.289 | no |
| mcts_judge | stateless | 80 | 70.0% | 70.0% | +0.0pp [-13.8, +13.8] | 1.000 | no |
| mcts_judge_memory | stateless | 78 | 71.8% | 69.2% | +2.6pp [-7.7, +12.8] | 0.804 | no |
| mcts_judge_memory | maj | 78 | 71.8% | 64.1% | +7.7pp [-1.3, +16.7] | 0.180 | no |
| mcts_judge_memory | mcts_judge | 78 | 71.8% | 70.5% | +1.3pp [-11.5, +14.1] | 1.000 | no |
| maj_oracle | maj | 80 | 60.0% | 65.0% | -5.0pp [-11.2, +0.0] | 0.219 | no |
| mcts_judge_memory_oracle | mcts_judge_memory | 76 | 64.5% | 71.1% | -6.6pp [-15.8, +2.6] | 0.267 | no |
| maj_poison_10 | maj_oracle | 80 | 62.5% | 60.0% | +2.5pp [+0.0, +6.2] | 0.500 | no |
| maj_poison_50 | maj_oracle | 80 | 62.5% | 60.0% | +2.5pp [+0.0, +6.2] | 0.500 | no |
| mcts_mem_poison_10 | mcts_judge_memory_oracle | 75 | 66.7% | 64.0% | +2.7pp [-6.7, +12.0] | 0.791 | no |
| mcts_mem_poison_50 | mcts_judge_memory_oracle | 76 | 64.5% | 64.5% | +0.0pp [-9.2, +9.2] | 1.000 | no |
| defense_on | defense_off | 80 | 65.0% | 68.8% | -3.8pp [-11.2, +5.0] | 0.549 | no |

## Notes

- Wilson intervals are exact under the binomial model; they are preferred over normal-approximation intervals for moderate n.
- McNemar's exact test conditions on the discordant pairs and is appropriate for paired binary outcomes on identical items.
- The ``Significant`` flag is conservative: it requires both the bootstrap CI on the paired delta to exclude zero AND the McNemar p-value to be below 0.05. A trend that flips with seed should not earn this flag.