# Handover: Leverage / gambler's-ruin paper vs the fuzzer (2026-11-10)

Status: **one diagnostic implemented (e-value column), PRNG analysis done, no other code changed.**
Repo: `github.com/daedalus/fuzzer`, analysed at HEAD `bbf7ae3` (feat: --influence window influence map).
Paper: William Kuszmaul, *With Leverage, the Gambler is Not the One Who Gets Ruined*, arXiv:2610.19239v1 [cs.DS], 13 Oct 2026.

## 1. TL;DR

The paper is a toy-model note, and the fuzzer has no leverage. It maps to the repo in one place that was worth building and one place that was worth measuring:

1. **Anytime-valid A/B diagnostic (built).** `mixture_e_value(b, c)` and an `E` column next to McNemar in `tools/lib/bench_paired.py`. Diagnostic only; every promotion rule still reads McNemar.
2. **PRNG state learner (analysed and measured, no change).** The paper's real contribution there is false-positive accounting across many attempts. The exposure is small and self-heals. Recommendation: leave the code alone.
3. Two weaker links (Kelly vs power utility; stepwise vs one-shot mutation) are recorded in section 5 and not worth building without an A/B.

## 2. What the paper says (only what was used)

- Game: Alice starts with 1, bets fair (martingale) bets, may borrow so `X_t >= -(L-1) X_{t-1}`; she wins at `m`, is ruined at `<= 0`.
- `L = 1`: Optional Stopping Theorem gives `P(reach m) = 1/m` for every strategy.
- `L > 1`: the best strategy is eps-doubling (multiply by `1+eps` with probability `L/(L+eps)`, else lose everything), success probability `m^(-1/L)` as `eps -> 0`. A single all-in bet only gets `L/(m+L-1)`.
- Upper bound, Prop 2.2: `Y_t = (max(X_t,0))^(1/L)` is a supermartingale, so no strategy beats `m^(-1/L)`.
- Lesson carried over: with non-negative wealth (no leverage) the Ville/OST bound is `1/r`. If wealth is allowed to go negative the bound weakens to `r^(-1/L)`. So a valid sequential test must cap the bet at current wealth.

## 3. Implemented: e-value diagnostic next to McNemar

Commit `cf27d12` (stacked on `bbf7ae3`), shipped earlier as `e_value_patch.zip`.

| Piece | Location |
|---|---|
| `mixture_e_value(b, c)` | `src/fuzzer_tool/core/exact_tests.py` |
| `e_value` key in `compare()`, `E` column in the pooled and per-target tables, one footnote | `tools/lib/bench_paired.py` |
| Tests (11) | `tests/test_e_value.py` |
| Changelog entry | `CHANGELOG.md`, Unreleased / Added |

Definition, with `n = b + c`, wins `b`, losses `c`:

    E = int_0^1 p^b (1-p)^c dp / 0.5^n = 2^n * b! * c! / (n+1)!

computed in log space, clamped at `exp(700)`.

Properties:
- Under the null (fair win/loss), `E` is a non-negative mean-1 martingale, so by Ville `P(E ever >= 1/alpha) <= alpha` at any stopping time. `1/E` stays valid after peeking.
- Exchangeable mixture, so cell order does not matter. This is why it is a mixture and not a running all-in bet: there is no leverage.
- Two-sided. Read direction from W vs L.
- Costs power at a fixed `n`. Example: 10 wins / 0 losses gives McNemar p = 0.00195 but `E ~ 93`, so `1/E ~ 0.011`. `E >= 20` is the alpha = 0.05 line.
- `mixture_e_value(0, 0) = 1`.

What the tests check: hand values (`(1,1) -> 4/6`, `(9,1) -> 1024/110`), symmetry, finiteness at 10^6, `E[E] = 1` exactly under the null for n in {1, 5, 17}, and a peeking simulation (2000 trials, 60 pairs each, stop at `E >= 20`) staying inside alpha plus slack.

Test results at commit time: 69 pass across `test_e_value`, `test_bench_paired_stats`, `test_bench_cpu_metric`, `test_bench_paired_directed`, `test_regression_multiplicity_gaps`. Not run: the full suite.

Not done, deliberately:
- No early-stop mode in `bench_paired`. Today the column is read-only.
- No Holm/Bonferroni treatment of `E` across rows. McNemar's Holm column is unchanged.
- Not an A/B, so no `docs/AB_LEDGER.md` row.
- Replicates of one seed are not independent. Cells are collapsed per `(target, seed)` by median before `E` is computed, which is the right unit, but only 10 to 20 cells per target means `E` will rarely reach 20 on this harness. Expect it to read as "no evidence" far more often than McNemar does.

## 4. PRNG state learner (`core/analyzers/analyzer_prng_state_learner.py`)

Analysed, then one mechanism measured. **No code change.**

Inventory: 14 candidate families, 12 with 4-byte output (xorshift32, taus88, taus113, xorshift128, java, msvc, posix_rand, minstd_rand0, minstd_rand, three MWC32) and 2 with 8-byte output (xorshift64, lfsr258). Per-family confident sample counts: 2 to 6. `_CONFIRM_WORDS = 1` in all three recovery modules, i.e. one extra output word of margin per attempt (about `2^-32` or `2^-64` per family per attempt).

### 4.1 Where the paper does not apply
The learner is not a fair-bet game with leverage. Each confirming word is a likelihood ratio of `2^32` or `2^64` against "unrelated constant", far stronger than anything eps-doubling addresses.

### 4.2 What transfers: look-elsewhere accounting
The code never sums the number of chances a campaign gives the one-word margin to be beaten by luck.

| Source of chances | Per-event probability | Over 1e7 drains |
|---|---|---|
| Recovery attempts, slide-by-one windows, up to 12 families at 32 bits | about `12 * 2^-32 = 2.8e-9` | about 0.03 false recoveries |
| `_continues_from_frontier` forward search, one-draw window (`_MAX_ADVANCE_SEARCH = 64`) | `64 / 2^32 = 1.5e-8` | about 0.15 false frontier advances on 4-byte sites |
| Same, 8-byte sites | `64 / 2^64` | negligible |

These are arithmetic from the code's constants, not a Monte Carlo of the true rate (about 6.7e7 calls per expected event in Python).

### 4.3 Measurement (real `PRNGStateLearner`, stub fuzzer, xorshift32)
- All 64 of the next 64 stream words are distinct, and a one-draw window is accepted at **every** offset 0..63. Offset 64 is rejected. So a coincidental word can advance the frontier by up to 63 draws.
- 20,000 random 32-bit words: 20,000 rejected (expected false accepts 0.0003). Consistent with the 64/2^32 figure, no extra leak.
- Two-draw window, first word right at some offset and the second wrong: rejected.
- Consequence of a forced false advance at offset 40: `predict()` no longer matched the true next draw. After two real draws, `_try_recover` rebuilt the correct state and predictions matched again. Cost is about two executions of wrong predictions, not a poisoned cache.
- The self-heal step replays the `observe_execution` window logic by hand (continuation check, `len(window) < _MIN_SAMPLES` keeps the stale state, then `_try_recover`). It did not go through `observe_execution` itself, which needs an in-process runner and cmplog conditions.

### 4.4 Why the real rate is lower than 1.5e-8
On a real generator site the candidates are the stream itself, so a coincidental word there needs a reseed, a second instance, or a mixed site. The figure is an upper bound for the non-generator case.

### 4.5 Considered and rejected
Require one more confirming draw before advancing past offset 0 for a window shorter than 2. This kills the false-advance case, but it delays the gap case that the 64-wide search exists for (targets that draw without comparing every value). Net negative for a failure that heals in two draws at about 0.15 events per 1e7 drains.

### 4.6 Not done
The end-to-end run on `targets/prng_token_read.c` with a 4-byte decoy comparison site was not run. The unit-level result already bounds the exposure.

## 5. Weaker links (recorded, not built)

1. **Binary drop vs credibility.** The cached state is cleared when a window of at least `_MIN_SAMPLES` fails both continuation and recovery (`observe_execution`, `_clear_state`, `_note_refuted`). A decayed credibility could survive a gap longer than 64 draws, since after a single miss a real generator is still overwhelmingly likely. A reseed makes dropping correct. Unmeasured.
2. **`core/randomness.py` `uniformity_report`.** KS and Fisher combine p-values at a fixed length. If it is ever re-run on a growing buffer, those p-values are not valid under peeking. `mixture_e_value`-style e-values would fix that. Nothing in the repo does this today.
3. **Kelly vs power utility in `op_monte_carlo`.** `--sharpe-kelly-blend` uses `kelly_fraction(mean, variance)` (log utility). The paper's optimal potential `X^(1/L)` is power utility with exponent `1/L`, so a blend parameter could interpolate. Speculative: fuzzing rewards are not fair bets with bounded loss. Do not build without an A/B.
4. **Stepwise vs one-shot mutation.** The paper's single-bet vs eps-doubling gap (`L/(m+L-1)` vs `m^(-1/L)`) is an argument for incremental operators (`arith_mid`, lineage steps) at deep gate chains. It supports the existing design and adds no mechanism.

## 6. Decisions open for the owner

| Item | Cost | Recommendation |
|---|---|---|
| Use `E` for early stopping in `bench_paired` (`--sequential`) | small, needs a stopping rule and a run loop change | only if matrix runs are being cut short in practice |
| Offset-0-only forward search for one-draw windows | small | no |
| End-to-end decoy-site run on `prng_token_read.c` | medium | skip unless a PRNG-related failure is seen |
| Credibility decay for the cached PRNG state | medium | no, unmeasured |

## 7. Reproduction notes

- Clone `github.com/daedalus/fuzzer`, check out `bbf7ae3`, apply `e_value_patch.zip` (`git am`).
- `PYTHONPATH=src python3 -m pytest tests/test_e_value.py tests/test_bench_paired_stats.py tests/test_bench_cpu_metric.py tests/test_bench_paired_directed.py tests/test_regression_multiplicity_gaps.py -q` (needs `pytest` and `numpy`).
- The PRNG measurement (section 4.3) is a standalone snippet: build `PRNGStateLearner(SimpleNamespace(_inprocess_runner=object()))`, recover xorshift32 from the first two stream words, `_set_state`, then call `_continues_from_frontier([w])` for `w` in the next 64 words.
