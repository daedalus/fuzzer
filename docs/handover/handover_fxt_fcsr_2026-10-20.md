# Handover: FXT book survey + MWC/FCSR recovery (2026-10-20)

Source: Arndt, *Matters Computational* (FXT, 978 pp). Read: full TOC, Ch. 1 sample, sec. 41.7. Chapter bodies otherwise unread.

## Done
- `core/fcsr_recovery.py`: multiply-with-carry (FCSR) state recovery, base 2^32, state `(x, c)`.
  - Known multiplier (`MWC_FAMILIES`, `a*b-1` safe prime): 2 words pin the state, 3 confident.
  - Unknown multiplier: `adic_synth` (Gauss lattice reduction, 2-adic rational approximation) gives `q = a*b-1` from 5 words; `a = (q+1)/b`.
- Wired in `PRNGStateLearner`: known families after the LCGs; `_try_unknown_mwc` only after every family refutes the window; persisted as `mwc32_<a>` (`mwc_family` rebuilds, invalid names fall back to taus88).
- Tests: `test_fcsr_recovery.py` (34), `test_prng_state_learner.py::TestMWCRecovery` (5). Fallback cost 45 us (5 words), 192 us (16).

## Findings
- FXT 41.7 emits `2^k mod c` LSBs: the time-reversed l-sequence. Its 2-adic complexity is ~T/2 (18 bits for c=37); reversed, it is log2(c). The module uses the forward word stream; measured, not assumed.
- Composite `a*b-1`: `p/q` can reduce, `q+1` stops being a multiple of `b`, result refused (never a wrong `a`). Measured 65-97% recovery for `a` = 1000003 and 196605, 100% from 5 words for multipliers near 2^32 with prime `a*b-1`.
- Shipped multipliers 698769069, 4294584393, 4294883355: `a*b-1` safe prime (verified). Provenance not verified; named by value.

## Not done
- 64-bit MWC (`b = 2^64`, `out_bytes` 8): spec is 32-bit only.
- Add-with-carry / subtract-with-borrow (lagged, Marsaglia-Zaman): carry per step is `(x[n-s] - x[n-r] - x[n]) mod b` in {0, 1}; needs an r-word state spec.
- `adic_synth` uses Euclidean-norm reduction plus a +-3 combo search, not the exact max-norm optimum. Fine for N >> complexity.
- No A/B on a live target, no fuzzgoat calibration (Rule 52), full suite and pre-commit hooks not run (affected tests only, Rule 50).
- `architecture.png` untouched (no new subsystem).
- Pre-existing: ruff B007 in `_try_recover_mt` (`site` unused).

## FXT candidates (status checked against `src/`)
| Item (FXT) | Status | Note |
|---|---|---|
| Walsh / Reed-Muller transforms (Ch. 23) | absent | Boolean-influence per input bit from a probed predicate: non-NN alternative to `pos_saliency`. |
| Hilbert / space-filling curves (1.31, 38.9) | absent | Morton exists (`int_encodings` bit interleave); Hilbert only as a name in `qea.py`. |
| Hadamard / conference matrices (Ch. 19) | absent | Orthogonal arrays for 2-level fields; complements `covering_array.py`. |
| Gosper / bit subsets (1.24-1.26) | done | `mutations/hackers_delight.py`; see `handover_hackers_delight_rightmost_gosper_2026-10-05.md`. |
| CRC / GF(2)[x] (Ch. 40, 41.3) | done | `crc_engine.py`, `crc32.py`, `crc_learn`. |
| LFSR / Berlekamp-Massey | done | `berlekamp_massey.py`, `prng_state_recovery.py`. |
| NTT (Ch. 26) | done | `ntt.py`. |
| Combinatorial generation (Ch. 6-18) | overlaps | Diff against `handover_combinatorics_permutations_2026-09-02.md` first. |
| FCSR (41.7) | done | This handover. |

## Out of fuzzer scope
- OEIS: 2-adic complexity of the binary expansion of n, sibling of A397621 (finite-prefix `adic_synth` is the engine). Definition and duplicates unchecked.
- Not worth it: bignum FFT multiplication in Python (use gmpy2/flint); SFC64 (ARX, not an FCSR).

## Licence
FXT is GPL as far as known: reimplement from the algorithm descriptions, do not port code verbatim.
