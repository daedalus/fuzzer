# `ratio_flip_e2..e5` A/B: null on fuzzgoat, no decade dominates

**Question.** The four zzuf `-r` arms (`ratio_flip_e5..e2`, one flip-density decade each,
`core/mutations/ratio_flip.py`) were shipped on unit tests only. Does adding them to the
`bit` band change edges at a fixed exec budget, and does the scheduler converge on one
decade? Owed in `docs/AB_PENDING.md` (closed by ledger #173).

**Answer: null on edges; no decade dominates.** 9 wins, 11 losses, median -8.5 edges
(-1.4%). Not distinguishable from zero at this cell count, and not powered to exclude a
loss or a gain of about +-70 edges (see Power).

## Setup

`fuzzgoat_read_noasan.so`, clang 18 `--clang-scov` build (Hard Rule 52),
`-D dictionaries/json.dict --boltzmann`, 10,000 execs per campaign, seeds 1-20, one
replicate per cell, single CPU (`tools/benchmark.py replicated --set fuzzgoat`, arms
back to back within a seed). Metric: `Edges discovered`.

| arm | tree |
|---|---|
| `off` (base) | HEAD with the four names deleted from `_CATEGORIES["bit"]` in `operator_registry.py` (handlers left in place, unreachable) |
| `ratio_on` | HEAD |

## Result

| | |
|---|---|
| Wins / losses / ties (`ratio_on` more edges) | 9 / 11 / 0 |
| Exact sign test, two-sided | p = 0.82 |
| Wilcoxon signed-rank | p = 0.42 |
| Median per-seed delta | -8.5 edges |
| Median edges | 617 (`off`), 607.5 (`ratio_on`) |
| Crashes summed over 20 seeds | 79 (`off`), 76 (`ratio_on`) |

Per-seed deltas: -37, 15, 71, -68, -209, 112, -20, 9, 81, -158, 92, -136, -81, 82, -1,
-581, 4, 18, -16, -31. Seed 16 (1160 vs 579 edges, `off` bimodal high) carries the mean
to -42.7; the median and both tests are not driven by it.

## Control

Same tree vs a copy of itself, seeds 1-20: 0 wins, 0 losses, 20 ties. The harness is
deterministic per seed (seed 1 gave 675 / 638 edges on all three reps in a pilot), so
the A/A has no spread. The spread in the A/B (sd of the deltas about 150 edges) is the
arm changing the RNG stream plus whatever the four extra operators do. That means the
A/B cannot separate the two. This is why the verdict is NULL and not LEAN: no
effect is claimed in either direction.

## Per-arm selection (10 seeds x 10k execs, `ratio_on`, `--stats-file`)

123,422 operator picks over 265 operators. `op_success` is the fuzzer's own tally.

| arm | picks | share | success | rate |
|---|---|---|---|---|
| `ratio_flip_e5` | 637 | 0.52% | 171 | 26.8% |
| `ratio_flip_e4` | 684 | 0.55% | 178 | 26.0% |
| `ratio_flip_e3` | 619 | 0.50% | 150 | 24.2% |
| `ratio_flip_e2` | 658 | 0.53% | 85 | 12.9% |
| all operators | 123,422 | | 24,510 | 19.9% |

Uniform share over 265 operators would be 0.38%. All four arms sit near 0.5%, so the
scheduler does not converge on one decade here. The success rate does separate them:
`e5..e3` beat the all-operator rate, `e2` (1e-2..1e-1, the densest decade) falls below it. On these small JSON seeds `e5` and `e4` mostly hit the
`MIN_FLIPS = 1` floor, so they behave like `bit_flip`.

## Power and limits

- SE of the median delta is about 150 / sqrt(20) = 34 edges, so a true effect inside
  about +-70 edges (+-11%) is not excluded. Not NULL-BOUNDED.
- One target, one budget, small seeds. Length scaling is the claim of the port; fuzzgoat
  inputs are short, so it is barely exercised. A format target with long inputs
  (png, jpeg, ffmpeg wav) is where the claim can show.
- `e5`/`e4` are floor-dominated here; a dead-arm decision needs the long-input run.
- Open item (c) in `docs/TODO.md` (per-seed fixed ratio, as zzuf does) is untouched.

## Reproduce

Arm trees: copy `src/` twice; delete the four `ratio_flip_e*` lines from the bit band in
`operator_registry.py` in one copy. Then
`python tools/benchmark.py replicated --set fuzzgoat --targets fuzzgoat_read_noasan.so --seeds 1,...,20 --reps 1 --iters 10000 --src-a <off> --src-b <on>`.
