# Handover — Porting from microsoft/rusty-radamsa

Date: 2026-10-04
fuzzer HEAD: `2b51500`
rusty-radamsa HEAD: `9dd9b1f`
Status: analysis only. Nothing implemented, nothing measured except §2.1 simulation.

## 1. Summary

Most of rusty-radamsa is already in the fuzzer. Four ports are worth doing. One upstream bug must not be copied.

| # | Port | Target | Value | Cost | Needs benchmark |
|---|------|--------|-------|------|-----------------|
| 1 | Matched-position jump | `_op_fuse_this`, `_op_fuse_next`, `_op_fuse_old` | High | Low | No |
| 2 | Cross-input line ops (`lis`, `lrs`) | `_LINE_MUTATORS` | Medium | Low | No |
| 3 | Decaying mutation density (`INITIAL_IP`) + burst locality (`bu`) | position arena arm / stack-depth chooser | Unknown | Medium | Yes |
| 4 | Random per-input score reset | op scheduler baseline arm | Low | Low | Yes |

Order: 1, 2, then 3 and 4 after the paired benchmark exists.

## 2. Findings

### 2.1 Fuse ops pick positions blindly

Radamsa `fuse.rs::find_jump_points` jumps between positions with a shared prefix, so both sides of the splice look alike. Narrows by first element, iteratively, until fuel (`SEARCH_FUEL=100000`) or a 1/`SEARCH_STOP_IP` (8) stop.

`_op_fuse_this` (`services/operators.py`): two uniform positions `p1`, `p2`, then computes the common prefix of up to 16 bytes. No search for a matching `p2`.

Simulation (`/home/claude/jump.py`, 400 files from the fuzzer repo, 392 text, 8 binary):

| Metric | Text | Binary |
|--------|------|--------|
| Draws where first bytes of `p1`/`p2` differ (prefix length 0, the fallback branch) | 93.3% | 92.7% |
| Positions with a 2-byte twin elsewhere in the buffer | 96.3% | 97.7% |

Reading: ~93% of `fuse_this` applications splice at unrelated positions. A matched partner exists for ~96% of positions. Caveat: sample is mostly source text; run on real corpora (png, ffmpeg containers) before claiming the same for binaries. 8 binary files is not a sample.

`_op_fuse_next`: split point is `randint(1, min(len)//2)`, same offset in both buffers, no matching. Behaves like `crossover`.

`_op_fuse_old`: both split points uniform. Also keeps `_fuse_memory` as lazy `hasattr` state (not persisted, not in `--resume`).

### 2.2 Line ops are intra-buffer only

`_op_line_mutate` docstring says "Insert a line from elsewhere in the buffer". `_LINE_MUTATORS` = del, dup, swap, perm, repeat, clone. `clone` is intra-buffer. Radamsa also has `lis` (insert a line from another input) and `lrs` (replace a line with one from another input). No donor-based line op exists.

### 2.3 Mutation density and bursts

Radamsa `patterns.rs`:
- `INITIAL_IP = 24`. Per block: mutate with probability 1/ip, where `ip` starts at `rands(24)` and increments by 1 after each mutation (`mutate_once`). Geometric fall-off in mutation count.
- `REMUTATE_PROBABILITY = 0.8`. `nd` repeats `mutate_multi` while a 0.8 coin hits. `bu` forces at least 2 passes.
- Effect of `bu`: several mutations land in the same neighbourhood.

Fuzzer havoc stack depth is chosen in `services/operators.py` (~line 5036, `randint_list(8,16)` / `(2,8)`). Position choice is the position arena (`services/position_arena.py`). No arm models "several edits close together" or decaying density. `pos_burn_front` is heat-based locality, not the same mechanism.

### 2.4 Radamsa mutator scoring

`mutations.rs::randomize()`: each mutator score ← uniform in [2,10] (`MIN_SCORE`, `MAX_SCORE`), weight = `rands(priority*score)`, sorted ascending, popped from the end. If the output equals the input, falls through to the next mutator.

### 2.5 Upstream defects — do not copy

1. `adjust_priority`: `max(MIN_SCORE, max(MAX_SCORE, pri+delta))`. Floor is effectively 10, no ceiling. Scores can only rise; feedback is nullified. Upstream radamsa uses `min(MAX, max(MIN, ...))`.
2. `ascii.rs::mutate_text_data`: inserts per byte with `Vec::insert`. O(n·m) for up to 65,536 newlines.

### 2.6 Already covered / out of scope

Covered: bit/byte/seq ops, `radamsa_num`, `utf8_widen`/`utf8_insert`, `tree_mutate`/`tree_generate`, line ops, fuse ops, `special_strings` (≈ `ab`), exec dedup (bloom/cuckoo), Elo + other schedulers.

Not worth porting:
- `str`, `word`, `xp`: marked incomplete upstream.
- Generators `tcp`, `udp`, `jump`, `pcapng`; outputs `tcp*`, `udp*`, `template`, `hash`; CRC82/CRC64 digests. Fuzzer runs targets through the shim. Revisit only if a network-target mode is added.
- Block streaming (256 B – ~4 KB blocks): whole seeds already in memory.

## 3. Implementation plans

Rules that apply (AGENTS.md): 12 (register only in `operator_registry.REGISTRY`), 16 (`rand_pool` PRNG), 23 (one falsification + one adversarial test each), 28 (constants), 29 (early return), 30 (names <30 chars), 36 (surgical), 38 (TDD), 39 (scripted RNG in tests, no retry loops), 14 (vectorize after correctness).

### 3.1 Port 1 — matched-position jump

Add a helper in `core/mutations/generic.py` (next to `splice_common_prefix`):

```
def jump_pair(buf, rng, k_range) -> tuple[int, int] | None
```

- Build `{buf[i:i+k]: [i, ...]}` for one `k` drawn from `k_range` (2..4). Cap index size on large buffers (sample positions; do not scan 64 KB per call). Candidate: sample up to `MAX_JUMP_SAMPLES` start positions.
- Pick `p1` uniform, `p2` uniform from `index[buf[p1:p1+k]]` minus `p1`. If none: widen to smaller `k`, then return `None`.
- Callers fall back to the current random-pair path on `None`.
- Equal-input case (`fuse_next`/`fuse_old` donor equals buf): alternate jump/land suffix sets as in `alernate_suffixes` so the jump never lands in the same place.

Wiring:
- `_op_fuse_this`: replace `p1`/`p2` draw with `jump_pair`.
- `_op_fuse_next`: pick split in `buf` at `p1`, land at a twin of `buf[p1:p1+k]` in `other`; fallback to current logic.
- `_op_fuse_old`: same against the remembered block.
- Keep the `max_len` clamps and the explicit `buf[:] = ...` semantics (see the in-place growth note in `_op_fuse_this`).
- Consider a flag `--fuse-match` (off by default) until measured; otherwise this changes existing operator behaviour. Decide with the user.

Tests:
- Scripted RNG, buffer with known twin positions: assert the exact output.
- Falsification: buffer with no repeated k-gram → returns `None`, old path output unchanged.
- Adversarial: all-same-byte buffer (every position is a twin, `p1 == p2` risk), 1-byte donor, empty donor, 64 KB buffer (time bound on index build), result length ≤ `max_len` and ≤ 2×len.

### 3.2 Port 2 — `lis` / `lrs`

In `services/operators.py` add `_line_ins_other(parts, rng, donor)` and `_line_rep_other(...)`. `_LINE_MUTATORS` entries take `(parts, rng)`, so these need donor access: either bind via closure in `_op_line_mutate` or extend the signature for all six (the latter touches unrelated lines; prefer a separate small dict for donor modes).

- Donor from `self._donor(data, corpus, rng)` (as in `_op_fuse_next`). Split on `b"\n"`, pick a non-empty line.
- `_op_line_mutate` `rng.choice([...])` gains two names. This shifts the mode distribution of the existing op; flag it in the commit message.
- Guard: donor has <1 line or is `data` itself → return (no change).

Tests: scripted RNG, exact output; falsification: single-line donor with empty line; adversarial: corpus of one entry, binary donor with no `\n`, donor line pushing past `max_len`.

### 3.3 Port 3 — density / burst

Two independent pieces:

A. Density chooser: `ip = rands(24)`, per-edit probability `1/ip`, `ip += 1` per edit. Use as the havoc stack-depth sampler alternative (not a replacement).
B. Burst arm: after the first edit position `p`, draw the next positions from a window around `p` (radamsa uses block adjacency). Candidate: `pos_burst` in `core/schedulers/`, following `select_pos`/`record` conventions of `pos_burn_front.py` / `pos_levy.py`. Register as an arena arm with `--pos-arena-arms` support; persist state via `state_store` if any.

Blocker: no paired benchmark yet (handover `position_arena` gap; `tools/lib/bench_paired.py` has `pos-arena-*` arms from `--pos-arena-arms`). Run uniform vs arena vs arena+new arm on fuzzgoat before merging. Do not add a third unmeasured arm.

### 3.4 Port 4 — random score reset baseline

New op scheduler `OpRadamsaScheduler` in `core/schedulers/`, registered in `_OPERATOR_STRATEGY_NAMES`, implementing `select_op`/`record`/`bandit_stats`. Score ∈ [2,10] redrawn per input; weight = `rands(score)`. Use the correct clamp `min(MAX, max(MIN, ...))` if feedback is added. Purpose: baseline arm for Elo comparisons, not expected to win. Needs the benchmark.

## 4. Open decisions (user)

1. Port 1 default-on or behind `--fuse-match`.
2. Port 2: accept the changed mode distribution of `_op_line_mutate`.
3. Whether to build the arm-subset benchmark first (unblocks 3 and 4).
4. Real-corpus check of §2.1 numbers (png, ffmpeg containers) before merging Port 1.

## 5. Not verified

- §2.1 numbers are from repo source files; binaries n=8.
- No test or benchmark run on the fuzzer; no code changed.
- Radamsa behaviour read from `rusty-radamsa` source only, not run.
