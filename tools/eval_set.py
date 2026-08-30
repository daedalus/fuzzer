"""LOCKED evaluation set for paired benchmarking.

Frozen once so arms cannot be tuned against a moving target. Changing
anything here invalidates comparison against results recorded before the
change -- add a new named set instead of editing an existing one, and say
in the commit which set a number came from.

Why a fixed matrix rather than "run the bench script again": the existing
harnesses (``tools/bench.sh``, ``tools/bench_sweep.sh``) run one unseeded
campaign per arm on ``targets/png_read`` and compare raw edge counts. That
is a single sample of a stochastic process, so the difference between two
arms is mostly the difference between two draws. A locked ``(target, seed)``
matrix makes the comparison *paired*: every arm sees exactly the same
starting conditions, and the unit of analysis becomes the per-cell
win/loss rather than the aggregate.

Selection criteria for the targets:

* mixed structure. ``png_read`` and ``jpeg_read`` are chunked/segmented
  formats where dictionary and grammar operators matter; ``zlib_read``
  and ``lz4_read`` are compressed streams where they do not; ``grep_read``
  is text; ``secp256k1_read`` is dense fixed-width equality and range
  checks over keys and signatures; ``cmplog_exercise`` is a synthetic
  comparison ladder that isolates the cmplog operators these ports touch.
* buildable from ``tools/build_targets.sh``. png/jpeg/zlib/grep need only
  system libraries; lz4 and secp256k1 need ``tools/vendor_lz4.sh`` and
  ``tools/vendor_secp256k1.sh`` to have been run first. A cell whose
  binary is missing is skipped by the harness, not silently scored zero.
* not saturated at the default budget. A target every arm solves
  completely measures nothing; see the p-bit study's note about retiring
  its first semiprime set once the leading arm hit 97%.
"""

from __future__ import annotations

# ── Target sets ────────────────────────────────────────────────────────

# Default: broad, seven targets. ~7 x n_seeds x n_arms campaigns.
LOCKED_TARGETS: list[tuple[str, str]] = [
    # (target path, extra flags)
    ("targets/png_read", "-D dictionaries/png.dict"),
    ("targets/jpeg_read", "-D dictionaries/jpeg.dict"),
    ("targets/zlib_read", ""),
    ("targets/lz4_read", ""),
    ("targets/grep_read", ""),
    ("targets/secp256k1_read", ""),
    ("targets/cmplog_exercise", "--cmplog"),
]

# Focused: for arms that only touch the cmplog operand-matching path
# (gradient_descent, climb_hill, magic_byte_search). Running the broad set
# for those spends most of the budget on targets the change cannot reach,
# which dilutes the paired test rather than strengthening it.
CMPLOG_TARGETS: list[tuple[str, str]] = [
    ("targets/cmplog_exercise", "--cmplog"),
    ("targets/png_read", "--cmplog -D dictionaries/png.dict"),
    ("targets/zlib_read", "--cmplog"),
    # secp256k1 parses fixed-width keys and signatures through dense
    # equality and range checks, so it is close to a natural cmplog
    # benchmark: almost every branch the fuzzer needs is an operand match.
    ("targets/secp256k1_read", "--cmplog"),
]

# direct_lite: the same targets as `locked`, as .so rather than executables.
#
# A new set rather than an edit to `locked`, per the rule at the top of this
# file: results are not comparable across the two, and every number quoted
# from here must say so.
#
# The reason it exists is that `locked` cannot resolve any arm whose effect
# depends on per-seed execution cost. Measured on png_read, mean cost per seed
# across a campaign's corpus:
#
#     targets/png_read     (subprocess)   p90/p10 1.06x   CV 0.023
#     targets/png_read.so  (direct_lite)  p90/p10 4.32x   CV 1.455
#
# ~9 ms of fork/exec dominates ~0.2 ms of decode, so in subprocess mode every
# seed costs the same to within noise and the signal is gone before any arm
# sees it. An arm read against `locked` would come back "no difference" whether
# or not it has one -- a false negative, not a null result.
#
# Cost is also why the whole matrix is affordable here: 22 s per cell against
# 217 s, a 10x speedup, because process spawn was most of the old budget.
#
# -m 65536 on the format targets: the 4096 default truncates away exactly the
# large inputs whose decode cost varies, so the default is not a neutral
# choice for a cost-sensitive arm -- it is one that suppresses the quantity
# under test. Measured on png_read.so, raising the cap moves p90/p10 from
# 2.00x to 4.32x. The compressed-stream targets keep the default.
DIRECT_LITE_TARGETS: list[tuple[str, str]] = [
    ("targets/png_read.so", "-D dictionaries/png.dict -m 65536"),
    ("targets/jpeg_read.so", "-D dictionaries/jpeg.dict -m 65536"),
    ("targets/zlib_read.so", ""),
    ("targets/lz4_read.so", ""),
    ("targets/grep_read.so", ""),
    ("targets/gzip_read.so", ""),
]

# direct_lite_signal: the three targets of `direct_lite` that can carry a
# result, meant to be run with replicated cells (bench_paired --reps).
#
# Again a new set, not an edit, per the rule at the top of this file.
#
# `direct_lite` fixed the flat-cost problem that disqualified `locked`, but
# it left a second one in place: it assumed a cell is a fixed function of
# its seed. It is not. The fuzzer carries wall-clock state -- `age = now -
# added_at`, and `mean_exec` itself -- so re-running one cell under the
# *same* arm does not reproduce its edge count. Measured, same arm, same
# seed, 5 replicates each:
#
#     target          spread                CV
#     png_read.so     45-63 / 39-66         0.118 / 0.189
#     jpeg_read.so    201-206 / 196-209     0.010 / 0.026
#     grep_read.so    792-803 (3 reps)      0.007
#     zlib_read.so    12 flat               0
#     lz4_read.so     12 flat               0
#     gzip_read.so    36 flat               0
#
# Two separate problems are visible there.
#
# Half the set carries no information at all. zlib, lz4 and gzip are
# bit-for-bit reproducible and saturated -- 12, 12 and 36 edges on every
# replicate of every seed. They cannot produce a discordant pair for any
# arm, so their 60 cells per arm are pure cost. The handover predicted
# zlib and gzip would be no-ops from the flat-cost identity; the real
# reason is stronger and applies to lz4 too, which that argument missed.
#
# The rest is noisy enough to matter. Across replicate pairs of the *same*
# arm, 46% differ -- each one a pair McNemar would have scored as a win or
# a loss had the two replicates carried different arm labels. That does not
# invalidate the test: arm assignment is independent of the noise, so the
# discordant pairs still split 50/50 under the null and the type I error is
# controlled. What it destroys is power. One replicate per cell on png puts
# a ~7-edge standard deviation against an unknown and probably smaller arm
# effect.
#
# Note what cannot be done about it. The obvious repair is to virtualise the
# clock so cells reproduce exactly, and it is unavailable: `effective_fuzz_count`
# is `total_time / mean_exec`, which under constant per-execution cost reduces
# to `fuzz_count` exactly -- the same identity that disqualifies `locked`. A
# deterministic clock would collapse the two arms into the same computation.
# The timing variance *is* the quantity under test, so the only way to buy
# power is replication.
#
# Hence: drop the three saturated targets, and spend the freed budget on
# replicates of the three that move. grep carries its own caveat -- it is the
# best cost-signal carrier in the set now that it runs its engines in-process
# (753 edges of real DFA work, cost varying with pattern complexity) but it is
# ~250 s per cell against ~22 s, so it is priced in seeds rather than reps.
DIRECT_LITE_SIGNAL_TARGETS: list[tuple[str, str]] = [
    ("targets/png_read.so", "-D dictionaries/png.dict -m 65536"),
    ("targets/jpeg_read.so", "-D dictionaries/jpeg.dict -m 65536"),
    ("targets/grep_read.so", ""),
]

TARGET_SETS: dict[str, list[tuple[str, str]]] = {
    "locked": LOCKED_TARGETS,
    "cmplog": CMPLOG_TARGETS,
    "direct_lite": DIRECT_LITE_TARGETS,
    "direct_lite_signal": DIRECT_LITE_SIGNAL_TARGETS,
}

# ── Seeds ──────────────────────────────────────────────────────────────

# Campaign seeds. 20 is enough to resolve a ~10-point difference in per-cell
# win rate on the broad set (120 paired cells); it is not enough for a
# 2-point difference, and the harness prints the achieved cell count so
# that is visible rather than assumed.
SEEDS: list[int] = list(range(20))

# Execution budget per campaign, in execs. Budget is counted in execs and
# not wall-clock on purpose: an arm that is slower per exec would otherwise
# be penalized twice, and an arm that is faster would bank the difference
# as extra attempts. Wall-clock is recorded separately so a cost-aware
# comparison is still possible after the fact.
DEFAULT_ITERS = 10000


def cells(target_set: str = "locked", seeds: list[int] | None = None):
    """Yield every ``(target, flags, seed)`` cell in the matrix."""
    targets = TARGET_SETS[target_set]
    for target, flags in targets:
        for seed in seeds if seeds is not None else SEEDS:
            yield target, flags, seed
