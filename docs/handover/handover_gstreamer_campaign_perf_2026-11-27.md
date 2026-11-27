# GStreamer campaign: performance bugs and base-seed reachability

**Date:** 2026-11-26/27
**Target:** `gstreamer_read_1.28.9_{asan,noasan}.so` (vendored, `tools/build_targets.sh`)
**Status:** 0 crashes in ~1.02M execs. Seven fuzzer bugs found and merged (#470, #476-#481). Base-seed experiment done.

## 1. Campaigns

| Run | Build | Options | Execs | Edges (shm) | eps end | Crashes |
|---|---|---|---|---|---|---|
| 1 | asan | defaults | 116k | 14.0k | 73 | 0 |
| 2-3 | asan | `--resume` | +58k | 14.3k | 41 | 0 |
| 4 | noasan | `--resume` | +410k | 15.5k | 114 | 0 |
| 5-8 | noasan | `--resume --elo all` | +177k | 14.9k | 12 -> 25 | 0 |
| 9 | noasan | fresh, `--elo all` | 102k | 15.4k | 28 | 0 |
| 10 | noasan | empty corpus, `--elo all`, `-m 8192` | 157k | 14.9k | 44 | 0 |

All runs: `-D mkv.dict riff.dict isomp4.dict`, `--push-on-crash`. Corpus `~/fuzzing/corpus_gstreamer/` (seeds from `tools/gen_ffmpeg_gst_seeds.py --target gstreamer`, 9 MB LZO seed excluded). Coverage plateaus near 15.4k edges.

## 2. Bugs fixed

Profiling: `py-spy record -p <pid> -d 90 -r 50 -f raw` on the live run.

| PR | Bug | Effect | Fix |
|---|---|---|---|
| #470 | Virtual clock ticks not persisted; `--resume` restarts at 0 | Restored `added_at` in the future; age -100 s zeroed `1 + age*0.01`; `_cdf_pick` raised, run aborted at 192 s | `clock_ticks` saved/restored (`Clock.resume`, never rewinds) |
| #476 | Katz recompute budget measured on the virtual clock | Rebuild cost read 0 s; rebuilt every 50 execs; 47.5% of wall | `KatzChannel` built on host time |
| #477 | `seq_mutate._id_shift` cubic (`substr in m` per window per message) | 84 s per call on a 64 KiB seed LF-split into 15.8k messages; 83% of wall | 4-gram document frequency (py < 512 B, numpy above); identical draws |
| #478 | MC null replicates bounded by a clock deadline | Virtual clock: all 40 replicates every refit, 19.4% of wall; wall clock: count tracks load, `--seed` diverges | Work budget `positions x (256 + obs)`, clamped [8, 40]; no clock |
| #480 | Shim vs glibc C23 `__glibc_const_generic` macros | `memchr`/`strstr`/`strpbrk` definitions fail to parse; master CI 127 red | `#undef` before the interposers |
| #481 | Edge tracker restored on target-hash mismatch | Old build's edge IDs kept with no holder: 42.6k tracked vs 15.2k live | Restore only on match/unknown |
| #481 | Minimize hashed every corpus seed with pure-Python XXH64 | 13.7% of wall; 1,118 ms per pass at 500 x 64 KiB | Memoized `CorpusManager.seed_key` |

Fuzzer overhead under `--elo all`: ~66 ms/exec -> ~17 ms/exec. Remaining profile (run 9): target 33%, Elo seed pick 18.6% (spread: weighted 5.7, lineage diversity 4.9, Katz 4.1, LCA 3.7), `_credit_ops` 10.9%, mutate 10.5%, `MutualInformationTracker.record` ~6.5%.

Pattern behind #470/#476/#478: the virtual decision clock reads 0 s for any work between ticks. Any budget on it is disabled. Audit remaining `clock_of(...)` budget/deadline readers.

## 3. Base-seed reachability experiment

Question: from an empty corpus, does the fuzzer reach the base seeds?

- 8 base seeds generated locally with ffmpeg (no download): mp4, mov, 3gp, mkv, webm, wav, avi, flv; 763 B-6.2 KB.
- Hashes (`hash_data`) in `~/fuzzing/experiments/gst_reach/base_hashes.txt`.
- 1 h campaign, empty corpus (run 10). Coverage via `fuzzer-tool edge-dump --inprocess-direct --inprocess-func fuzz_shm_run`.

**Hash test: 0/8.** No byte-identical file among 2,634 kept.

**Coverage test: 77.3% of base-seed edges reached** (7,879 / 10,188).

| Seed | Covered |
|---|---|
| wav | 96.9% |
| flv | 96.7% |
| webm | 90.0% |
| mkv | 89.0% |
| avi | 83.3% |
| mp4 / mov / 3gp | 80.8 / 80.7 / 80.2% |

Edge sets: both 7,879; base-only 2,309; fuzzed-only 6,883.

Reading: every container is found from nothing; the MP4 family lags (nested atoms with consistent sizes/offsets). Base-only edges are likely valid-file paths (track setup, sample tables) — not verified per function. Fuzzed-only edges are presumably error paths. **Conclusion: base seeds are needed; add them to the GStreamer corpus.** Artifacts: `~/fuzzing/experiments/gst_reach/` (`report.txt`, both edge pickles, `campaign.log`). Analysis script was ad hoc (not committed).

## 4. Open items

- Edge-keyed state still crosses builds on `--resume`: `sensitivity`, `crash_mi`, Katz masks, per-seed `coverage_edges` (TODO).
- MC null: 64 KiB inputs still pay the 8-replicate floor per refit (TODO).
- `split_msgs` LF mode accepts binary media (15.8k "messages" from one seed); consider rejecting high-entropy input (TODO).
- `_bulk` pure-Python XXH64 still runs once per admitted seed (TODO).
- `--elo all` seed pick 18.6%: plain `--elo` likely cheaper; not measured.
- RSS grows with corpus (1.2 GB -> 3.6 GB over 1 h at 2k -> 3.6k seeds); bounded so far.
- Shim: recheck if it ever interposes `strchr`, `strrchr`, `bsearch`, `wcs*` (TODO).
- Local env: 3 pre-existing shim-test failures here (`test_cmplog_simd_vcmp` plugin build, `test_memfd_input` control, gcc warning-free private-heap); same with the pre-#480 shim.
- Container setup: `libclang-rt-18-dev` was missing (no ASAN/sancov runtime); installed by hand, not persistent.
- Next: seed corpus with the 8 base files (+ real MP4/FLV variants), long noasan run with `--elo`.
