# A/B ledger

Every paired run's result, one row each. Append-only (Hard Rule 59).
Owed runs: `docs/AB_PENDING.md`. Mechanism detail: `docs/DEEP_DIVE.md`.

## Row rules

- One row per arm-vs-base comparison per run. Oldest first; add at the bottom.
- Never edit a past row's numbers. A rerun is a new row whose Result names the row it supersedes (`supersedes #N`).
- Record every outcome: win, loss, null, aborted, contaminated. A run not in the ledger did not happen.
- Result: W/L(/T), test and p, median Δ on the stated metric. Interim splits are not results.
- A/A: the control's W/L and p (Hard Rule 46), or `none` — then the verdict cannot be WIN or LOSS.
- Source: the doc holding the full write-up; it must exist.
- Verdict, one of:
  - `WIN` / `LOSS` — significant, outside the A/A spread.
  - `NULL-BOUNDED` — null with a stated power bound.
  - `NULL` — inside the A/A spread or n.s.
  - `LEAN+` / `LEAN-` — n.s. point estimate one way; rerun owed.
  - `EXPLORATORY` — no threshold frozen beforehand; cannot adopt.
  - `VOID` — aborted, contaminated or broken oracle.

## Entries

| # | Date | Arm vs base | Target · design | Result | A/A | Verdict | Source |
|---|---|---|---|---|---|---|---|
| 1 | 2026-08-30 | cost Boltzmann energy vs count | png_read.so · 10 seeds × 3 reps × 10k, time-paired, `--lock-single-thread` | 3/6/1, McNemar 0.508, med −2.0 edges | noise probe, CV 0.12–0.19 | NULL-BOUNDED (< ~5 edges) | `docs/learnings/2026-08-30-boltzmann-ab-result.md` |
| 2 | 2026-08-30 | cost Boltzmann energy vs count | jpeg_read.so · 10 seeds × 3 reps × 10k | 2/5/3, McNemar 0.453, med −1.5 | noise probe, CV 0.01–0.03 | NULL-BOUNDED (< ~5 edges) | `docs/learnings/2026-08-30-boltzmann-ab-result.md` |
| 3 | 2026-08-30 | cost Boltzmann energy vs count | grep_read.so · 20 seeds × 3 reps × 10k | 12/7/1, McNemar 0.359, med +3.0 | noise probe, CV 0.007 | NULL-BOUNDED (< ~10 edges) | `docs/learnings/2026-08-30-boltzmann-ab-result.md` |
| 4 | 2026-09-20 | `--shaped-reward` vs `--elo` | fuzzgoat noasan clang-scov · 12 seeds × 2k | 5/7, McNemar 0.774, med −7.0 | none | NULL | `docs/learnings/2026-09-20-shaped-reward-ab-result.md` |
| 5 | 2026-09-20 | `--shaped-reward-floor 0.25` vs `--elo` | fuzzgoat noasan · 12 × 2k | 5/7, McNemar 0.774, med −2.5 | none | NULL | `docs/learnings/2026-09-20-shaped-reward-ab-result.md` |
| 6 | 2026-09-22 | `--continuum-reward` vs `--elo` | fuzzgoat noasan · 12 × 2k | 4/8, McNemar 0.388, med −3.5 | none | NULL | `docs/learnings/2026-09-22-continuum-reward-ab-result.md` |
| 7 | 2026-09-25 | `--dirichlet-alpha learned` vs fixed | fuzzgoat · 3k execs | 3/5, n.s., med −13.5 | none | LEAN- | `docs/TODO.md` (dirichlet/dict-thompson) |
| 8 | 2026-09-25 | `--dict-thompson` vs off | fuzzgoat · 3k execs | 2/5/1, n.s., med −8 | none | LEAN- | `docs/TODO.md` (dirichlet/dict-thompson) |
| 9 | 2026-09-26 | `pos-fibonacci` vs `baseline` | png_read_noasan · 20 × 10k | 8/12, McNemar 0.50, med −8; 5/20 collapse cells (bin clamp bug) | none | VOID | `docs/TODO.md` (pos_fibonacci) |
| 10 | 2026-09-26 | `pos-fibonacci` vs `baseline` (after 42ac183) | png_read_noasan · 20 × 10k | 9/11, McNemar 0.82, med −2; supersedes #9 | run-vs-run med −5 | NULL | `docs/TODO.md` (pos_fibonacci) |
| 11 | 2026-09-26 | `pos-fibonacci` vs `pos-round-robin` (after 42ac183) | png_read_noasan · 20 × 10k | 12/7, McNemar 0.36, med +5 | run-vs-run med −9.5 | NULL | `docs/TODO.md` (pos_fibonacci) |
| 12 | 2026-09-26 | `pos-round-robin` vs `baseline` (after 42ac183) | png_read_noasan · 20 × 10k | 11/9, McNemar 0.82 | run-vs-run med −5 | NULL | `docs/TODO.md` (pos_fibonacci) |
| 13 | 2026-09-26 | `--splice-donor gravity` vs uniform | fuzzgoat; container_signal | 2/3; 20/19/1, p=1. Fit never refit (< `MIN_POSITIVES`) | none | NULL | `docs/TODO.md` (gravity splice) |
| 14 | 2026-09-27 | `elo-alphabeta` vs `elo-mcts` | png_read_noasan · 20 × 10k | 12/8, McNemar 0.50, med +8 (< rep noise 10) | within-cell range med 10 | NULL | `docs/learnings/2026-09-27-alphabeta-vs-mcts-png.md` |
| 15 | 2026-09-27 | `elo-op-minimax` vs `elo` | png_read_noasan · 20 × 10k, 3 shards on 4 cores | 10/10, McNemar 1.0, med 0 | none | NULL | `docs/learnings/2026-09-27-minimax-phases-png.md` |
| 16 | 2026-09-27 | `wall-order` vs `baseline` | png_read_noasan · 20 × 10k | 6/14, McNemar 0.115, med −4.5 | none | LEAN- | `docs/learnings/2026-09-27-minimax-phases-png.md` |
| 17 | 2026-09-27 | `minimax-select` vs `minimize-2k` | png_read_noasan · 20 × 10k | 10/10, McNemar 1.0, med −0.5 | none | NULL | `docs/learnings/2026-09-27-minimax-phases-png.md` |
| 18 | 2026-09-27 | `inline-8bit-counters` vs `trace-pc-guard` | fuzzgoat asan · 10 × 5k, oracle replay | 1/8, McNemar 0.04, med −23.5 oracle edges | 5/5, p=1.0 | LOSS | `docs/DEEP_DIVE.md` §Other SanitizerCoverage modes |
| 19 | 2026-09-27 | `inline-bool-flag` vs `trace-pc-guard` | fuzzgoat asan · 10 × 5k, oracle replay | 1/9, McNemar 0.02, med −38.5 | 5/5, p=1.0 | LOSS | `docs/DEEP_DIVE.md` §Other SanitizerCoverage modes |
| 20 | 2026-09-27 | `+trace-loads,trace-stores` vs `trace-pc-guard` | fuzzgoat asan · 10 × 5k | 6/4, McNemar 0.75, med +6.5 | 5/5, p=1.0 | NULL | `docs/DEEP_DIVE.md` §Other SanitizerCoverage modes |
| 21 | 2026-09-27 | `+pc-table` vs `trace-pc-guard` | fuzzgoat asan · 10 × 5k | 7/3, McNemar 0.34, med +10.5 | 5/5, p=1.0 | NULL | `docs/DEEP_DIVE.md` §Other SanitizerCoverage modes |
| 22 | 2026-09-27 | COMPCOV L2 vs L0 | png_read_tracecmp (hand-built) · 10 × 10k, real-edge replay | 7/3, McNemar 0.34 | 7/3, med +96 | NULL | `docs/TODO.md` (COMPCOV review leftovers) |
| 23 | 2026-09-27 | COMPCOV L2 vs L0 | ffmpeg_read_nosan · 12 × 2k, real-edge replay | 5/7, McNemar 0.77; corpus +70% (12/0, p=0.0005); eps −5% | 3/9 | NULL (edges); corpus bloat | `docs/TODO.md` (COMPCOV review leftovers) |
| 24 | 2026-10-03 | forkserver vs `--no-forkserver` | fuzzgoat asan · 10 × 90 s | 10/0, sign 0.002, med +53 edges, 2.30× execs | 2/4, p=0.69 | WIN | `docs/DEEP_DIVE.md` (forkserver, cmplog FIFO) |
| 25 | 2026-10-03 | `--entropic-seed` vs default | fuzzgoat | 9/7, p=0.80, med +3 | none | NULL | `docs/DEEP_DIVE.md` §Entropic seed arm |
| 26 | 2026-10-07 | `--consolidated-prior blup` vs fixed | fuzzgoat noasan · 6 × 3k | 4/2, mean +5.7 | FIXED rerun −42..+35 | NULL | `docs/DEEP_DIVE.md` (consolidated BLUP) |
| 27 | 2026-10-14 | pattern tables vs table commit reverted | fuzzgoat nosan clang · 8 seeds × 10k, order alternated | 6/2, sign 0.29, med +4.5 (bimodal ~460/~970, sd Δ 277) | none | NULL | `docs/handover/handover_pattern_gaps_2026-10-12.md` |
| 28 | 2026-10-15 | `-M 1` vs `-M 8` | fuzzgoat asan · 20 × 20k | 19/1, sign < 0.001, med +94.5; eps 278 → 480 | 8/12, p=0.50 | WIN (default now 1) | `docs/DEEP_DIVE.md` (`-M` default 8 -> 1) |
| 29 | 2026-10-15 | `-M 1` vs `-M 8` | png · 20 × 20k | 9/10, p=1.0, med −0.5; eps 175 → 384 | 11/9, p=0.82 | NULL (edges); WIN (eps) | `docs/DEEP_DIVE.md` (`-M` default 8 -> 1) |
| 30 | 2026-10-16 | `gradient_descent` NUMERIC vs base | ffmpeg 9.0.2 asan · 8 × 3k | med +1056, sign-flip 0.29 | med +902, p=0.77 | LEAN+ (~45 seeds owed) | `docs/TODO.md` (gradient_descent NUMERIC) |
| 31 | 2026-10-17 | `literal_mismatch` on vs off | fuzzgoat noasan · 4 × 20k | 2/2; literal-chain frontier 5 vs 9 | none | NULL (n=4) | `docs/DEEP_DIVE.md` §`literal_mismatch` op |
| 32 | 2026-10-17 | `literal_mismatch` on vs off | FFmpeg 9.0.2 noasan · 4 × 20k | mean −553 edges; both-ways +8 | off-vs-off +54 | NULL | `docs/DEEP_DIVE.md` §`literal_mismatch` op |
| 33 | 2026-10-17 | `--frontier-bin` seed arm vs default | png_read · — | both-ways +2.0, edges −19.4, eps −21% | +1.0 | NULL | `docs/TODO.md` (`literal_mismatch` + frontier) |
| 34 | 2026-10-17 | `--frontier-bin` seed arm, no Elo, vs default | FFmpeg | 0/6, edges −43% | none | LEAN- | `docs/TODO.md` (`literal_mismatch` + frontier) |
| 35 | 2026-10-17 | `--frontier-bin` under `--elo all` vs `--elo all` | FFmpeg | edges −505, both-ways −3, eps −44% | +290 / +62 | NULL | `docs/TODO.md` (`literal_mismatch` + frontier) |
| 36 | 2026-10-17 | `pos_centroid` vs uniform | FFmpeg 5k · 5 reps | 7/3 | 6/4 | NULL | `docs/TODO.md` (`pos_centroid`) |
| 37 | 2026-10-17 | `--edge-clusters` vs default | fuzzgoat · `--reps 3` | 10/10, med +0.0 | 12/8, +7.5 | NULL | `docs/TODO.md` (`--edge-clusters`) |
| 38 | 2026-10-22 | `seed-dpp` vs `baseline` | fuzzgoat · 20 seeds | 5/15, McNemar 0.041 (Holm 0.083), med −23.5 | 8/12, p=0.50 | LEAN- | `docs/TODO.md` (DPP seed arm) |
| 39 | 2026-10-25 | `--stage-execs 8` vs 1 | fuzzgoat asan direct_lite · 3 reps × 20k | wall and edges inside control | control spread | NULL | `docs/DEEP_DIVE.md` §Seed stage |
| 40 | 2026-10-25 | `--stage-execs 8` vs 1 | fuzzgoat nosan exe · 8 reps × 20k | wall 8/8, sign 0.008, −5%; edges p=0.73 | 4/8, p=1.0 | WIN (wall); NULL (edges) | `docs/DEEP_DIVE.md` §Seed stage |
| 41 | 2026-10-25 | `elo-reactive` vs `elo-gradient` | FFmpeg 9.0.2 noasan · 10 × 10k | 7/3, McNemar 0.344, med +679 | 5/5, p=1, med −467 | NULL | `docs/TODO.md` (`--reactive`) |
| 42 | 2026-10-26 | `--novel-crash-reward` vs default | fuzzgoat noasan · 20 × 10k | unique 8/5/7 p=0.58; edges 11/9 p=0.82 | none | NULL | `docs/AB_PENDING.md` (Crash handling) |
| 43 | 2026-10-26 | `--novel-crash-reward` vs default | fuzzgoat noasan · 20 × 50k | unique 8/8/4 p=1; edges 9/11 p=0.82, med −13 | unique 4/6 p=0.75; edges 11/9 | NULL | `docs/AB_PENDING.md` (Crash handling) |
| 44 | 2026-10-27 | `--seed-residual` vs `--elo` | fuzzgoat noasan trace-pc-guard · 24 × 2k | 14/10, McNemar 0.541, med +9 | none | EXPLORATORY · NULL | `docs/learnings/2026-10-27-seed-residual-op-credit-ab.md` |
| 45 | 2026-10-27 | `--op-credit` vs `--elo` | fuzzgoat noasan · 24 × 2k | 15/9, McNemar 0.307, med +42; tail on 8/24 seeds | none | EXPLORATORY · LEAN+ | `docs/learnings/2026-10-27-seed-residual-op-credit-ab.md` |
| 46 | 2026-10-29 | `--novel-crash-reward` (abort) vs `abort-crash` | FFmpeg 9.0.2 `--set ffmpeg` · 3k, 20k | 0 crashes in any arm | none | VOID (no signal) | `docs/AB_PENDING.md` (Crash handling) |
| 47 | 2026-10-29 | `--novel-crash-reward` (abort) vs `abort-crash` | FFmpeg 8.0.3 seeded (`vpk_div0`) · 20 × 10k | unique 20 ties; edges 7/13 p=0.26 (Holm 0.53), med −726 | edges 8/12 p=0.50 | NULL | `docs/AB_PENDING.md` (Crash handling) |
| 48 | 2026-10-29 | `--seed-branch-scheduler` vs `elo` | `fuzzgoat_katz` · 20 × 10k | 14/6, McNemar 0.115 (Holm 0.23), med +124; 9/60 early exits | 11/9, p=0.82 | LEAN+ | `docs/AB_PENDING.md` (Seed scheduling) |
| 49 | 2026-10-29 | `--field-hmm` vs default | fuzzgoat noasan · 20 × 10k | 9/11, McNemar 0.82, med −21.5; eps 753 vs 845 | 9/11, p=0.82 | NULL | `docs/DEEP_DIVE.md` §Viterbi field segmentation |
| 50 | 2026-10-31 | `elo-tang` vs `elo` | fuzzgoat noasan · 20 × 10k | 12/8, McNemar 0.503 (Holm 1.0), med +47.5; eps 3/17 (Holm 0.005), med −38 | none | NULL | `docs/TODO.md` (r1_tang) |
| 51 | 2026-10-31 | `elo-tang-residual` vs `elo` | fuzzgoat noasan · 20 × 10k | 12/8, McNemar 0.503 (Holm 1.0), med +19.5; eps 5/15 (p 0.041), med −41 | none | NULL | `docs/TODO.md` (r1_tang) |
| 52 | 2026-10-31 | `elo-tang-residual` vs `elo-tang` | fuzzgoat noasan · 20 × 10k | 9/11, McNemar 0.824, med −9 | none | NULL | `docs/TODO.md` (r1_tang) |
| 53 | 2026-10-31 | shim plain touched OR while single-threaded vs always atomic | standalone 600-block `.so`, 8192 map, bitmap on, pinned, layout pinned · 40 alternating pairs × min-of-15 | 40/0, sign p<1e-11, med −15.6% ns/exec | 16/40, p=0.27 | WIN | `docs/DEEP_DIVE.md` §First-touch and GOT cuts |
| 54 | 2026-10-31 | protected `__afl_area`/`__afl_prev_loc` vs default visibility | same `.so`, no `-Bsymbolic`, bitmap off, layout pinned · 40 pairs | 23/17, sign p=0.43, med −0.2% ns/exec; callgrind −2.9% Ir | 19/40, p=0.87 | NULL | `docs/DEEP_DIVE.md` §First-touch and GOT cuts |
| 55 | 2026-10-31 | rows 53+54 shim vs master shim | fuzzgoat nosan exe, forkserver, `--touched-scan` · 4 alternating pairs × 20k, `-s 7` | wall 2/2, eps med 911 vs 856 | none | NULL | `docs/DEEP_DIVE.md` §First-touch and GOT cuts |
| 56 | 2026-10-31 | `elo-tang` vs `elo` | FFmpeg 9.0.2 noasan · 10 × 10k, 4 arms in parallel | 7/3, McNemar 0.344 (Holm 1.0), med +676; eps med −1.6 | `elo-tang-ctl` vs `elo-tang` 3/7, p=0.344, med −981 | NULL | `docs/TODO.md` (r1_tang) |
| 57 | 2026-10-31 | `elo-tang-residual` vs `elo` | FFmpeg 9.0.2 noasan · 10 × 10k | 5/5, McNemar 1.0, med +452.5; eps med −2.4 | as #56 | NULL | `docs/TODO.md` (r1_tang) |
| 58 | 2026-10-31 | `elo-tang-residual` vs `elo-tang` | FFmpeg 9.0.2 noasan · 10 × 10k | 6/4, McNemar 0.754 (Holm 1.0), med +437.5; eps 2/8, p=0.109, med −2.2 | as #56 | NULL | `docs/TODO.md` (r1_tang) |
| 59 | 2026-10-31 | rows 53+54 shim vs pre-merge shim (`c1e96ff`) | fuzzgoat clang ASAN `.so`, direct_lite, `--touched-scan` · 4 alternating pairs × 20k, `-s 7` | wall 3/1, eps med 546 vs 580 | none | NULL | `docs/DEEP_DIVE.md` §First-touch and GOT cuts |
| 60 | 2026-10-31 | cfp-keyed ctx-cache hit vs ra-only cache | standalone 600-block `.so`, 20 calls/exec, 8192 map, layout pinned · 40 alternating pairs | 22/18, sign p=0.64, med −0.1% ns/exec; callgrind −0.78% Ir | 17/40, p=0.43 | NULL | `docs/DEEP_DIVE.md` §First-touch and GOT cuts |
| 61 | 2026-10-31 | latched generation tag vs `*__afl_gen_word` per edge (prototype) | same bench | 16/24, sign p=0.27, med +0.5%; callgrind −0.04% Ir | 17/40, p=0.43 | NULL | `docs/DEEP_DIVE.md` §First-touch and GOT cuts |
| 62 | 2026-10-31 | cfp-keyed ctx-cache hit vs master shim | fuzzgoat clang ASAN `.so`, direct_lite, `--touched-scan` · 4 alternating pairs × 20k, `-s 7` | wall 3/1, eps med 600 vs 580 | none | NULL | `docs/DEEP_DIVE.md` §First-touch and GOT cuts |
| 63 | 2026-10-31 | `-fno-jump-tables` vs master | fuzzgoat clang ASAN `.so`, direct_lite · 10 seeds × 10k, arms interleaved, oracle replay (both builds' nosan `.so`) | 3/7, sign 0.34, med −49 oracle edges; vs A/A copy 6/4, p 0.75, med +7.5; wall 4/6, med −2.4 s | 2/8, p 0.11, med −47.5 | NULL | `docs/DEEP_DIVE.md` §Switch folding |
| 64 | 2026-10-31 | `-fno-jump-tables` vs master | FFmpeg 9.0.2 noasan clang `.so`, direct_lite · 10 seeds × 10k, 3 WAV seeds, arms interleaved, oracle replay (both builds) | 4/6, sign 0.75, med −480.5 oracle edges; vs A/A copy 1/9, p 0.02, med −414; vs arm mean 2/8, p 0.11, med −316; fuzz time med +2 s (n.s.) | 6/4, p 0.75, med +499.5 | LEAN- | `docs/DEEP_DIVE.md` §Switch folding |
| 65 | 2026-10-31 | background GDB crash replay vs inline | fuzzgoat clang ASAN `.so`, direct_lite, fresh crash dir · 8 reps × 20k, master/branch/master interleaved | wall 8/0, sign p=0.0078, med −25% (40.1 → 30.0 s), eps 514 → 696; edges 6/8 p=0.29 | 3/8, p=0.73 | WIN | `docs/DEEP_DIVE.md` §Background GDB crash replay |
| 66 | 2026-10-31 | input-free gates every 64 calls + geometric trickle vs per-call gates + per-format coins | fuzzgoat clang ASAN `.so`, direct_lite, fresh crash dir · 8 reps × 20k, master/branch/master interleaved | wall 6/2, sign p=0.29, med 30.9 → 30.7 s; edges 5/8 p=0.73, med 865 → 840 | wall 3/8 p=0.73, edges 4/8 p=1.0 | NULL | `docs/DEEP_DIVE.md` §Availability gate cadence |
| 67 | 2026-10-31 | availability gate cadence + geometric trickle vs master | FFmpeg 9.0.2 noasan clang `.so` (full components), direct_lite, 3 WAV seeds, fresh crash dir · 8 reps × 10k, master/avail/head/master2 interleaved | wall 4/8 vs master, 5/8 vs master2; edges 4/8, 5/8 | 6/8 p=0.29, edges 2/8 | NULL | `docs/DEEP_DIVE.md` §Availability gate cadence |
| 68 | 2026-10-31 | Katz observe/energy only with a consumer vs always (on top of row 67) | FFmpeg 9.0.2 noasan clang `.so` (full components), direct_lite, 3 WAV seeds, fresh crash dir · 8 reps × 10k, master/avail/head/master2 interleaved | wall 4/8, med ratio 1.00; edges 2/8 p=0.29 | 6/8 p=0.29, edges 2/8 | NULL | `docs/DEEP_DIVE.md` §Katz consumers |
| 69 | 2026-10-31 | rows 67+68 together vs master | FFmpeg 9.0.2 noasan clang `.so` (full components), direct_lite, 3 WAV seeds, fresh crash dir · 8 reps × 10k, master/avail/head/master2 interleaved | wall 7/8 vs master (p=0.07, med ratio 0.85) and 7/8 vs master2 (p=0.07, 0.90); edges 1/8 vs master, 4/8 vs master2 (spread 17.9k-23.8k) | 6/8 p=0.29, edges 2/8 | LEAN+ | `docs/DEEP_DIVE.md` §Katz consumers |
| 70 | 2026-10-31 | fluctuation edge set only with the tracker vs always built (on top of rows 67+68) | FFmpeg 9.0.2 noasan clang `.so`, direct_lite, 3 WAV seeds, fresh crash dir · 8 reps × 10k, katz/head/katz2 interleaved | wall 5/8 vs both, med ratio 0.94 / 0.95; edges 3/8, 2/8 | wall 3/8 p=0.73, edges 7/8 p=0.07 | NULL | `docs/DEEP_DIVE.md` §Katz consumers |
| 71 | 2026-10-31 | Katz consumers + lazy fluctuation edges vs availability cadence alone | fuzzgoat clang ASAN `.so`, direct_lite, fresh crash dir · 8 reps × 20k, avail/head/avail2 interleaved | wall 8/0 vs both arms, sign p=0.0078 each, med ratio 0.84 / 0.82 (30.1 → 25.3 s), eps 692 → 827; edges 4/8 | 6/8 p=0.29, edges 6/8 | WIN | `docs/DEEP_DIVE.md` §Katz consumers |
