# A/B pending — feature register

Every shipped feature still owed an A/B, or whose A/B was inconclusive. Source:
open items in `docs/TODO.md` (2026-10-22); mechanism detail in `docs/DEEP_DIVE.md`.
Close a row in the commit that records the result (DEEP_DIVE + TODO).

## Protocol (applies to every row)

- Target: clang-built fuzzgoat first (Hard Rule 52); format targets where noted.
- Harness: `tools/benchmark.py paired` (`tools/lib/bench_paired.py`); arm → baseline in `ARM_BASELINES`.
- Control: same arm vs itself first (Hard Rule 46). fuzzgoat is bimodal and one `-s` does not reproduce a run: use `--reps >= 3` or ≥ 20 seeds.
- Operators have no per-op off switch: the off arm is a tree without the op (`UNWIRED_ARMS`).
- Metric: real edges replayed on a non-marking build. Campaign `Edges discovered` counts COMPCOV marks.

Status: **U** unmeasured · **S** synthetic/harness only · **N** null/inconclusive, rerun owed.

## Crash handling

| Feature (switch) | What | St | A/B owed |
|---|---|---|---|
| Novel crash reward (`--novel-crash-reward`) | Only new-signature crashes pay operators / Elo | N | fuzzgoat noasan, 20 seeds. 10k: unique W8 L5 T7 p=0.58, edges W11 L9 p=0.82. 50k: unique W8 L8 T4 p=1, edges W9 L11 p=0.82 (median -13), crashes W9 L11. Controls (`novel-crash-reward-ctl`) clean: 50k unique W4 L6 p=0.75, edges W11 L9 p=0.82. Fuzzgoat yields a few crashes per run, so repeat-crash reward is rare there: needs a target where one shallow crash dominates. FFmpeg 9.0.2 (`--set ffmpeg`, arms `abort-crash` / `novel-crash-reward-abort` / `abort-crash-ctl`) from the set's empty corpus: 0 crashes at 3k and 20k execs. Seeded FFmpeg 8.0.3 (`--set ffmpeg_803`, `vpk_div0` SIGFPE dominates: 45 crashes / 1 signature at 5k), 20 seeds x 10k: unique all 20 ties (1 each arm), edges W7 L13 p=0.26 (Holm 0.53, med -726), crashes W11 L9 p=0.82. Control clean: edges W8 L12 p=0.50, crashes W9 L11 p=0.82. Null; no second bug within 10k in any arm. Stays opt-in |

## Seed scheduling

| Feature (switch) | What | St | A/B owed |
|---|---|---|---|
| Prune cost tie (`--prune-cost`) | Evict the costliest of equally redundant tracked seeds first | U | needs > 1,000 tracked seeds (FFmpeg, or a long fuzzgoat run); vs default, A/A control |
| Seed stage (`--stage-execs K`) | One pick serves up to K execs, sized by perf score | S | fuzzgoat exe K=8 vs 1, 8 paired reps: wall -5% (8/8, p=0.008; A/A 4/8), edges unchanged (p=0.73). Open: K in {4, 16}, FFmpeg, edges at fixed wall time; default flip |
| Branch seed arm (`--seed-branch-scheduler`) | Seeds near least-tried frontier branches, line-graph hops over the Katz ICFG | N | `--set fuzzgoat_katz`, 20 seeds x 10k, vs `elo` (Katz on in both). Edges W14 L6, McNemar p=0.115 (Holm 0.23), med +124; eps W7 L13 p=0.26, med -2.8; unique W7 L8. A/A (`seed-branch-ctl`) W11 L9 p=0.82, med +38: clean. 9 of 60 campaigns ended early (< 50 s, rc 0; elo 5, branch 3, ctl 1), not reproducible solo or 3-parallel; without them W9 L3 p=0.15, A/A W7 L7. Leans positive, unresolved: `--reps 3` or 40+ seeds |
| Newest-seed picker (`--seed-newest-scheduler`) | Growing-tree pick: newest with prob p, else random | U | arms `seed-newest-p{0,30,60,100}`, `seed-round-robin` vs `baseline`; compare lineage shape |
| Entropic seed (`--entropic-seed`) | libFuzzer Entropic energy | N | fuzzgoat null (9W/7L); png/grep, `--elo` vs weighted |
| Good-Turing seed/op/pos (`--good-turing-seed`, `--op-good-turing`, `--pos-good-turing`, `--good-turing-estimator`) | Unseen-edge mass per seed / op / bin | U | `elo-good-turing-seed`, `elo-op-good-turing` vs `elo`, `pos-arena-good-turing`; prior sweep; add `elo-good-toulmin-seed` |
| DPP seed arm (`--seed-dpp-scheduler`) | DPP over hashed edge sets | N | 5W/15L p=0.041 (Holm 0.083); `--reps 3`; as Elo arm vs `elo` |
| Doppler energy (`--schedule doppler`) | Power-Doppler seed energy | U | vs `fast` |
| Consolidated seed arm | Merged seed-arm features | U | vs `p2c`/`drr` |
| OS/net ports (`mlfq`, `stride`, `eevdf`, `bfq`, `sfq`, `codel`, `aimd`, `p2c`, `op_stride`, `op_p2c`) | Kernel/network schedulers as seed/op arms | U | each vs `seed_round_robin`/`drr` |
| LST revisit (`--lst-revisit`) | Revisit seeds overdue by cost deadline D | U | `tools/cost_dispersion.py` first; sweep D |
| Kruskal count (`--kruskal-count`) | Coupling-chain seed score | U | vs default and a `1/len` control |
| Entropy LOO (`--entropy-loo`) | Leave-one-out entropy seed value | U | LOO vs Shapley rank disagreement first |
| Strata seed arm (`--strata`, `--op-strata`) | Family-stratified pick | U | hit criterion ~0; E-1 then `strata-a0..a4` |
| Target arena (`--target-arena`, `auction`) | Elo over target schedules + Gale-Shapley | U | vs `--target-schedule weighted`, ≥ 2 targets |
| Cost in eviction ordering | Cost per retained edge in `_maybe_prune` | U | design + A/B (changes which seeds exist) |

## Operator scheduling

| Feature (switch) | What | St | A/B owed |
|---|---|---|---|
| Elo time credit (`--elo-time-credit`) | Strategy matches judged per second (autofz) | U | `--elo all --elo-time-credit` vs `--elo all`, fuzzgoat clang, wall-clock budget (not exec budget: per-exec budgets hide the gain); A/A first |
| Tempered Elo Thompson | Draw width `2 − favourite win rate` | S | `--elo` vs previous commit |
| Elo miss scoring | Rate on Beta hit rates / played arms only | U | design, then fuzzgoat |
| MC Thompson estimand (`op_monte_carlo`) | Fractional Bernoulli + decay-to-prior | S | vs previous commit |
| Consolidated v2 (`--consolidated-v2`) | v1 scored by a tempered Thompson draw | S | +1.3–1.8% synthetic; vs `--consolidated-v1` |
| BLUP prior (`--consolidated-prior blup`) | Shrinkage prior across ops | S | ASAN fuzzgoat |
| TPE (`--op-tpe`) | Tree-Parzen operator pick | U | vs `--bo-gp-ucb` under `--elo`; gamma sweep |
| AFL det arm (`--op-afl-det`) | Deterministic stage as an arm | U | cold-start curves (first 5 min) |
| Kuramoto floor (`--op-kuramoto`) | Badness floor like `--op-katz` | U | lock-in harness, then fuzzgoat |
| EXP3-IX / RM+ / automaton / ant colony | Elo-only bandits | S | vs MOSS/consolidated_v2 |
| PHE (`--phe`) | Perturbed-history exploration | S | vs MOSS/consolidated_v2 |
| Gamma-Poisson (`--gamma-poisson`) | Discounted Gamma–Poisson Thompson | S | vs `--kalman-ts` |
| Tsallis-INF / Kalman-TS / IDS | Elo-only bandits | S | vs MOSS/consolidated_v2 |
| Bayes-UCB (`--bayes-ucb`) | Quantile UCB | U | vs MOSS and default |
| Reactive gradient (`--reactive`) | Gradient + success-clocked forgetting, spike guard | N | FFmpeg 2026-10-25 null (W7/L3, p=0.34; control W5/L5): 20+ seeds, `--reps 3` |
| Chaotic inertia (`--swarm-inertia chaotic`) | Logistic-map PSO inertia | U | vs `constant` for `--mopt`, `--op-firefly` |
| Learned α / dict Thompson (`--dirichlet-alpha learned`, `--dict-thompson`) | Learned Dirichlet α; Thompson over tokens | N | 3k trend negative; 10k+ execs |
| Minimax (`--op-minimax`, `--minimax-select`, `--wall-order`) | Minimax op / minimization / ordering | N | png null; `--reps 2 --set cmplog` |
| Lottery ports (`--seed-stride-cost`, `--inverse-lottery-select`) | Compensation tickets on the stride arm; inverse-lottery eviction in `auto_minimize_corpus` | U | arms `seed-stride-cost` vs `seed-stride`, `inverse-lottery-select` vs `minimize-2k`; judge stride-cost on edges at fixed wall time, `--set ffmpeg` (widest exec-cost spread) |
| Lottery arm (`--seed-lottery-scheduler`, `--seed-lottery-currency {equal,sqrt}`, `--seed-lottery-transfer F`) | Randomized lottery over compensated tickets (Fenwick tree); per-component ticket currencies; lineage-parent credit | U | arms `seed-lottery` vs `seed-stride-cost`, `seed-lottery-currency` vs `seed-lottery`, `seed-lottery-transfer` vs `seed-lottery-lineage`; first record the component-size distribution on fuzzgoat, png_read, ffmpeg (a giant component makes currencies a no-op) |
| MCTS vs alpha-beta (`--mcts`, `--alphabeta`) | Lineage tree search | N | png null; ffmpeg set, then retire one |
| Preset ledger (`--preset-ledger`) | UCB1 over flag presets across campaigns | U | vs `baseline` via `benchmark.py replicated` |
| Recurrence gate (`--recurrence`) | RQA TRAPPED read on the exec stream | U | does forcing the gate shorten plateaus |
| `-M` default 1 | One op per mutant | N | jpeg, ffmpeg |
| `--hail-mary` SLOPT cap | Cap SLOPT exponent under the arena | U | wall-time lever, A/B |

## Position arms

All pair against `pos-arena-uniform`.

| Feature (switch) | What | St | A/B owed |
|---|---|---|---|
| Saliency (`--pos-saliency`) | NEUZZ gradient over a fitted MLP | U | `pos-arena-saliency`; read `stats()` |
| PCA (`--pos-pca`) | Byte groups by edges their flip moved | U | `pos-arena-pca`; must beat `finch`/`effector` |
| Kadane (`--pos-kadane`) | Max-excess window over bin rates | S | `pos-arena-kadane`; window hook rides with it |
| effector / token / chunk / changed / rare_mask | Det-stage map, dict tokens, chunk kinds, path-hash credit, FairFuzz mask | U | `pos-arena-{…}` on png_read |
| Consolidated (`--pos-consolidated`) | Merged learning arms | U | `pos-arena-consolidated` |
| Centroid (`pos_centroid`, `--cluster-jaccard`) | Cluster medoid offsets | N | inside control; seeded corpus needed |
| Fibonacci vs round robin | Bin visit order | N | `--reps >= 3` on png_read |
| Phase lock | Record-stride extrapolation of TE map | U | lock rate by stride; 64-byte TE cap cost |

## Operators

Off arm needs a tree without the op.

| Feature | What | St | A/B owed |
|---|---|---|---|
| `overflow_pair` | Two-field overflow sums | S | fuzzgoat; merge with `div_trap` if both null |
| FXT ops (`rational_pair`, `sum_shift`, `multiset_next`, `haar_coef`, `revbin_permute`, `hilbert_patch`, `hadamard_oa`, `fib_encode`) | Arndt FXT-book ops | S | fuzzgoat, then FFmpeg / png |
| Eleven numeric/text/merge ops (`varint_reencode`, `position_marginal`, `line_ending_swap`, `nul_inject`, …) | Field re-encodings, merges | S | fuzzgoat; drop zero-yield arms |
| Pair ops (`arith_mid`, `geo_mid`, `power_mid`, `line_blend`, `field_minmax`, `nm_reflect`, `opposition`) | Field interpolation between seeds | S | fuzzgoat; drop any not beating `arith_mid` |
| Lineage ops (`lineage_step`, `diff_evo`, `bit_majority`, `field_median`, `bit_cross`) | Parent-step / population ops | S | fuzzgoat with `--lineage` |
| Pattern tables, `seq_mutate`, `hash_len_ext` (`--hlx-algo`) | Extended tables; framed sequences; length extension | N | 8-seed run p=0.29; needs targets |
| `rev_circuit`, `bit_swap_*` | Reversible-circuit / fixed-width swaps | S | vs previous commit; check `Declin` |
| Ten format ops (`json_mutate` … `recompress_png_idat`) | Text/format-aware | S | each masked vs live |
| `parity_lock` / `parity_break` | Parity-preserving flips | S | needs a 7E1 / DES-key target |
| Line codes (`--op-line-code`) | Manchester/HDLC/COBS re-encode | S | needs a line-code target |
| `literal_mismatch` + frontier (`tools/frontier.py`) | Literal-chain gap op | N | `--reps 3`, frontier counts as metric |
| `gradient_descent` NUMERIC | Trace-cmp objective | N | +1056 edges p=0.29; ~45 seeds |
| `coupled_pair` (`--byte-interaction`) | Interacting byte pairs | S | fuzzgoat; tune top-K |
| `hmm_segment` + span positions (`--field-hmm`) | Viterbi field spans | N | fuzzgoat noasan 20 seeds x 10k: edges W9 L11 McNemar p=0.82, med -21.5; A/A `field-hmm-ctl` W9 L11 p=0.82, med -8 (clean). eps med 753 vs 845 (ctl 817). PNG kill criterion (field_map truth, 104 PNGs) failed: ~700 spans/seed vs ~17 true cuts; F1 0.028 vs random 0.026 exact, worse than random at ±1. Null; over-segments. Drop or retry with sticky transitions / min span length |
| Grimoire (`--grimoire`) | Generalised input splicing | U | text target (grep/sqlite) |
| `op_append` (`--op-append`) | Append one record | U | needs linear-instruction target |
| `taint_boost` / `--priming-check` | Revizor input boosting; primed-id filter | U | `--colorize` with / without |
| `fsm_regen` (`--fsm`) | FSM regeneration | S | n=1 smoke; needs state-machine target |
| Live format seeds (`--learn-format`) | 32 field variants per 5000 execs | U | queue on vs off; sweep interval |
| `wfc_reorder_learned` (`--wfc`) | WFC reorder, now learning | U | isobmff/riff (ffmpeg) |
| Gravity splice (`--splice-donor gravity`) | Gravity-model donor choice | N | fit never refit; log refits or retire |
| BMP WFC budget | Lower `BMP_WFC_MAX_TOTAL` | U | BMP target |
| Det arith 1..35, 16/32-bit | AFL deterministic arith | U | vs previous commit |

## Solvers and checksums

| Feature (switch) | What | St | A/B owed |
|---|---|---|---|
| Derived compares (`--derived-cmp`) | Fit compare operand from input bytes (affine / poly / GF(2)-quadratic) | S | fuzzgoat, png, sqlite; cost per admitted seed |
| Affine mod (`--affine-mod`) | Solve ring-affine compares | S | share of seeds that spend it for nothing |
| I2S fixpoint (`--i2s-fixpoint`) | Iterate I2S patches to self-consistency | S | fuzzgoat |
| Encoding coverage | Decoder-layer Redqueen encoders | S | vs previous commit (+29% Redqueen cost) |
| FNV / CRC inverses (`fnv1a_*`, `crc32_*`), fake-novelty penalty | Hash preimages; BranchTrap defeat | S | no target |
| Checksum sites (`--checksum-sites`) | Locate + repair checksum fields | U | png/zlib/gzip |
| Adler patch, linear checksums, `null_lattice` | Integer checksum recovery | S | zlib/PNG target |
| COMPCOV levels | Compare-progress marks | N | png/ffmpeg null; fuzzgoat owed |

## Coverage, corpus, runtime

| Feature (switch) | What | St | A/B owed |
|---|---|---|---|
| Edge clusters (`--edge-clusters`) | Class rarity + near-miss bonus | N | fuzzgoat null; FFmpeg/png |
| Inline sancov modes | `inline-8bit-counters`, `inline-bool-flag` | N | −23/−38 edges; fold buckets, 20×20k rerun |
| Shim bit tricks | SWAR fold, masked probe, `bit_compress` | S | rebuild + fuzzgoat calibration |
| Map size 2^16 vs 2^19 | Map size from observed edges | U | interleaved A/B + A/A on ffmpeg |
| FFmpeg coverage-first build | Parsers/bsfs, `--disable-asm` | U | edges vs old flags; exec/s |
| Zipf veto | Saturation-gate veto on Zipf tail | U | needs a CLI toggle and a saturating target |
| Joint liveness (`--joint-liveness`) | Block-sensitivity region probing | S | DEAD→LIVE flips on png/gzip/sqlite |
| Negative bloom single hash (`--negative-bloom`) | One digest for both blooms | S | exec/s on large inputs |
| xxhash removal (`key16`, `hash64`) | In-house hashes | S | exec/s |
| MinHash LSH layout | Compact buckets | S | exec/s |
| Two-stack `RunningMoments` | Windowed moments | S | exec/s |
| Resume SHM novelty | Skip seed rerun on `--resume` | U | start-to-first-round time, edges |
| `reset_bitmap()` drop | Skip redundant memset | U | exec/s in direct mode |
| Auto-dict + `--crash-explore` | Byteflip tokens; AFL `-C` | U | fuzzgoat |
| Pool drift (`--pool-drift`) | `dH`/`JS` of the pool | U | calibrate against edge curve |
| PLL (`--pll`) | Lock transitions vs stalls | U | ≥ 3 targets before a consumer |
| Shm write guard (`--shm-write-guard pkey`) | PKU-protected SHM | U | needs PKU host |
| Markov plateau null | iid-context JS threshold | U | plateau rate before/after |
