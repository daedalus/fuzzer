# Handover: port candidate survey (2026-08-28)

**Bases:** Part I grepped against `300d649` ("Count comparisons per call site, not only per
callback"); Part II against `a8ccf8c` ("Treat a state transition as coverage") with FFmpeg
at `d411d9e` (2026-08-27).
**Trigger:** survey of other fuzzers for features worth porting.
**Predecessor:** `docs/web_research_port_candidates_2026-08.md` (2026-08-25). That doc's
Tier tables are still the master list for the engine side.

Two passes, merged into one document because they share the same open threads and the same
verification recipe, and because neither is complete on its own:

- **Part I — engine features.** Sources: AFL++ `docs/features.md`, LibAFL component list,
  libFuzzer, HexHive artifacts, `wtdcode/sand-aflpp`. What the *fuzzing loop* is missing.
- **Part II — harness features.** Source: FFmpeg's own six OSS-Fuzz harnesses under
  `FFmpeg/tools/`. What the *targets* are missing. The engine passes never looked at this
  surface, and it turns out to be where the cheapest coverage is.

Both passes were audited against the live tree, which is the part that matters — Part I
began with five "gaps" and four of them were already implemented.

---

# Part I — Engine features (GitHub fuzzer survey)

## Headline finding: most of the obvious shortlist was already implemented

The first-pass shortlist from the sources was: Redqueen colorization, CmpLog
transformation solving, MendelFuzz deterministic-stage pruning, Entropic power
schedule, autotokens. **Four of the five already existed in the tree.** The
per-row audit is pruned — these are shipped features, not port candidates — but
the list of names stays, because the survey sources describe all five as
AFL++/libFuzzer novelties and they read as gaps until you look:
`core/colorization.py`, `core/rq_encodings.py` +
`services/operators.py::_op_redqueen_xform`, `core/skipdet.py`,
`core/schedules.py::SeedScorer._entropic_factor`, and
`import_corpus.py` + `import --autotokens`. **Grep before writing anything.**

Two things from that audit are open work rather than a record of it, and stay:

- **Colorization is opt-in (`--colorize`, `colorize_max_execs=512`) and should
  probably be opt-out.** This is the single most actionable item in this
  document: colorization is the precondition that makes the transformation
  solver's operand→offset mapping unambiguous, and the per-site PC keys added in
  `300d649` make the candidate filter sharper than AFL++'s callback-level one.
  Measure the exec cost against the `--colorize` budget on a real target first
  (ffmpeg recipe below).
- **Entropic and our Chao2 rewrite (`good_turing_estimate`) come from the same
  STADS framework** and should be reasoned about together rather than as two
  unrelated estimators. Nobody has.

## Genuinely absent — ranked

### I.1 SAND — decouple sanitization from the fuzzing loop
`wtdcode/sand-aflpp` (ICSE'25, merged upstream as AFL++ PR #2288). Fuzz a plain
build; forward only inputs with a *unique execution pattern* to separately-built
ASan/MSan binaries. The sanitizer builds carry sanitizer instrumentation and a
forkserver but **no coverage instrumentation**, so they never touch the bitmap.
Authors state the approach is fuzzer-agnostic and easy to port.
- Lands in: `build_targets.sh` (a `--san` sibling of the existing `.so` targets),
  `services/runner.py` (second executor), the interesting-input gate in
  `services/fuzzer.py`.
- Fits: we already have the bloom filter for execution dedup and multi-build
  target scripts; the "execution pattern" predicate is close to what the bloom
  path already computes.
- Effort: M. Highest expected value of the absent set.

### I.2 OptiMin — corpus minimization as MaxSAT
`HexHive/fuzzing-seed-selection` (ISSTA'21). Edge coverage as hard constraints,
seed exclusion as soft constraints; solved with EvalMaxSAT. Produces markedly
smaller corpora than greedy `afl-cmin`.
- Lands in: `services/minimize.py`, whose coverage mode is currently greedy
  set-cover over edge maps.
- Fits: z3 is already a CI extra; the paper's weighted variant (minimize file
  size while maximizing edge hit counts) maps onto the rarity/crowding weights
  from the edge-distribution work.
- Effort: L–M. Caveat from the paper: minimized-corpus *size* differences did not
  translate to statistically significant bug/coverage differences — the win is
  iteration rate, not coverage. Do not oversell it.

### I.3 Gramatron — grammar automatons instead of parse trees
`HexHive/Gramatron` (ISSTA'21); also vendored as `custom_mutators/gramatron` in
AFL++. Restructures the CFG into an FSA so an input is a *walk*; splice becomes a
state-matched cut of two walks. Claims unbiased sampling from the input state
space and much more aggressive mutation than parse-tree operators.
- Lands in: `core/grammar.py` alongside `TreeMutator`/`SubtreePopulation`; the
  `.gram` loader is the input side already.
- Caveat: exactness holds only for non-self-embedding grammars; self-embedding
  rules give a subset of the language.
- Effort: M.

### I.4 Grimoire-style generalization
LibAFL `GeneralizationStage`. Blank spans of an input, re-execute, and keep the
spans whose removal does not change coverage — yields structure (and
recombinable tokens) with **no grammar supplied**. Complements the ~155
hand-written format mutators exactly where we have no format mutator for a target.
- Lands in: a new stage next to the deterministic/havoc schedule; reuses the
  colorization executor loop almost verbatim (same "mutate, compare path
  checksum" shape).
- Effort: M. Cheapest of the structure-aware options because the plumbing exists.

### I.5 FairFuzz-style rare-branch masks
Compute, per rare edge, which byte positions can be mutated while still hitting
it, and restrict mutation to the complement.
- Fits: `_edge_owner_count` rarity is now correct (edge-distribution work,
  `0afc439`), so the "which branch is rare" half is done; only the mask half is
  missing.
- Effort: L–M. Lower confidence than the others — measure against the existing
  rarity bonus before committing, the two may overlap.

Not carried forward: N-gram coverage, trace-div/trace-gep, Nyx, Intel PT, LLM
paths — all already tabled in `docs/web_research_port_candidates_2026-08.md`
with the same or better analysis.

---

# Part II — Harness features (FFmpeg `tools/*_fuzzer.c`)

## The source material

Six files, 1620 lines total, all under `FFmpeg/tools/` at `d411d9e`:

| File | Lines | Library under test | Entry surface |
|---|---|---|---|
| `target_dec_fuzzer.c` | 656 | libavcodec decoders | `avcodec_send_packet` / `receive_frame` / `decode_subtitle2` |
| `target_dem_fuzzer.c` | 226 | libavformat demuxers | `avformat_open_input` / `find_stream_info` / `av_read_frame` |
| `target_enc_fuzzer.c` | 216 | libavcodec encoders | `avcodec_send_frame` / `receive_packet` |
| `target_bsf_fuzzer.c` | 166 | libavcodec bitstream filters | `av_bsf_send_packet` / `receive_packet` |
| `target_sws_fuzzer.c` | 203 | libswscale | `sws_init_context` / `sws_scale` |
| `target_swr_fuzzer.c` | 153 | libswresample | `swr_init` / `swr_convert` |

Our single harness, `targets/ffmpeg_read.c` (749 lines), covers roughly the union of
`dem` + `dec` and none of the other four.

## Status against the live tree

Grepped against `a8ccf8c`, not guessed — same discipline as the Part I table.

| FFmpeg mechanism | Where it lives there | Status here |
|---|---|---|
| Trailing parameter block | `dec:383` `if (size > 1024)`, `dem:133`, `sws/swr:128` | **Absent.** Attempted once as the "footer header" of `285d0fa` and no longer present at `a8ccf8c` — `grep -r 'FuzzHeader\|fuzz_parse_header\|max_iterations' targets/` is empty. See II.1 for why the first attempt was wrong. |
| `FUZZ_TAG` packet framing | `dec:523-540`, `bsf` same | **Absent.** One input = one `avformat_open_input`. No delimiter-aware operator in `core/mutations/generic.py` either. |
| Rotating pattern registers (`keyframes`, `flushpattern`) | `dec:537,573` | **Absent.** No flush, no discard/keyframe flags, no reset schedule. |
| Allocation budget via `get_buffer2` | `dec:111-175` | **Absent.** `grep -c max_pixels targets/ffmpeg_read.c` → 0; same for `max_samples`. |
| Deterministic iteration bound | `dem:38-43` interrupt counter; `maxiteration` | **Partial and wall-clock.** `ffmpeg_read.c:269` `g_watchdog_budget_ms = 900` plus `:610` `total_packets > 500`. Neither bounds work *inside* one packet. |
| Seekable fuzzed I/O + declared filesize | `dem:72-95`, `seekable` from input | **Absent.** `ffmpeg_read.c:160` passes `NULL` for both write and seek. All seek-dependent demuxer code is unreachable. |
| Filename/extension synthesis for probing | `dem:147-167` | **Absent.** `avformat_open_input(&fmt_ctx, NULL, NULL, NULL)`. |
| Decoder knobs from input | `dec:392-498` | **Absent.** 0 hits for `err_recognition`, `lowres`, `idct_algo`, `skip_frame`, `flags2`, `workaround_bugs`, `strict_std`. |
| `extradata` injection | `dec:489-496`, `bsf` same | **Absent.** The three `extradata` hits in our target are `fuzz_touch` reads of demuxer output, not injection. |
| Parser stage (`av_parser_parse2`) | `dec:545-566` | **Absent.** |
| Drain at end (`send_packet(ctx, NULL)`) | `dec:631-644` | **Absent.** Delayed frames and the flush path are never exercised. |
| Contract assertion (`av_assert0(ret != AVERROR_BUG)`) | `dec:88,98,503,577` | **Absent**, here and in every other target. We have no oracle beyond crash/timeout/sanitizer. |
| `av_force_cpu_flags(0)` | `dec:413`, `enc`, `sws`, `swr` | **Absent, and mostly moot** — see "Not worth porting". |
| bsf / sws / swr / enc coverage | four separate harnesses | **Absent.** `grep -rn 'swr_\|sws_\|av_bsf_\|avcodec_send_frame' targets/` is empty. Note `libswresample.a` is *already linked* into every ffmpeg target (`build_targets.sh:675`) and never called. |

## II.1 The trailing parameter block

This is the structural idea. Everything in II.4 depends on it, so it goes first.

FFmpeg carves a fixed window off the **end** of the input and reads harness configuration
from it with `bytestream2`: 1024 bytes in `dec`/`bsf`, 2048 in `dem` (1024 of which is a
filename), 128 in `sws`/`swr`. The remainder is the payload.

Three properties make it work, and the `285d0fa` attempt violated all three:

**(a) Guarded by size, read nowhere else.** `if (size > 1024)`. Below that the defaults
stand and the block is never touched. `285d0fa` read `hdr` on the parse-*failure* path in
zlib, gzip, jpeg, png, ffmpeg and fuzzgoat — uninitialised stack read whenever the footer
is absent, which with real corpus is the common case. Only `sqlite_read.c` and
`tailslayer_read.cpp` got this right, with `&& hdr.max_iterations > 0` in the condition.

**(b) The reader saturates.** `bytestream2_get_le32` and friends (`libavcodec/bytestream.h:72-79`)
return 0 and clamp the cursor when fewer bytes remain than requested. A short or truncated
block degrades to zeros rather than reading past the window.

**(c) Every bit pattern is a valid configuration.** Every field is clamped at read time:
`% FF_ARRAY_ELEMS(formats)`, `% 25`, `& 0x7FFFFFFF`, `% AV_PIX_FMT_NB`, table lookups over
`codec_tags`. There is no rejected value, so the mutator can write anything there.
`285d0fa` inverted this: it let the input raise `chunk_cap` (1000), `round_cap` (4096),
`pkt/frm_cap` (500) and `g_watchdog_budget_ms` (900) **upward without bound**. Those are
hang and OOM guards; an input that widens its own guards is an input that manufactures a
finding.

**Why the tail and not a prefix.** Same reasoning that kept the mode byte out of
`sqlite_read.c`: a prefix shifts the payload by one byte, and every magic-based sniffer in
`core/mutations/` keys on offset 0 (`zlib_read.c:39` dispatches on `buf[0]==0x1F && buf[1]==0x8B`;
`sqlite_chunk_mutate` needs `d[:16] == MAGIC`). A prefix silently demotes the whole corpus
to flat-bytes mutation with no visible symptom. A tail block leaves offset 0 alone. It is
`lz4_read.c:157` (`unsigned char mode = buf[0]`) that is the odd one out, and it survives
only because lz4 frames have no sniffer keyed on offset 0.

**Corpus stability is the real cost.** The block's layout is a wire format shared with
every file in the corpus. Change a field's offset and every stored input is silently
reinterpreted — not invalidated, *reinterpreted*, which is worse because nothing fails.
FFmpeg treats theirs as frozen; `dec:446-447` keeps deprecated `request_channel_layout`
handling explicitly so old fuzzing failures stay reproducible. Adopt the same rule: append
only, never reorder, never resize.

**Landing:** new `targets/fuzz_params.h` (header-only, no build changes). A `FuzzParams`
cursor with saturating `fp_u8/u16/u32/u64` readers and a `fp_init(&fp, data, size, window)`
that returns 0 when `size <= window` so callers can skip the whole block. Consumed by
`fuzz_ffmpeg` at `targets/ffmpeg_read.c:418` before `fuzz_open_input`.

**Mutator interaction, and why nothing needs to change on the Python side.** Our operators
are format-aware but have no notion of a protected tail region, so splice, truncate and
`byte_delete` will move and destroy the block constantly. FFmpeg lives with exactly this —
libFuzzer's mutators are format-blind — and it works *because of* property (c): whatever
lands in the window is a valid configuration, so a destroyed block is a re-rolled
configuration, not a broken input. Do not add a tail-preserving operator as part of this
work. If measurement later shows the configuration space is being explored too slowly, that
is a separate change with its own evidence.

## II.2 `FUZZ_TAG` framing: one input, many API calls

`dec:523-540` and `bsf` scan the payload for the 8-byte little-endian tag
`0x4741542D5A5A5546` — ASCII `FUZZ-TAG` — and treat each span between tags as a separate
packet. One flat input becomes a *sequence* of `send_packet` calls with decoder state
carried across them.

Two reasons this is worth more here than it looks:

1. **It is the input format that SGFuzz state coverage needs.** `a8ccf8c` landed
   `__sfuzz_state` transition coverage. A state machine only shows transitions if a single
   execution drives a sequence of calls; a one-shot harness records the same handful of
   transitions on every input regardless of payload. Tag framing is the cheapest way to
   make one input a sequence.
2. **It is reachable by the mutators we already have.** The tag is a literal, so
   `extract_corpus_literals` and the autotokens path will pick it up from any corpus file
   containing one, and `byte_insert` / splice / `chunk_shuffle` will move and duplicate it.
   Add it to a new `dictionaries/ffmpeg.dict` as well so the deterministic dictionary stage
   can place it without waiting for a lucky corpus.

**Rotating pattern registers.** Paired with the framing, `dec` pulls two 64-bit words from
the parameter block and consumes them a few bits at a time per packet:

```c
pkt->flags   = (keyframes & 1) * AV_PKT_FLAG_DISCARD + (!!(keyframes & 2)) * AV_PKT_FLAG_KEY;
keyframes    = (keyframes >> 2) + (keyframes << 62);
if (!(flushpattern & 7)) avcodec_flush_buffers(ctx);
flushpattern = (flushpattern >> 3) + (flushpattern << 61);
```

A rotate, not a shift, so the schedule is periodic and every packet gets a defined value
however long the sequence runs. Reproducible, mutation-reachable, and the whole schedule
costs 16 bytes of parameter block. This generalises past ffmpeg: the same two registers
would drive `sqlite3_reset`, `inflateReset` and `LZ4F_resetDecompressionContext` in the
other targets.

**Landing:** `targets/ffmpeg_read.c` packet loop (`:565-611`), replacing the flat
`av_read_frame` walk with a per-span loop. Keep the existing `total_packets > 500` cap as a
backstop; do not let the parameter block raise it (II.1c).

## II.3 Deterministic budgets in place of the wall clock

`ffmpeg_read.c` bounds an execution with a POSIX timer (`:269`, 900 ms) and a packet count.
FFmpeg bounds it with counters, and no clock anywhere:

- `maxiteration = 8096`, a hard loop bound.
- `maxpixels` / `maxsamples`, accumulated across the whole execution, with
  `goto maximums_reached` on overrun.
- an allocator-side budget: `fuzz_get_buffer2` overrides `ctx->get_buffer2` and returns
  `AVERROR(ENOMEM)` once `alloc_pixels` passes the budget (`dec:111-161`). The library then
  takes its own out-of-memory path, cleanly, instead of the harness being killed.
- `dem` adds an input-controlled `interrupt_counter` that decrements per callback and
  aborts the demuxer from inside (`dem:38-43`) — the fix for HLS's sleep loop.

This matters beyond tidiness. A wall-clock timeout is **not reproducible**: the same input
times out or does not depending on machine load, and a timeout finding recorded on a busy
box will not replay. Our own profiling run is evidence of how much the clock moves — the
hotpath work measured 230.2 s → 70.6 s on the same 1500 executions. Every timeout finding
recorded under the old timing is suspect under the new. Counter budgets do not have this
property.

Keep the watchdog as a last-resort backstop for genuine infinite loops. Add the counters as
the *primary* bound, and record which bound fired so triage can tell "hit the work budget"
from "hung".

**Landing:** `fuzz_get_buffer2` equivalent installed on each `DecoderSlot` context at
`ffmpeg_read.c:526`; `ctx->max_pixels` / `ctx->max_samples` set there too; accumulators and
a `maximums_reached` label in `fuzz_ffmpeg`; interrupt callback on the `AVFormatContext` in
`fuzz_open_input`.

## II.4 Decoder and demuxer knob surface

Cheap once II.1 exists, and this is where most of the raw coverage is. From `dec:392-498`:

`width`, `height`, `bit_rate`, `bits_per_coded_sample`, `sample_rate`, `block_align`,
`ch_layout.nb_channels` (`% FF_SANE_NB_CHANNELS`), `codec_tag` (indexed into the codec's
own `codec_tags` table), `idct_algo` (`% 25`), `lowres` (`% (max_lowres+1)`),
`skip_frame`, `workaround_bugs`, `strict_std_compliance`, `err_recognition`
(`AV_EF_AGGRESSIVE|COMPLIANT|CAREFUL`, optionally `|EXPLODE`), `flags2 |= FAST`,
`export_side_data`, `debug |= SKIP|QP|MB_TYPE`, a parser on/off bit, and an `extradata`
slice taken from the tail of the *payload* after the fixed block.

Three of these deserve individual mention:

- **`err_recognition |= AV_EF_EXPLODE`** turns tolerated corruption into a hard error
  return. It is a different code path, not just a stricter one.
- **`extradata` injection** reaches decoder init paths that no bitstream can reach, because
  extradata normally arrives from the container. Note the layout:
  `[payload][extradata][fixed block]`, with `if (extradata_size < size)` as the guard.
- **The parser stage** (`av_parser_parse2`, `dec:545-566`) is a whole library layer we do
  not touch at all. Behind one bit.

`dem` adds: `seekable`, a *declared* `filesize` that need not match the data (so
`AVSEEK_SIZE` can lie), `io_buffer_size`, and the extension-synthesised filename that makes
extension-keyed probing reachable.

**Landing:** `ffmpeg_read.c:526` for the codec knobs, `fuzz_open_input:146-205` for the I/O
and filename ones. Add `io_seek` to the `avio_alloc_context` call at `:160` — currently
`NULL`, which is why no seek path in any demuxer has ever been executed by this harness.

## II.5 The other four libraries as extra entry points

**Do not copy FFmpeg's build model.** They compile one binary per codec via
`-DFFMPEG_DECODER=xxx` (`dec:198-207` picks the symbol at compile time), which is hundreds
of binaries. We already have `--inprocess-func` (`services/fuzzer.py:634`,
`cli/commands.py:2364`) selecting an exported symbol by name, so several harnesses can live
in one `.so` and be selected at run time. That is the same isolation at a fraction of the
build cost.

Ordered by cost:

- **`fuzz_swr`** — free. `libswresample.a` is already in the link line
  (`build_targets.sh:675`) and nothing calls it. Port `target_swr_fuzzer.c` nearly verbatim:
  the format and layout tables, the `% FF_ARRAY_ELEMS` indexing, the 128-byte block, the
  `in_sample_nb > 1000*1000` bail. ~150 lines, no build change.
- **`fuzz_bsf`** — the vendored non-minimal build already has `--enable-bsfs`
  (`vendor_ffmpeg.sh:118`). Shares the tag framing and the parameter block with the decoder
  path, so it lands almost entirely on II.1 and II.2 machinery.
- **`fuzz_sws`** — needs a build change: libswscale is not in the link line at all
  (grep `swscale` in `tools/` returns nothing). Also brings `alloc_plane`, which is worth
  reading independently — it allocates each plane with 32 bytes of slack so a linesize
  overrun lands in the redzone rather than in the next plane.
- **`fuzz_enc`** — needs `--disable-encoders` removed from `vendor_ffmpeg.sh:119`, which
  grows the vendored build noticeably. Lowest priority. The interesting part is that the
  input is raw *frame* data rather than a bitstream (`enc:185-191`), which is a different
  mutation problem than everything else in `targets/`.

## II.6 Cross-cutting ideas, not ffmpeg-specific

**Contract assertions as an oracle.** `av_assert0(ret != AVERROR_BUG)` appears four times
in `dec`. `AVERROR_BUG` is documented as a return the library must never produce; asserting
on it converts a silent API-contract violation into a crash the fuzzer can see. We have no
oracle beyond memory safety and hangs, and this class costs nothing per execution.
Candidates in our other targets: `sqlite3_step` returning a code outside its documented set,
`inflate` returning `Z_STREAM_ERROR` (documented as "state was inconsistent"),
`LZ4F_isError` on a code that should be unreachable. Worth its own commit, separate from
the ffmpeg work.

**Bounded-product dimension sampling.** `sws:80-88` `mapres()` maps two uniform `uint32`s
to `(w, h)` with `w = round(exp(d) * 16384 / e)` and `h ≤ 16384 / w`, so the *area* stays
under a cap while the aspect ratio spans exponentially. Our image generators
(`mutations/webp.py`, `png.py`, `isobmff.py`) pick dimensions independently, which means
they generate area outliers that are pure timeout fodder.

**Reject impossible configurations early.** `sws:130-135`:
`if (mask && (mask & (mask - 1))) return 0;` — more than one scaler bit set is not a
configuration, so the execution ends immediately instead of being spent. A validity gate on
the parameter block, one popcount.

**Per-mode cost table.** `dec:221-362` is 140 lines of `maxpixels /= N` per codec, hand-
calibrated so slow codecs get proportionally smaller budgets (`CFHD` and `JPEG2000` get
1/16384, `BMP` gets 1/16). If we adopt work budgets (II.3) with several entry points, the
same table shape applies per entry point. Do not hand-write ours — derive it from the
per-execution timing we already collect.

## Not worth porting

**`av_force_cpu_flags(0)`.** In FFmpeg's own builds this bit reaches the scalar reference
implementations, which is real coverage. In ours `vendor_ffmpeg.sh:191` already passes
`--disable-x86asm`, so most of the SIMD is not compiled in and the knob would be close to a
no-op. Revisit only if the vendored build ever enables asm.

**One binary per codec.** Covered in II.5. Use `--inprocess-func` instead.

**`error()` → `exit(1)` on allocation failure.** FFmpeg exits the process when a harness
allocation fails, which under our in-process runner would take the campaign down rather
than the test case — the same reasoning already documented for the watchdog at
`ffmpeg_read.c:255-262`. Return early instead.

## Suggested commit sequence for Part II

Each step is independently testable and independently revertable. Do not collapse them.

1. **`targets/fuzz_params.h`** — saturating tail-block reader, plus
   `tests/test_fuzz_params.py` compiling it with gcc and asserting the three invariants of
   II.1 directly (short input → block skipped; truncated block → zeros not overrun; every
   random 1024-byte block → in-range values). This is the file `285d0fa` was missing, and
   its absence is why that commit did not compile.
2. **Work budgets** — `get_buffer2` override, `max_pixels`/`max_samples`, interrupt
   callback, accumulators. Watchdog demoted to backstop, and the fired-bound recorded.
   Measurable on its own: run the existing ffmpeg corpus before and after and compare the
   timeout count.
3. **Tag framing + rotating registers** — packet loop rewrite, `dictionaries/ffmpeg.dict`.
   Coverage delta is the acceptance criterion here, and with `a8ccf8c` in the tree the
   state-transition count is the more sensitive of the two signals.
4. **Decoder knob surface** — the II.4 field list, wired to the block from step 1.
5. **Seekable I/O + filename synthesis** — `io_seek`, lying filesize, extension table.
6. **`fuzz_swr`** as a second entry point in the same `.so`; `fuzz_bsf` after it.
7. **Contract assertions** across all targets, as its own commit (II.6).

Steps 5-7 are optional for a first pass. Steps 1-4 are the ones that carry the value.

---

# Shared: open threads, verification, method

## Open threads inherited by whoever picks this up

Re-verified against live source. Two of the four threads this document opened have
since closed upstream and are pruned: the weight-cache key is no longer
`(corpus_version, exec_count // 50)` — it is `exec_count // 200` with a
corpus-growth threshold of 20 (`seed_picker.py:1273-1285`) — and the
positional-argument defect in the generators is fixed, by keyword call sites
plus an `isinstance(..., int)` coercion in ten generator signatures, not the
three this document named. What remains:

1. **`_record_cmp_progress` is still callback-granular** (`services/fuzzer.py:3077`,
   called at `:3550`). There are only 27 callback buckets, so the reward it feeds
   is 27-dimensional. The per-site PC table from `300d649` is what makes it dense.
   The measurement that justifies it: a target with one always-satisfied and one
   never-satisfied `memcmp` reads as `memcmp (4, 3)` per callback — 75% satisfied,
   no wall visible — but `(3,3)` and `(1,0)` per site. The bucket inverts the
   reading, it does not merely blur it.
2. **`tools/profile_hotpath.py` cannot drive the ffmpeg target.** It has
   `os.chdir("/home/dclavijo/my_code/fuzzer")` hardcoded and does not emit
   `--inprocess-direct/--inprocess-func`. Any before/after measurement for Part II
   steps 2 and 3 needs this fixed or needs another instrument.

## Verification recipe

Standard, unchanged: `git am` the series onto a fresh clone at the base commit, build, run
the full suite, compare against a **base run in the same container** — not against the
7-failure figure from the author's machine. In a container without clang the base is 13
failures / 5665 passing; with clang it is 33 / 5728. The distance, ICFG, scov and tracecmp
families skip rather than fail when clang is missing, which is the whole difference.

Container notes for reproducing the ffmpeg target from scratch:
`apt-get update` first (without it `apt-get install clang` 404s on libc6-i386/libxml2-dev),
then `tools/vendor_ffmpeg.sh --nosan --minimal` (ffmpeg.org is blocked; it falls through to
the codeload mirror, ~2 min on one core), then
`tools/build_ffmpeg_ready.sh --minimal --no-vendor`. Note `--minimal` restricts the build to
mov/matroska/wav/aiff/flac/mp3/ogg demuxers and seven decoders — enough for Part II steps
1-5, not enough for a coverage comparison that means anything.

Watch disk during the suite: run it in chunks with a private `TMPDIR` and clean between
chunks (~5 MB of scratch that way).

## Method note

Part I's sources are web-only; its effort estimates are guesses until a per-item audit
against live code happens. What is *not* a guess is its status column — that was grepped
against `300d649`. Part II's status table was grepped against `a8ccf8c` and its FFmpeg line
references against `d411d9e`; both are checkable.

The rankings, the effort ordering and the claim in II.2 that tag framing is what makes
SGFuzz state coverage pay are judgment, and none of it has been measured. Part II steps 2
and 3 both have a cheap measurement attached — do those measurements before building on top
of them. And repeat the grep step before starting any item here: this survey began with
five "gaps" on the engine side and four of them were already in the tree.
