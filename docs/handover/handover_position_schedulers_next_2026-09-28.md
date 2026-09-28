# Handover: next position schedulers for the Elo position arena (2026-09-28)

Base: `daedalus/fuzzer` HEAD `c48d5a1d` (PositionFractalScheduler, on top of
`99800db4` kl_ducb, `92cb37eb` ffmpeg multi-version vendoring).

**Update:** arms 2 (`lineage`, section 5.2) and 3 (`cmplog`, section 5.3) are now implemented
(`pos_lineage.py`, `pos_cmplog.py`); `fractal` and `kl_ducb` landed earlier. Arm 1 (`context`, section 5.1) is implemented (`pos_context.py`, `_bytecls.py`, off-policy extra as specified). Arm 5 (`levy`, section 5.5) is implemented (`pos_levy.py`, off-policy extra as specified; deviations
listed in the paragraph below). Arm 4 (`boundary`, section 5.4) is implemented (`pos_boundary.py`, off-policy extra as specified; it
reuses `_bytecls.py`). Still unbuilt: 0
(benchmark prerequisite). `lineage` deviates from 5.2 in one
place: it is wired tracker-style (gate = `--lineage` on) instead of as an off-policy extra,
because `parent_sites` is recorded only under `--lineage`; without it the arm would be a pure
decliner and get flagged by the uniform-floor inspection. `record()` stays a no-op, so the
difference is only pool membership. The `_DELOCALISED_OPS` set is injected through the constructor
(`core` must not import `services`). Reflection at the edges is a triangle-wave fold, which
equals the single reflections in 5.2 and also handles a site far past a shrunken buffer.
Unmeasured, like every arm below.

**Status: design only** (as written; see the update above for what has since landed). Nothing below is implemented, run, or benchmarked.
Every constant is an initial value to sweep, not a measured optimum. Line
numbers refer to HEAD `c48d5a1d` and will drift.

Companion doc (what exists today): `docs/handover/handover_position_arena_2026-09-24.md`.

---

## 1. Summary

Five new arms for the third scheduling axis (where a mutation lands):

| # | arm (`pos_<n>`) | class | signal | learns? | state | wired as |
|---|---|---|---|---|---|---|
| 1 | `context` | `PositionContextScheduler` | byte content + coverage feedback, shared across all seeds | yes (global) | 560 Beta pairs | extra (off-policy credited) |
| 2 | `lineage` | `PositionLineageScheduler` | `seed_meta[...]["parent_sites"]` (already recorded) | no (v1) | none | extra, needs `meta_of` |
| 3 | `cmplog` | `PositionCmplogScheduler` | `redqueen_offsets` + Weizz `StructureMap` spans | no (v1) | derived cache | tracker-style arm with gate |
| 4 | `boundary` | `PositionBoundaryScheduler` | byte content only (class/entropy/delimiter edges) | no | derived cache | extra (stateless) |
| 5 | `levy` | `PositionLevyScheduler` | last gain offset per seed | yes (per seed) | anchor per seed | extra (off-policy credited) |

Ranking by expected value: 1, 2, 3, 4, 5. Recommended build order (cost/risk):
**0 (benchmark prerequisite), 2, 4, 3, 5, 1**. See section 8.

Why these gaps: the three learning arms today (`burn_front`, `kl_ducb`,
`fractal`) all (a) keep per-seed state in an LRU of 256 seeds and start from
zero on every new seed, and (b) learn only from coverage feedback. Redqueen
offsets, Weizz structure tags, lineage information and raw byte content
never reach the arena.

---

## 2. Ground truth: contract and arena behaviour

### 2.1 Protocol (`core/schedulers/pos_base.py`)

```python
class PositionScheduler(Protocol):
    name: str
    def propose(self, data: bytes, buf_len: int) -> int | None: ...   # None = no opinion
    def record(self, data: bytes, offsets: Sequence[int],
               outcome: Outcome, weight: float = 1.0) -> None: ...
```

- `data` is the **parent seed**; `buf_len` is the **live buffer** length
  (earlier operators in the round may have resized it). The arena clamps the
  result to `[0, buf_len-1]`.
- `Outcome.GAIN` / `Outcome.MISS`.
- `CallablePosition(name, fn)` adapts a bare `fn(data, buf_len)`; its
  `record` is a no-op. Used by `sensitivity/te/phase/mi/crash_mi/region/field`.

### 2.2 Arena (`services/position_arena.py`)

- Arms are `dict[str, (scheduler, gate)]`. `uniform` is always first and is
  the floor. Only arms whose gate is true join the pool (no phantom matches).
- Elo picks the arm (`select_strategy` over `pos_<n>` keys), then
  `propose()`. **A decline (`None`) yields a uniform offset charged to the
  declining arm**, so a pure decliner rates exactly as uniform (measured, see
  the arena docstring). New arms may therefore decline freely on cold data.
- `settle()`: every "extra" (`burn_front, kl_ducb, canary, round_robin,
  fibonacci, fractal`) gets `record()` **every round, whoever served** (off-policy
  credit). Then Elo matches: each served arm plays each unserved pool member
  with the round score.
- `_check_canary_inspection` flags any real arm rated at or below `uniform`.
  Every new arm must beat uniform or it will be flagged.

### 2.3 Which sites reach `record()`

`Fuzzer._settle_positions` (fuzzer.py ~7900) builds
`sites = [s for op, s in self._last_ops_with_sites if op not in _DELOCALISED_OPS]`
(`block_shuffle_variable, byte_shuffle, chunk_shuffle, token_shuffle`) and calls
`arena.settle(parent_seed, sites, outcome, weight, score)` with
`score = weight if GAIN else 0.0`.

Known imprecision (accepted by all existing arms): sites are in **live-buffer
coordinates**, which can differ from parent coordinates if an earlier operator
in the round resized the buffer.

### 2.4 Conventions to copy (from `pos_burn_front`, `pos_kl_ducb`, `pos_fractal`)

- Constants at module top: `MAX_BINS = 4096` (import from `pos_burn_front`),
  `MAX_SEEDS = 256`, `STATE_VERSION = 1`.
- Per-seed key: `xxhash.xxh3_64_intdigest(data)`; per-seed table is an
  `OrderedDict` with `move_to_end` on access and `popitem(last=False)` beyond
  `MAX_SEEDS`.
- Bin width: `max(1, -(-len(data) // MAX_BINS))` (fixed from the parent seed),
  except `fibonacci`, which bins the live buffer.
- `record()`: drop `o < 0`; if no offsets left, return; split `weight` evenly
  (`share = weight / len(offsets)`); clamp offsets past the seed end.
- Persistence (learning arms only): `to_dict()` / `from_dict()` via
  `Fuzzer._state_store`; `from_dict` validates `version`, and on any
  `AttributeError/KeyError/TypeError/ValueError` logs a warning and starts
  fresh; oldest entries dropped to respect `MAX_SEEDS`.
- RNG: `RandPool` (`self._rng`). The test double `ScriptedRng` in
  `tests/test_pos_fractal.py` scripts `random()`, `randint(a,b)` and an
  argmax `weighted_choice`. **Verify the exact RandPool API before use**
  (in particular that `weighted_choice` exists with the signature the
  existing arms use), and write against that.

---

## 3. Wiring checklist (template: commit `c48d5a1d`, 6 files)

Repeat per arm (`<n>` = `context`, `lineage`, `cmplog`, `boundary`, `levy`):

1. **`core/schedulers/pos_<n>.py`**: the class; docstring in the same
   style as `pos_fibonacci.py` / `pos_kl_ducb.py` (what it does, why, what
   `record` does, persistence).
2. **`services/position_arena.py`**:
   - module docstring "Arms::" entry;
   - append `"<n>"` to `POSITION_STRATEGY_NAMES`;
   - add `<n>: PositionScheduler | None = None` to `PositionArena.__init__`,
     store `self._<n>`, add to the `for extra in (...)` tuple **and** to the
     `extras` tuple in `settle()` (two places; missing the second means the
     arm never gets `record()`).
   - Exception: `cmplog` is wired inside `_add_trackers` (section 5.3).
3. **`services/fuzzer.py`**:
   - ctor kwarg `pos_<n>=False`;
   - `self._pos_<n> = None; if pos_<n> or position_arena: ... = Pos...(self._rng)`
     with a `log.info("Position <n> scheduling enabled")`;
   - pass `<n>=self._pos_<n>` where `PositionArena(...)` is built;
   - add `names.append("<n>")` in the banner block that lists enabled arms;
   - if the arm has state: `self._state_store.set("pos_<n>", self._pos_<n>.to_dict())`
     in `_save_learned` and `self._pos_<n>.from_dict(self._state_store.get("pos_<n>", {}))`
     in `_load_learned`.
4. **`cli/commands.py`**: `--pos-<n>` (`store_true`), add `"pos_<n>"` to the
   args-to-ctor name list, extend the `--position-arena` and `--hail-mary`
   help strings, and make `--hail-mary` enable it.
5. **`tests/test_pos_<n>.py`** (new) and **`tests/test_position_arena.py`**
   (pool membership/off-policy credit assertions, mirroring the fractal diff).
6. **`tools/lib/bench_paired.py`**: see section 7.

**Legacy path note.** `OperatorEngine.select_position` (operators.py ~5501)
has a non-arena branch that picks uniformly among candidates from
`_side_positions` (sens, crash_mi, region, field, burn, round_robin,
fibonacci) plus te/phase/mi. Reading that code, `kl_ducb` and `fractal` are
**not** candidates there, i.e. they act only inside the arena (not verified by
running). Follow the same rule for all new arms: **arena-only**, do not
extend `_side_positions` (it would change its 7-tuple contract and every
caller). This has a benchmarking consequence, see section 7.

Also update `docs/architecture.dot` if any arm adds a node (the 09-24
handover left it stale; do this once for all arms).

---

## 4. Shared design rules for all five arms

- **Decline liberally.** Returning `None` is cheap and correct: it degrades
  to uniform, charged to the arm. Cold start, missing metadata, degenerate
  buffers (`buf_len < 1`, `len(data) == 0`) all return `None`.
- **Never raise.** Malformed meta, stale tags, offsets past the end: clamp or
  decline. `select_position` is on the hot path of every mutation.
- **Keep a uniform escape.** Each informed arm mixes in `EPSILON` uniform
  draws (implemented as returning `None`, or an explicit `randint`). Without
  it a wrong signal locks the arm onto a fixed subset.
- **No new Elo machinery.** No changes to `analyzer_elo.py`; `POS_STRATEGY_PREFIX`
  keys are derived from `POSITION_STRATEGY_NAMES`. Verify
  `core/analyzer_registry.py` pre-registers keys from that tuple (the
  09-24 handover says `pos_` keys are pre-registered); if it lists them by
  hand, add the new names there too.
- **Delocalised operators** are already filtered by `_settle_positions`; do
  not re-filter in the arm.

---

## 5. Per-arm specifications

### 5.1 `pos_context.py`: `PositionContextScheduler` (arm `context`)

**Hypothesis.** Which offsets pay off depends partly on what the bytes
*around them are like* (a delimiter, a high byte after a length-looking
byte), and that relationship transfers across seeds of the same format. The
existing learners key on offsets **within one seed** and are blind to this;
this arm keys on local byte context and pools evidence across the corpus, so
a brand-new seed starts with a warm prior instead of zero.

**Constructor.** `PositionContextScheduler(rng: RandPool)`; `name = "context"`.

**Features** for offset `o` in `buf` (use the parent `data` as the content
source; offsets `>= len(data)` are skipped in `record` and unscored in `propose`):

- `cls(b)`, 7 byte classes: `0x00`, `0xFF`, `0x01-0x1F` (control),
  digit, alpha, other printable (punct/space, `0x20-0x7E` minus the previous two),
  `0x80-0xFE`. (Put `0x7F` in "other printable" or the control class; pick one
  and pin it in a test.)
- `prev_cls`: `cls(data[o-1])`, or a dedicated 8th value for `o == 0`.
- `decile`: `min(9, o * 10 // len(data))`.
- `ctx = (cls, prev_cls, decile)` -> index into a flat table of
  `7 * 8 * 10 = 560` cells (`NUM_CTX`).

**State.** Two float arrays `succ[NUM_CTX]`, `fail[NUM_CTX]` (discounted
Beta counts), plus `obs` (total records) and a record counter for discounting.
Global, **not per seed**: no LRU, no eviction.

**`propose(data, buf_len)`.**
1. If `obs < MIN_OBS` (200) or `not data`: return `None`.
2. Draw `K = 16` candidate offsets uniformly from `[0, min(buf_len, len(data)) - 1]`
   via `rng.randint`.
3. For each candidate, weight
   `w = (succ + PRIOR_A) / (succ + fail + PRIOR_A + PRIOR_B)` divided by the
   global mean rate, clamped to `[W_MIN, W_MAX] = [0.25, 4.0]`
   (`PRIOR_A = 1`, `PRIOR_B = 20`, reflecting a miss-dominated base rate).
4. Pick one candidate by `weighted_choice`. Expected effect: a mild tilt over
   uniform, never a collapse (the clamp is the safety).

**`record(data, offsets, outcome, weight)`.** For each `o` in
`offsets` with `0 <= o < len(data)`: `share = weight/len(offsets)`; add to
`succ[ctx(o)]` on GAIN or `fail[ctx(o)]` on MISS. Every `DISCOUNT_EVERY` (256)
records multiply both arrays by `DISCOUNT = 0.95` (nonstationarity). Off-policy:
credited every settled round.

**Persistence.** `to_dict()`: `{"version", "succ": [...], "fail": [...], "obs"}`;
`from_dict` rejects wrong array length (`!= NUM_CTX`) as malformed and starts
fresh. ~1.1k floats, negligible.

**Diagnostics.** `top_contexts(n)` returning the best/worst contexts by rate
with counts, for the report ("which context did it learn?"). This is also the
main interpretability payoff of the arm.

**Flag.** `--pos-context`; implied by `--position-arena`; in `--hail-mary`.

**Failure modes / risks.**
- Base-rate collapse: mitigated by the weight clamp and `PRIOR_B`.
- Format confounding: a corpus mixing formats blends statistics. Acceptable
  for v1; v2 could key the table by a coarse format id.
- Offset attribution noise (live-buffer vs parent coordinates) blurs the
  labels, same as every other arm.
- Sampling K uniform candidates then reweighting reduces to a tilt; if the
  measured effect is null, raise `K`, or reduce to full-buffer scoring for
  small seeds.

**Tests (`tests/test_pos_context.py`).** Satisfies the protocol; cold
(`obs < MIN_OBS`) declines; a class with many gains gets picked more than one
with none (ScriptedRng, fixed candidate draws); weight clamp holds under
extreme counts; offsets `>= len(data)` and negatives ignored in `record`;
class function boundaries (`0x00`, `0x1F`, `0x20`, `0x7E`, `0x7F`, `0x80`,
`0xFE`, `0xFF`, digit/alpha edges); `o == 0` prev-class; discounting shrinks
counts; round trip; malformed/wrong-length state resets; empty data/buffer.

---

### 5.2 `pos_lineage.py`: `PositionLineageScheduler` (arm `lineage`)

**Hypothesis.** The offsets that produced a new seed are direct evidence
about where mutations pay off *for that seed's descendants*. A child that
was admitted because a mutation at offset `s` found coverage is the best
cold-start prior for the next mutations of that child.

**Data already exists.** `corpus_manager.py` writes
`seed_meta[child]["parent_sites"]` (list[int], line ~1038, from
`f._last_ops_with_sites`), alongside `parent_key`, `parent_ops`, and
`lineage_depth`. No new plumbing in the corpus path.

**Caveats to handle in the arm (not in the corpus writer).**
- `parent_sites` is stored **unfiltered**, including sites of
  `_DELOCALISED_OPS`. When `len(parent_ops) == len(parent_sites)` (aligned),
  drop entries whose op is in `_DELOCALISED_OPS`; if not aligned, use all
  sites. Do **not** change the corpus writer: `tmin.py`, `crash_metadata.py`
  and `core/lineage.py` consume the same field.
- One path (corpus_manager ~1198) appends `len(data)//2` as a synthetic
  site to an inherited list. Treat as ordinary data; it is a harmless midpoint.
- Sites are offsets in the **parent's** coordinates; for length-preserving
  ops they are also valid in the child. v1 clamps and accepts drift for
  length-changing ops.

**Constructor.** `PositionLineageScheduler(rng, meta_of)` where
`meta_of: Callable[[bytes], dict | None]`; Fuzzer passes `self.seed_meta.get`
(seed_meta entries are dicts keyed by seed bytes). `name = "lineage"`.

**`propose(data, buf_len)`.**
1. `meta = meta_of(data)`; no meta, or no usable `parent_sites`
   (`lineage_depth == 0` seeds, the initial corpus) -> `None`.
2. With probability `EPSILON = 0.2` return `None` (uniform escape).
3. Pick one site: uniform over the (filtered) list. Add jitter drawn from a
   two-sided geometric with mean `JITTER = 8` bytes (sign uniform), so a
   run of mutations covers the neighbourhood, not one byte.
4. Reflect at the edges (`-x -> x`, `x >= n -> 2(n-1) - x`), then clamp.

**`record`.** No-op in v1 (stateless; the meta is persisted by the corpus).

**v2 (deferred, needs a decision).** Walk `parent_key` up to
`LINEAGE_DEPTH_MAX = 4` generations with weight `0.5**depth`, correcting
offsets across length-changing steps by common prefix/suffix alignment
between parent and child bytes (`o < prefix`: unchanged; `o >= len(parent) -
suffix`: shift by `len(child) - len(parent)`; inside the changed region:
clamp). Requires a parent-bytes accessor from `parent_key`; find it in
`CorpusManager` (`seed_key`) before committing. Also consider feeding
lineage sites as pseudo-observations into `pos_context`.

**Persistence.** None. **Flag.** `--pos-lineage`; implied by
`--position-arena`; in `--hail-mary`. Cheap (a dict lookup), so it can be
implied like `fibonacci`.

**Tests (`tests/test_pos_lineage.py`).** Declines with no meta / empty
sites / depth 0; picks near a recorded site (fixed jitter draw); filtered
delocalised sites are excluded when ops align, kept when misaligned;
reflection at both edges; clamped to a shrunken buffer; uniform escape
(`random()` below `EPSILON`) returns `None`; meta_of raising or returning a
non-dict is handled (decline, never raise).

---

### 5.3 `pos_cmplog.py`: `PositionCmplogScheduler` (arm `cmplog`)

**Hypothesis.** Comparison operands (magic numbers, lengths, checksums) are
where non-redqueen operators (bit flips, arithmetic, havoc) are most likely
to unlock branches, and the information is already collected but only feeds
the redqueen operator.

**Signals (all pre-existing).**
- `seed_meta[data]["redqueen_offsets"]`: `list[int]`, up to 50, set at
  fuzzer.py ~5690 from `redqueen_matches`.
- Weizz `StructureMap` from `load_tags_from_meta(meta)` (see
  `OperatorEngine._weizz_structure_map`), returned as `None` when
  `meta["weizz_tags_dirty"]`. Useful spans:
  `smap.flagged_spans(mask)` with
  `mask = TagFlags.IS_MAGIC | TagFlags.IS_LEN | TagFlags.IS_CHECKSUM | TagFlags.IS_INPUT_TO_STATE`
  (the method tests `tags[start].flags & flag`, so a combined mask matches
  any of the bits). Returns `(start, end, cmp_id)` triples.

**Constructor.** `PositionCmplogScheduler(rng, meta_of, smap_of)`, where
`smap_of(data)` is `OperatorEngine._weizz_structure_map` (returns `None` when
dirty/absent). `name = "cmplog"`.

**Per-seed cache** (derived, not persisted): `OrderedDict[key] -> _Targets`
(`MAX_SEEDS = 256`) holding `spans: list[(start, end, weight)]` and
`points: list[int]`, rebuilt when the seed's meta identity or
`weizz_tags_dirty` state changes (cache key includes the tags `dirty` flag and
`len(redqueen_offsets)`).

**Weights** (initial): `IS_LEN` 1.5, `IS_CHECKSUM` 1.0, `IS_MAGIC` 1.0,
`IS_INPUT_TO_STATE` 1.0, each redqueen offset as a point target 1.0.

**`propose(data, buf_len)`.**
1. Build/fetch targets. None at all -> `None` (arm silently equals uniform on
   targets without cmplog data).
2. `EPSILON = 0.1` -> `None`.
3. Weighted pick among spans+points; inside a span pick uniform in
   `[start, end)`; points get +/-1 jitter. Clamp to `buf_len`.

**`record`.** No-op v1 (passive). v2: per-`cmp_id` gain credit to reorder
targets.

**Wiring differs from the others.** Do **not** add it as an `extra`. Add it
to `PositionArena._add_trackers` `specs` like `field`, with a gate: join the
pool only if cmplog/Weizz collection is on. Find the exact attribute names
(`getattr(f, "weizz_tags", False)` is used in corpus_manager ~1046; the
cmplog/redqueen flag is around fuzzer.py ~1624). Wrap as
`CallablePosition("cmplog", scheduler.propose)`. Flag `--pos-cmplog` should
force the arm on (and require cmplog); also enabled by `--hail-mary` when
cmplog is.

**Risks.**
- **Overlap with `field`** (FormatLearner: *confirmed* coverage-causal
  offsets) vs this arm's *presence*-based spans. Benchmark them head to head;
  if `cmplog` never beats `field` and never beats uniform, drop it.
- **Double-dipping:** the redqueen operator already targets these offsets
  (operators.py ~4275). The interesting question is whether *other* operators
  landing there pay; check gain share by operator in the report.
- Cost is one dict/tag lookup per pick after the first (cache).

**Tests (`tests/test_pos_cmplog.py`).** No meta -> decline; dirty tags fall
back to redqueen offsets only; flagged spans respected; jitter clamped;
cache invalidation when tags go dirty; single-flag vs combined-mask
`flagged_spans` behaviour pinned; gating: arm absent from the pool when
cmplog is off (arena test).

---

### 5.4 `pos_boundary.py`: `PositionBoundaryScheduler` (arm `boundary`)

**Hypothesis.** Field boundaries (token starts, class changes, run edges)
are enriched for interesting mutations regardless of feedback, so a
content-only prior helps from the first pick, like `fibonacci` but informed.

**Constructor.** `PositionBoundaryScheduler(rng)`; `name = "boundary"`; stateless
apart from a derived cache.

**Score array (per seed, computed once).** For each boundary position `i`
(between `b[i-1]` and `b[i]`), `score[i]` is the sum of:
- class change: `cls(b[i]) != cls(b[i-1])`: +1.0 (reuse the class function
  from `pos_context` via a shared helper module, e.g. `core/schedulers/_bytecls.py`,
  to keep one definition);
- delimiter start: `b[i-1] in {0x00, 0x0A, 0x0D, 0x20, ',', ':', ';', '=', '/', '<', '>', '"', '{', '}', '[', ']'}`: +1.0;
- entropy step: `|H(b[i:i+W]) - H(b[i-W:i])|` with `W = 16`, Shannon
  entropy in bits normalised by 4 (cap 1.0);
- run edge: end or start of a run of `>= 4` equal bytes: +0.5;
- 4-byte alignment: `i % 4 == 0`: +0.25.

Vectorise with numpy (already a dependency); cap the scan at
`SCAN_CAP = 65536` bytes (score only the first `SCAN_CAP`, plus uniform for the
rest). Keep only the top `TOP_K = 256` `(offset, score)` pairs per seed; cache
in an `OrderedDict` (`MAX_SEEDS = 256`). Cache is derived; no persistence.

**`propose(data, buf_len)`.**
1. Empty data -> `None`. `EPSILON = 0.15` -> `None`.
2. Weighted pick among the top-K by score; jitter in `{-1, 0, +1}`
   (a boundary offset is where a field starts; the byte before is often the
   previous field's tail).
3. Clamp to `buf_len`.

**`record`.** No-op.

**Reuse check before writing.** `core/mutations/fractal_voronoi.py`
already biases "boundary bytes" toward parser transitions inside the
operator. If it exposes (or can trivially expose) a boundary detector, reuse
it instead of duplicating. Confirm by reading the file; I did not inspect it
in this session.

**Reuse check result (done).** `fractal_voronoi.py`'s `_is_boundary` is the boundary of a hash-driven Voronoi partition of the index space, not a byte-content detector; nothing to reuse, so the detector lives in `pos_boundary.py` (`score_boundaries`, `top_boundaries`). Deviations from 5.4: the delimiter term requires `b[i] != b[i-1]` (otherwise a NUL/space padding run filled the whole top-K with ties); the entropy window must fit the scanned prefix (no entropy term in the first and last 16 bytes); candidates past a shrunken buffer are dropped rather than clamped; the unscored tail of a seed longer than 64 KiB is drawn uniformly with probability equal to its share. Unmeasured.

**Flag.** `--pos-boundary`; implied by `--position-arena` (cheap, cached);
in `--hail-mary`.

**Risks.** Format-specific: on high-entropy binary seeds (compressed
streams) boundaries are noise and the arm degrades to uniform, and Elo
will show it. Cost: O(n) per seed on first sight; watch corpus churn on large
seeds (the ffmpeg vendored binaries).

**Tests (`tests/test_pos_boundary.py`).** Score peaks at a known
class change/delimiter in a crafted buffer; run edges detected; `SCAN_CAP`
respected; top-K bound; jitter clamped; determinism per seed; empty and
1-byte buffers; cache LRU bound.

---

### 5.5 `pos_levy.py`: `PositionLevyScheduler` (arm `levy`)

**Hypothesis.** Gains cluster near previous gains, but at unknown scales.
`burn_front`'s Gaussian conduction has a fixed sigma (3 bins) and `fractal`
adapts resolution rather than jump length. A heavy-tailed jump from the last
gain offset explores locally most of the time and occasionally far, with
one integer of state per seed.

**Correction to the earlier chat note.** `core/zipf.py` is a *fitting*
toolkit (`fit_zipf`, `TailLaw`, `hurwitz`), not a sampler. Sampling is a
one-liner; `fit_zipf` is only relevant to the optional v2 (adapt alpha).

**Constructor.** `PositionLevyScheduler(rng)`; `name = "levy"`.

**State.** `OrderedDict[key] -> _Walk(anchor: int, misses: int, gaps: list[int])`,
`MAX_SEEDS = 256`. `gaps` is a ring of the last 64 distances between
consecutive gain sites (diagnostic/v2).

**`propose(data, buf_len)`.**
1. No walk or `anchor is None` -> `None`.
2. `u = max(rng.random(), 1e-12)`; step `s = floor(X_MIN * (u ** (-1/(ALPHA-1)) - 1))`
   with `X_MIN = 1`, `ALPHA = 2.0` (Lomax / Pareto II tail; unbounded tail);
   `s = min(s, buf_len)`; sign uniform. **As implemented, the `- 1` is a
   deviation from the first draft of this spec**, which had the plain Pareto
   `floor(X_MIN * u ** (-1/(ALPHA-1)))` ("median about 2 bytes"): that has a
   minimum step of 1, so the byte that gained could never be re-proposed. With
   the shift, `P(s = 0) = 1/2`, `P(s = 1) = 1/6`, `P(s >= k) = 1/(k+1)`.
3. `pos = anchor + sign * s`; **reflect** at the edges (triangle-wave fold,
   not a clamp, which would pile overshoots on byte 0 / the last byte). An
   anchor past a shrunken buffer is clamped to the last byte before the step.
4. Before step 2, `SPARK_RATE = 0.05` of draws are a uniform offset (the
   section 4 "keep a uniform escape" rule; this section's first draft did not
   list it). Draw order is spark, `u`, sign.

**`record(data, offsets, outcome, weight)`.**
- GAIN: `anchor = a uniformly chosen valid offset from offsets` (all
  offsets of a gain round are equally credited, matching the weight split
  convention); append distance to the previous anchor into `gaps` (only when
  the previous anchor was still live: none is recorded after a stale drop);
  `weight` is accepted for protocol parity and unused; reset
  `misses = 0`.
- MISS: `misses += 1`; at `STALE = 32` consecutive misses set `anchor = None`
  (drop the walk to the uniform floor rather than orbit a dead site).
- Drop `o < 0`; off-policy credited every round.

**Persistence.** `to_dict()/from_dict()` with `version`; per seed
`(anchor, misses, gaps)`. Small; keeps `--resume` parity with the other
learners.

**v2.** Fit alpha from `gaps` with `fit_zipf` once `>= MIN_SAMPLES = 64`
gaps have accumulated, clamped to `[1.5, 3.0]`; keep alpha per seed or
global (decide by measurement).

**Flag.** `--pos-levy`; implied by `--position-arena`; in `--hail-mary`.

**Risks.** It is an "exploit the last gain" arm: on seeds whose gains are
non-local it approximates uniform with a heavy tail, so expect it to help
only where locality exists. Sharing that assumption with `burn_front` means
the two may be redundant; the benchmark should say.

**Tests (`tests/test_pos_levy.py`).** Cold seed declines; a gain sets the
anchor; a miss streak of `STALE` clears it; step distribution is heavy-tailed
(scripted `random()` values map to expected steps incl. the `u -> 0` guard);
reflection at both edges and on shrunken buffers; negatives ignored; LRU
bound; round trip and malformed state; multi-offset gain picks a valid
offset.

---

## 6. Interaction matrix (what to watch for in Elo)

| new arm | most likely rival | expected relationship |
|---|---|---|
| `context` | `burn_front`, `kl_ducb` | complementary: cross-seed vs within-seed |
| `lineage` | `burn_front` | same locality assumption; lineage is the cold-start version |
| `cmplog` | `field` | overlap on comparison-relevant offsets; may be dominated |
| `boundary` | `fibonacci`, `region` | informed vs uninformed coverage prior |
| `levy` | `burn_front`, `fractal` | scale-free local search; may be redundant |

---

## 7. Benchmark plan (prerequisite: option A built; nothing measured yet)

> **Status.** Option A is implemented: `--pos-arena-arms` (`PositionArena(arms=...)`)
> and bench arms `pos-arena-uniform` (control, baseline `elo`),
> `pos-arena-{burn-front,kl-ducb,fractal,context,levy,round-robin,fibonacci}`
> (baseline `pos-arena-uniform`) and `pos-arena-all` (`ARENA_TESTABLE`,
> `POSITION_ARENA_ARMS` in `bench_paired.py`). Named `pos-arena-<n>`, not the
> `pos-arena-only-<n>` proposed below. `cmplog`, `lineage` and the tracker arms
> are not testable by subset alone (each joins only while its own feature is on;
> an arm would need that flag and stop being a one-knob change). The
> leave-one-out ablation (`arena{all minus X}`) is available from the flag but
> has no registered arms yet. Runs on fuzzgoat/png/ffmpeg are still to do.

**Existing gap.** No learned arm (`burn_front`, `kl_ducb`, `fractal`) has
an A/B measurement. `tools/lib/bench_paired.py` currently defines only
`pos-round-robin` and `pos-fibonacci` (`POSITION_ARMS`, `ARM_BASELINES`, both
paired against plain `baseline`).

**Problem found while planning.** Those two work as *standalone* arms
because the legacy `select_position` includes them as candidates. The new
arms (like `kl_ducb`/`fractal`) are arena-only, and `--position-arena`
implies every arm, so **a single arm cannot be toggled inside the arena
today**. Standalone `--pos-<n>` flags would be no-ops without the arena.

**Decision needed (pick one before building arms):**
- **A. Arena arm-subset flag** (recommended): e.g.
  `--pos-arena-arms uniform,fractal` (comma list; default all). Then the
  paired benchmark is `arena{uniform}` vs `arena{uniform, X}` for each arm
  `X`, plus `arena{all}` vs `arena{all minus X}` as the leave-one-out
  ablation. Bench arm names: `pos-arena-only-<n>` with baseline `pos-arena-uniform`.
- B. Add each arm to the legacy candidate list. Rejected: widens
  `_side_positions`' tuple contract and measures the wrong thing (uniform
  choice among candidates, not Elo arbitration).

**Protocol (same as existing bench arms).** `tools/benchmark.py` /
`tools/analyse_replicated.py` with `--baseline`; target `fuzzgoat` first
(fast, corpus in `tools/corpus_fuzzgoat.py`), then `png`, then ffmpeg for
large-seed behaviour; replicated runs, paired by seed; metric = edges found
at fixed exec budget plus time-to-edge. Add the three existing learned arms
to `POSITION_ARMS` and `ARM_BASELINES` in the same change so the new arms
have peers. Run the `tests/test_bench_paired_pos_arms.py` invariant (every arm
is its baseline plus added flags).

**Acceptance criteria per arm (proposal).** Beats `uniform` in Elo (not flagged
by the canary inspection) **and** non-negative paired delta versus arena
without it, on at least two targets. An arm that ties uniform is not worth
its flag; delete rather than keep.

---

## 8. Build order and sizing

Rough cost is relative to `pos_fibonacci` (72 lines) and `pos_fractal` (275
lines + 227 test lines):

0. **Benchmark prerequisite** (section 7): arena arm-subset flag, bench
   arms for existing learners. Without this, everything below is unmeasured.
1. `lineage`: smallest; data already exists; useful as the first proof of
   the `meta_of` injection pattern.
2. `boundary`: stateless; the one design question is reusing
   `fractal_voronoi`'s boundary detector; share `_bytecls.py` with `context`.
3. `cmplog`: needs the gate/attribute names confirmed; tracker-style wiring.
4. `levy`: small; persistence follows the burn-front template.
5. `context`: largest and highest hypothesised value; build once `_bytecls.py`
   and the benchmark exist.

**Per-arm definition of done:** class + docstring, wiring in 6 places,
`tests/test_pos_<n>.py`, arena test updated, bench arm added, and the
affected test files run. Note the 3 pre-existing kwargs-order failures
(`target_schedule` must be last) reproduce on a clean HEAD; they are not
caused by this work.

---

## 9. Not done / open questions

- Nothing here is implemented or measured.
- `RandPool` API (`weighted_choice`, `random`, `randint`) not re-verified.
- `fractal_voronoi.py` boundary detector not inspected (reuse for `boundary`).
- cmplog/redqueen enabling attribute names for the `cmplog` gate not confirmed.
- Whether `core/analyzer_registry.py` derives `pos_` keys from
  `POSITION_STRATEGY_NAMES` or lists them by hand.
- `lineage` v2 needs a parent-bytes accessor from `parent_key`.
- `docs/architecture.dot` remains stale from 2026-09-24.
- ~~The 2026-09-24 handover's "burn-front not persisted" item is stale.~~ Confirmed and corrected 2026-09-28 (persistence is wired via `_state_store` in `services/fuzzer.py`).
