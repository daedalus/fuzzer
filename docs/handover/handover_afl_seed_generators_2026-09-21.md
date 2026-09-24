# Handover — seed generator ports from AFL / aflgo / AFLplusplus

**Date:** 2026-09-21
**Status:** analysis only (four explore subagents; no code written)
**Scope:** which seed-*generation* capabilities in `~/code/AFL`, `~/code/aflgo`,
`~/code/AFLplusplus` are worth porting into fuzzer-new, given what already exists.

## Trigger

"Analyze what seed generators can we port from ~/code/[AFL,aflgo,AFLplusplus]."
The fuzzer already generates; the question was which upstream *generation* shapes
fill a real gap. Answers are verified against upstream source paths below.

## What upstream actually has (summary)

### AFL 2.x
- No standalone seed synthesizer. Generation-adjacent machinery:
  - **auto-extras token harvesting** — `afl-fuzz.c:1803` `maybe_add_auto()`, magic-string sniffing during the bitflip stage; persists to `out/queue/.state/auto_extras/`.
  - **deterministic stage sweep** — `afl-fuzz.c` `fuzz_one()` (~5000): bitflip 1..32 → arith8/16/32 → interest8/16/32 → extra insert/overwrite; every coverage-new output saved as a new queue seed by `save_if_interesting()` (~3163).
  - splice (`~6575` `retry_splicing`), havoc (`~6119`).
  - `afl-cmin` (greedy set-cover corpus reduction), `afl-tmin.c` (input minimizer), `afl-showmap.c`, `afl-analyze.c`.
  - `libtokencap/libtokencap.so.c` — LD_PRELOAD strcmp/memcmp token capture → dict.
  - `experimental/post_library/post_library_png.so.c` — on-the-fly PNG chunk/CRC repair of mutated buffers.
  - `testcases/`, `dictionaries/` — static assets.

### aflgo
- Nothing to port for generation. Only directed *prioritization* of supplied seeds:
  - `afl-2.57b/afl-fuzz.c:4914` `calculate_score()` distance-annealed power schedule (`-z`, `-c`).
  - `distance/gen_distance_fast.py` — CG/CFG distance tables feeding the schedule.
  - `examples/*.sh` — trivial corpus bootstraps (empty file, magic bytes, project's own tests).
- Everything else (afl-cmin/tmin/testcases/dicts/post_library) inherited unchanged.

### AFLplusplus 5.02c
- Real generation payload lives in `custom_mutators/`:
  - **Gramatron** `custom_mutators/gramatron/` — PDA grammar automaton (JSON) generative mutator.
  - **libafl_nautilus** `custom_mutators/libafl_nautilus/` — Nautilus grammar fuzzer; `cargo run --bin dump_inputs` writes batch generated seeds as raw bytes.
  - **grammar_mutator** — stub in tree; generated via `build_grammar_mutator.sh` (external repo).
  - **autotokens** `custom_mutators/autotokens/` — token-level fuzzing; `AUTOTOKENS_CREATE_FROM_THIN_AIR=1` builds the first queue entry purely from dict tokens.
  - Radamsa port (`custom_mutators/radamsa/`), atnwalk, XmlMutatorMin, libprotobuf-mutator-example.
  - **TritonDSE** `custom_mutators/aflpp_tritondse/aflpp_tritondse.py` — Python concolic explorer; pre_exec hook writes solved inputs as new queue seeds.
- In-fuzzer construction: Redqueen/CmpLog `input_to_state_stage` (`src/afl-fuzz-redqueen.c`), **FrameShift** (`src/afl-fuzz-frameshift.c`, size/offset field auto-repair after insert/delete).
- `afl-addseeds` (inject files into a running campaign), autodict_ql (CodeQL static dict), compiler autodict (`AFL_LLVM_DICT2FILE`).
- Note: `-g`/`-G` is NOT generation — it clamps produced input length (AFL_INPUT_LEN_MIN/MAX).

## Already present in fuzzer-new — do not re-port

| Capability | Location |
|---|---|
| Grammar generation + Boltzmann | `core/grammar.py:267` `generate()`, `:434` `generate_boltzmann()` |
| Markov seed generation | `core/markov.py:105` `generate()`; `SeedPicker._pick_markov_seed` (`seed_picker.py:706`) |
| 35 `_generate_random_<fmt>` synthesizers | `core/mutations/` (png, jpeg, webp, zip, sqlite, gif, bmp, …) |
| Versifier / tree / cycle-lemma generation | `_op_versifier_generate` (`operators.py:3119`), `_op_tree_generate` (`operators.py:1194`) |
| Scheduler that synthesizes a seed | `core/schedulers/seed_kruskal_count.py:224` `generate(anchor, donors)` |
| Redqueen + constraint-derived candidates | `core/cond_stmt.py:345`, `core/rq_encodings.py:334` |
| aflgo distance-*selection* | `SeedPicker._pick_aflgo_seed` (`seed_picker.py:596`); `_weight_entropy_and_distance` (`seed_picker.py:1134`) |
| cmin / tmin equivalents | `fuzzer-tool minimize` (`services/minimize.py`), `fuzzer-tool tmin` (`services/tmin.py`) |
| Corpus import + autotokens | `fuzzer-tool import` (`services/import_corpus.py`, incl. `build_autotoken_dictionary`) |
| Radamsa-band operators | `core/mutations/__init__.py` + REGISTRY band `radamsa` |

## Port candidates (ranked)

### Tier 1 — port now, cheap, pure-Python, no new deps
1. **AFL deterministic-stage seed bootstrap.** Sweep fresh seeds through
   bitflip 1/4/8/32 → arith → interesting-values, keeping coverage-new outputs as
   corpus seeds. Not the bandit ops (those mutate in-loop); the AFL stage
   *sequence as a seed factory* is absent (`_is_deterministically_redundant` at
   `operators.py:4297` is the dedup half only). Insert as a bootstrap operator +
   `genseed` subcommand.
   **Shipped 2026-09-24** as the `afl_det` arm (`--op-afl-det`). `genseed`
   half not done: an offline sweep needs target execution to keep only
   coverage-new outputs, which `genseed` does not do.
2. **`fuzzer-tool genseed` subcommand.** Expose existing `_generate_random_<fmt>`,
   `Grammar.generate`, `markov.generate` as a corpus-writing CLI. Plumbing only;
   model on `cmd_minimize` / `cmd_import` (`cli/commands.py`), write via
   `import_corpus._write_seed` / `filesystem.save_to_corpus`.
3. **Autotokens from-thin-air.** Synthesize an initial corpus purely from a
   loaded dictionary when starting from garbage. Fixes the self-heating gap:
   `_FORMAT_BOOTSTRAP_RATE = 0.02` (`operator_registry.py:529`) never raises
   format-synthesized seeds to population scale on an empty corpus.

### Tier 2 — medium
4. **Auto-extras in-loop token harvesting** (AFL `maybe_add_auto`) — coverage-
   checksum magic-byte sniffing feeding the dict operators live; distinct from
   today's import-time autotokens.
5. **FrameShift-style size/offset repair** — post-mutation field fixup so
   generated/mutated inputs stay structurally valid.
6. **`afl-addseeds`** — live-campaign seed injection (tiny; corpus_manager already
   has `save_to_corpus`).

### Tier 3 — large or low value
7. **Solver-driven genesis** (TritonDSE / laf-intel shape) — overlaps redqueen;
   pulls in `tritondse` dependency. Only the redqueen-into-standalone-factory
   slice is worth it.
8. **aflgo directed *synthesis*** — nothing upstream to port; aflgo only
   re-weights selection (already ported). Would be a new feature needing per-BB
   distance profiling.
9. **Nautilus / Gramatron / Radamsa / LPM** — subsumed by existing
   grammar/markov/radamsa facilities.

### Rejected
- `afl-cmin` / `afl-tmin` — `minimize` / `tmin` already exist.
- `libtokencap` — needs LD_PRELOAD; import-time autotokens cover the use case.
- classic `post_library` — format-aware mutators already exist.

## Design rules (from `docs/port-backlog.md`)
1. A generator must be an **arbitrated arm** (Elo/bandit), never a fixed pipeline
   stage — it can decay to near-zero share.
2. Cost budget ~1 ms per output; bound work at the caller (WFC cost law,
   `docs/learnings/2026-08-10-wfc-pixel-hang-cost-law.md`).
3. The case for generators is **cold start and plateaus**, not steady state.
4. fitzgen structure-aware experiment (the one properly-powered source): mutation
   beats generation 36–49% at 5 min, 1–2% at 24 h — hence 1–3.

## Wire-up for Tier 1 (TDD per Hard Rules 37/38)

- `genseed` subcommand: `cli/commands.py` new parser + `cmd_genseed`, writing through
  `adapters/filesystem.save_to_corpus` (or `import_corpus._write_seed`), honoring
  corpus dir defaults from `_get_dirs` (`cli/commands.py:129`).
- Deterministic bootstrap operator: register in `_CATEGORIES` (one entry) + optional
  `_AVAILABLE` predicate + `_op_<name>` handler on `OperatorEngine`
  (`services/operators.py`). Every new `_op_` needs a handler or dispatch-build raises.
- Thin-air autotokens: reuse existing dict machinery; `build_autotoken_dictionary`
  (`import_corpus.py:220`) is the reference for a token source.
- Regression tests: `test_regression_<brief>`; assertion `assert len(X) >= N`, never
  `==` (operators get added); one falsification + one adversarial test per feature
  (Hard Rule 23); scripted RNG via `tests/support/scripted_rng.py` (Hard Rule 39).
- Verify no speed regression (Hard Rule 41); run affected tests only, not the ~10k
  full battery (Hard Rule 50).

## References
- Classic AFL 2.x: `~/code/AFL` (`afl-fuzz.c`, `afl-cmin`, `afl-tmin.c`,
  `libtokencap/`, `experimental/post_library/`)
- AFLGo: `~/code/aflgo` (`afl-2.57b/afl-fuzz.c`, `distance/gen_distance_fast.py`,
  `examples/`)
- AFLplusplus 5.02c: `~/code/AFLplusplus` (`custom_mutators/{gramatron,libafl_nautilus,autotokens,radamsa,aflpp_tritondse}`,
  `src/afl-fuzz-redqueen.c`, `src/afl-fuzz-frameshift.c`, `afl-addseeds`)
- Prior generator audit that inspired this one:
  `docs/handover/handover_generators_2026-09-20.md`
