# Handover - seed generators from Angora / go-fuzz / honggfuzz / wtf

**Date:** 2026-09-21
**Status:** analysis only (four explore subagents; verified against local code with
targeted greps). Complements `handover_afl_seed_generators_2026-09-21.md`.
**Base:** this repo; no code written.

## Trigger

"Analyze what seed generators can we port from ~/code/[Angora, go-fuzz, hongfuzz, wtf]."
Completes the three failed surveys in `handover_ports_pending.md` sections 5-7
(Angora, go-fuzz, hongfuzz) and surveys wtf (done nowhere else).

## What each repo actually has

### Angora (mutation-based taint-guided; Rust v1.3.0)

- No from-scratch/grammar seed generator; panics on an empty seed dir (`fuzz_main.rs`).
- Canonical seed-writer is coverage-gated only: `Executor::do_if_has_new` ->
  `Depot::save` -> `queue/id:NNNNNN` (`executor/executor.rs`, `depot/depot.rs`).
  Taint/gradient search produce candidate buffers, never direct saves.
- The taint -> `CondStmt` -> search pipeline is a byte-synthesis engine:
  - `track/fparser.rs` `get_offsets_and_variables` - concatenates captured magic
    bytes (`save_magic_bytes`, `runtime/src/track.rs`) or encodes the compared
    constant (`write_as_ule`) into `cond.variables`.
  - Search ops: `search/mb.rs` (write magic bytes), `cbh.rs` (climb-hill),
    `gd.rs` (gradient descent, +-1 partial derivatives), `one_byte.rs`
    (exhaustive 0..255), `det.rs` (bitflips over tainted region),
    `afl.rs` (havoc + two-seed splice), `exploit.rs` (write all interesting
    values), **`len.rs` (LenFuzz `COND_LEN_OP`)**, `cmpfn.rs` (FnFuzz:
    insert/remove + per-byte delta to match the compared string).
  - `runtime/src/tag_set.rs` `infer_shape` - DFSan label tree -> input byte
    offsets and multi-byte type grouping (the "type inference", not ML).
- Tools: `tools/lava_validation.py` (post-run crash replay/selection),
  `sync_afl` (imports AFL queue seeds), `parse_track_file` (offline CondStmt dump).

### go-fuzz (dvyukov)

- **Versifier** (`go-fuzz/versifier/versifier.go`, 911 lines) - structural
  generative synthesizer for text inputs. Tokenize (4-state DFA over UTF-8) ->
  structure (`extractNumbers` 0x/e/-/. merging -> brackets -> key-value -> lists
  -> lines) -> node tree with per-node dicts; `Generate` mixes dict replay
  (4/5, 1/2, 9/10) with fresh synthesis; `RandNode` cross-block recall;
  `Rhyme()` picks a random block. Runs 1/10 of iterations (`worker.go`), rebuilt
  incrementally from every accepted input (`hub.go:250`).
- **Mutation engine** (`mutator.go`, 20-case table). Generative entries: insert
  random byte block (1), block dup (2), block copy-over (3), **number
  regeneration (case 15: replace ASCII integer with a fresh value; distributions
  `rnd(1000)`, `randbig()`, `randbig()*randbig()`, `-randbig()`; sign-flip;
  length re-splice)**, two-input splice (16), cross-input chunk insert (17),
  literal insert/replace (18/19 from statically harvested source literals - no
  `.dict` files in-tree).
- **Sonar** (`sonar.go`): comparison hint generation - replaces operand v1 with
  v2 in the input across encodings (lower/upper case, incr/decr, big-endian
  mirror, **base-128 uvarint +-1**, ASCII decimal, hex) plus a string
  **length-byte +diff tweak**.
- Bootstrap: single **empty `[]byte{}` seed** injected when the corpus is empty
  (`coordinator.go:60`); user seeds auto-minimized on connect; the target is a
  pure consumer (never generates seeds).
- `gen` package (`go-fuzz/gen/main.go`) - offline `gen.Emit(data, hint, valid)`
  corpus-writer API for the toy `examples/*/gen` generators.

### honggfuzz (this fork)

- **ELF/BFD dictionary synthesis** - `linux/bfd.c`:
  `arch_bfdExtractStrArray` (named pointer-array symbols like Lemon `yyTokenName`,
  Bison `yytname` -> dict tokens), `arch_bfdExtractRodataStrArrays`
  (auto-discovered `.data`/`.rodata` pointer arrays), `arch_elfCollectRoValues`
  (`ro32[]`/`ro64[]` - every aligned 4/8-byte word in `.rodata`, 128K entries,
  deduped, binary-searched by `libhfuzz/instrument.c:477` for cmp solving).
  All feed the shared 32K-token dict (`dict.c`, FNV-1a dedup).
- **Self-fertile dynamic corpus** (`input.c`): `input_addDynamicInput` saves
  coverage-new inputs to `--covdir_new` (CRC64 names) - "start fuzzing without an
  input corpus"; `input_getDiverseInputAsBuf` picks crossover partners via a
  16-window max-`cov[0]` scan plus lineage bonus.
- **External generator hooks** - `--mutate_cmd` / `--pprocess_cmd`
  (`input.c:906/939`): an external program fully generates or post-processes
  testcases.
- Runtime constant capture (`libhfuzz/memorycmp.c` `instrumentAddConstStr`) -
  in-process strcmp args feed the feedback dict.

### wtf (0vercl0k)

- Exactly **one** true synthesizer: `CustomMutator_t::Generate()` in
  `src/wtf/fuzzer_tlv_server.cc:204-365`, registered for the `tlv_server` JSON
  target. 20% of the time (1-in-5 roll) it generates instead of mutating: N = rnd(1,10)
  packets, per packet Command = rnd(0,10), Body = rnd(0,100) zero bytes, BodySize =
  Body.size() then with prob 1/3 XOR-flips one of the low 16 bits (deliberate
  constraint breach). Serialized to JSON.
- Corpus growth is pure selection: master keeps an input only if the client's
  coverage GVA set grew globally (`server.h` `HandleNewResult` -> `OnNewCoverage` +
  `Corpus_.SaveTestcase`, blake3-hash-deduped into `outputs/`).
- **Empty corpus aborts** (`mutator.cc:27`) - antithesis of our bootstrap.
- No EVM support (prompt premise wrong), no `mutate`/`minimize`/`generate`
  subcommands (only `master`, `run`, `fuzz`); minset = `master --runs=0`.
- Linux-mode QEMU/GDB snapshot generator produces MEM dumps (execution state,
  not input seeds) - out of scope.

## Already present in fuzzer-new - do not re-port

| Capability | Where |
|---|---|
| Angora magic-bytes + climb-hill search | `core/mb_cbh.py:75` `magic_byte_search`, `:124` `climb_hill` (ported) |
| Angora gradient descent | `core/gradient_descent.py` (ported; see `handover_optimization_algorithms_2026-09-15.md`) |
| Magic-byte / cmp-operand solving | redqueen `_op_redqueen` (`operators.py:3365`), `core/rq_encodings.py` generate_mutations, `core/cond_stmt.py:345` |
| go-fuzz versifier | `_op_versifier_generate` (`operators.py:3119`) - verify fidelity, see Tier 2.1 |
| Number regeneration | `operators.py` "Replace a whole multi-digit ASCII number with a random value" + Radamsa number + ASCII-int arithmetic ops |
| Sonar-style encodings | `rq_encodings.py` encoders incl. reverse option; ASCII decimal/hex/octal; split; cstring. Missing base-128 uvarint, see Tier 2.2 |
| Length-field solving / repair | `_op_length_offset_goal` (`operators.py:3689`), `_op_field_repair` (`operators.py:3718`, dependency-ordered, FrameShift-like) |
| Length grow/shrink/truncate/boundary | `_op_length_grow/shrink/truncate/boundary/miscalculate` (`operators.py:2295-2322,4056`) |
| Empty-corpus fallback | synthetic `b"AAAAAAAA"` (`filesystem.py:402,549`) + 2% format bootstrap trickle |
| Self-fertile corpus growth / crossover partner | `save_to_corpus` admission + `_op_splice*` families |
| Corpus minimization + crash replay | `fuzzer-tool minimize`, `tmin`, `import` (`import_corpus.py:220` autotoken dict) |
| Taint-offset selection of bytes to mutate | cmplog offsets + `_op_colorization` (`operators.py:1106`) |

## Port candidates (ranked)

### Tier 1 - port now, pure-Python, no new deps

1. ✅ **Honggfuzz target-binary dictionary synthesis.** Statically mine the *target
   executable* like `linux/bfd.c`: (a) string-table pointer arrays (Bison `yytname`
   / Lemon `yyTokenName` symbols) -> dictionary tokens; (b) `.rodata`/`.data`
   aligned 4/8-byte words -> a deduped constant table for cmp-helping. fuzzer-new
   already parses ELF (`core/elf.py`), so no libbfd dep - pure-Python re-read.
   Distinct from import-time autotokens (`import_corpus.py`) and AFL in-loop
   auto-extras. Distributes across formats via `-x`-style dict operators + a new
   "rodata constant" interesting-values table. (SHIPPED 2026-09-21: `core/elf.py`
   `extract_data_word_constants` + `_iter_sections`; `target_profiler.py`
   `rodata_word_constants` field and parser-token pointer-array walks;
   `fuzzer.py` `_merge_profile_dictionary`.)
2. **Angora LenFuzz (`COND_LEN_OP`) length-constraint search.** Solve
   read-length-vs-file-length comparisons by resizing the buffer: extend by
   `delta*size+1` with random bytes, append special chars (0,10,13,32), and test
   truncations (exact / `<`). The existing length ops are mutational, not
   gated on a recorded cmplog length record; `_op_length_offset_goal` solves
   offset/size pairs, not whole-buffer length checks. Needs a cmplog length
   channel - verify what `core/cond_stmt.py` records today.

### Tier 2 - verify or small

3. **go-fuzz versifier fidelity audit** (not a new port). `_op_versifier_generate`
   exists; audit it against `versifier.go` for the exact mechanism: per-node dict
   replay probabilities, `extractNumbers` merging, `RandNode` cross-block recall,
   and the incremental shared verse rebuilt from accepted inputs.
4. **Sonar exotic encodings.** Add a `base-128 uvarint (+-1)` encoder to
   `rq_encodings.py` if missing; verify big-endian mirror (= `reverse` option) and
   hex/octal/decimal (= `AsciiEncoder`) are already represented.
5. **wtf generate-with-deliberate-length-corruption** idea. The reusable shape is
   "synthesize structurally valid input, then break one claimed field on purpose"
   (1/3 corrupt BodySize) - target length checks. Partial overlap with format
   mutators + `_op_length_miscalculate`; propose as a format-aware generate arm,
   not a new engine.

### Rejected

- Angora `GdSearch`/`Grad` gradient machinery - `core/gradient_descent.py` exists.
- Angora `MbSearch`/`CbhSearch`/`ExploitFuzz`/`DetFuzz`/`OneByteFuzz` - magic bytes,
  climb-hill and deterministic tables already ported; one-byte exhaustive and
  tainted-bitflip are trivial instantiations of existing small-setting operators.
- Angora DFSan taint tracker / `tag_set.rs` shape inference - full LLVM/DFSan
  instrumentation pipeline, not a pure-Python port; cmplog offsets cover the
  "which bytes matter" ask.
- go-fuzz `gen` package - trivial `Emit()` helper; the `genseed` subcommand already
  planned in `handover_afl_seed_generators_2026-09-21.md` Tier 1.2 subsumes it.
- go-fuzz smash / minimization-intermediates - minimizer already exists.
- go-fuzz empty `[]byte{}` bootstrap - our fallback is richer.
- honggfuzz dynamic-input diversity crossover partner (`input_getDiverseInputAsBuf`)
  - nice-to-have variant of `_op_splice_diff_located`; low ROI.
- honggfuzz `--mutate_cmd` / `--pprocess_cmd` - harness feature, not generation;
  candidate for a future CLI flag.
- wtf custom-mutator-per-target factory - fuzzer-new's format-aware mutators +
  REGISTRY cover it.
- wtf empty-corpus abort - anti-pattern.
- wtf QEMU/GDB snapshot generator - VM state snapshots, out of scope.

## Wire-up notes (if Tier 1 is approved)

- Item 1: new `_CATEGORIES` entry (e.g. `dict` or `adaptive` band) + optional
  `_AVAILABLE` predicate gated on an ELF target + `_op_<name>` handler on
  `OperatorEngine`; or a one-shot pre-fuzz enrichment in `_load_corpus`/import.
  Reuse `core/elf.py` (read program headers/sections/symbols + `.rodata`).
  Regression: falsification + adversarial test per Hard Rule 23; scripted RNG
  (Hard Rule 39); `assert >=` bounds.
- Item 2: needs cmplog length records; gate on `_has_cmplog_pairs`-style
  predicate. TDD per Hard Rules 37/38 (test -> feature).
- Verify no speed regression (Hard Rule 41); run affected tests only (Hard Rule 50).

## References

- Angora: `/home/dclavijo/code/Angora/fuzzer/src/{depot,executor,search,track}/`,
  `runtime/src/{track.rs,tag_set.rs}` - this is a Rust rewrite (v1.3.0); the
  classic paper pipeline lives in the C++ history.
- go-fuzz: `/home/dclavijo/code/go-fuzz/go-fuzz/{versifier/versifier.go,mutator.go,
  sonar.go,worker.go,hub.go,coordinator.go}`
- honggfuzz: `/home/dclavijo/code/honggfuzz/linux/bfd.c`, `input.c`, `dict.c`,
  `libhfuzz/{instrument.c,memorycmp.c}`
- wtf: `/home/dclavijo/code/wtf/src/wtf/{fuzzer_tlv_server.cc,mutator.cc,server.h}`
- Prior seed-generator handover: `docs/handover/handover_afl_seed_generators_2026-09-21.md`
