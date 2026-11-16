# Action plan: mutant-kill A/B metric (corpus replay on fuzzgoat mutants)

Source: Görz et al., "Systematic Assessment of Fuzzers using Mutation
Analysis", USENIX Sec '23 (arXiv 2212.03075). Port only its cheap Phase I:
replay a finished corpus against a fixed mutant set, count kills. No
supermutants, no per-mutant fuzzing (paper: 3 of 690 stubborn mutants killed
in 24 h; Phase II buys almost nothing).

**Why.** 62 ledger rows score on edges; 83 of 149 rows are NULL. Edges
saturate. Kills ask a different question: does the corpus *expose* faults,
not merely reach code.

**Expectation (paper §5.2).** 97.5% of kills come from coverage alone, so for
plain coverage arms kills track edges. Discrimination is likeliest for
targeting arms (cmplog, grammar, dictionaries, IJON).

```
 build once (cached by key)                        per A/B cell
 ─────────────────────────                         ────────────
 fuzzgoatNoVulns.c                                 fuzz run ─► workdir corpus
   │ clang -fsanitize=address                                    │ iter_seed_batches
   │   -Xclang -disable-llvm-passes -emit-llvm -S                ▼
   ▼                                                        pack (u32 len|bytes)*
 base.ll ──core/ir_mutants──► site list ──sample(RandPool)       │
   │                              │                              ▼
   │                        mutant_k.ll ──clang -O2 asan──► mutant_k (fork driver)
   ▼                                                             │ status per input
 original (fork driver) ──► statuses ──► clean-input mask ───────┤
                                                                 ▼
                                    kill_k = ∃ i: orig[i]=CLEAN ∧ mut_k[i]=CRASH
                                                                 ▼
                                      row["kills"] ─► compare(metric="kills")
```

## Evidence (spike, 2026-11-16, clang 18.1.3, this container)

| Check | Result |
|---|---|
| Textual `.ll` mutation compiles | yes; `llvm-as` verifies |
| ASAN on mutated `.ll` | **only** if the frontier ran with `-fsanitize=address` (attaches `sanitize_address`); emit with `-Xclang -disable-llvm-passes`, instrument at the `.ll` → binary step. Without it: no ASAN, no kill. |
| Kill oracle on toy (`n < 4` → `n <= 4`, `malloc(4)`) | killed by `abcd` only; `abc`, `abcdefg` clean on both |
| Fork driver, 500 inputs, `ASAN_OPTIONS=symbolize=0:...` | 1.95 s (≈ 4 ms/input); symbolized: 36.9 s |
| `fuzzgoat.c` original on 5 stock seeds | 4/5 crash (planted bugs) → most inputs unusable as kill evidence |
| `fuzzgoatNoVulns.c` original on same seeds | 0/5 crash |
| NoVulns `.ll` sites | 155 `icmp`, 155 `br i1`, 322 `store` |
| Container ASAN runtime | missing by default; `apt-get install libclang-rt-18-dev` fixes it |

Decision: mutate **`fuzzgoatNoVulns.c`** (same parser, planted bugs removed).
Corpora still come from fuzzing `fuzzgoat.c`; inputs are JSON either way.

## P1 — `core/ir_mutants.py`: site enumeration + apply (pure, no I/O) (shipped)

Measured on fuzzgoatNoVulns: 612 sites (store 230, br 155, eq/ne 140,
add/sub 52, boundary 15, sign 15, call 3, alloc 2); all 612 mutants pass
`llvm-as`. `find_sites` 3.9 ms (gate-union pre-filter; 4.9 ms without).

Line-level operators on `-O0`-shaped IR, subset of paper Table 10:

```python
class MutOp(Enum):
    ICMP_BOUNDARY = 1  # slt<->sle, sgt<->sge, ult<->ule, ugt<->uge
    ICMP_SIGN = 2  # s<pred> <-> u<pred>   (SIGNED_TO_UNSIGNED, ...)
    ICMP_EQ_NE = 3  # eq <-> ne
    BR_NEGATE = 4  # br i1 %c, label %a, label %b -> swap labels
    STORE_DELETE = 5  # drop a store (DELETE_STORE_PATTERN)
    ARITH_SWAP = 6  # add <-> sub           (SWITCH_PLUS_MINUS)
    ALLOC_SHRINK = 7  # malloc/calloc/realloc size - 16 (MALLOC/CALLOC)
    CALL_DELETE = 8  # void call, non-intrinsic (DELETE_CALL_INSTRUCTION)


@dataclass(frozen=True, slots=True)
class Site:
    line: int
    op: MutOp


def find_sites(ll: str) -> list[Site]: ...
def apply_site(ll: str, site: Site) -> str:
    ...  # exactly one line changed
    # (ALLOC_SHRINK: +1 `sub` line)


def sample_sites(sites, cap: int, rng) -> list[Site]: ...  # RandPool (Rule 16)
```

Not sites: `llvm.*` intrinsics (`dbg`, `lifetime`), unconditional `br label`,
text inside string constants/metadata, entry-block spills of arguments
(`store ptr %x, ptr %x.addr`: compiler artefacts, not source statements —
deleting them makes trivially killed garbage pointers).

## P2 — `adapters/mutant_build.py` + `adapters/mutant_replay.c` (shipped)

Measured (fuzzgoatNoVulns, 4 cores): 467 kept, 145 equivalent, 0 failed;
build 43.5 s, 34 MB; cached rebuild instant. 5 stock seeds kill 63/467 in
22 s, dominated by ~47 ms process start per mutant binary. Deviations from
the sketch: shared ASAN runtime + rpath (23x smaller binaries, same speed);
IR compiled from stdin (ASAN embeds the input file name in `.rodata`, which
defeated TCE); TCE hashes the whole object, not `.text` (a mutation can
change only `.rodata`, e.g. a jump table).

- `emit_ir(src, out)`: `clang -O2 -fsanitize=address -fno-delete-null-pointer-checks -Xclang -disable-llvm-passes -emit-llvm -S` (flags as `compile_fuzzgoat_object`).
- `build_mutant(ll, out)`: `.ll` → `.o` → link with wrapper object
  (`targets/fuzzgoat_read.c -Dmain=fg_unused_main`), `mutant_replay.c`, and a
  no-op `__afl_map_edge`. **Rule 9 exception, stated:** these are oracle
  binaries, not fuzz targets — ASAN yes, AFL shim no (precedent:
  `fuzzgoat_read_cov` + `frontier_stub.c`).
- Trivial-compiler-equivalence: hash each mutant's `.text` (`hash_data`);
  identical to original or to an earlier mutant → drop (equivalent/duplicate).
- `replay(binary, pack) -> array('b')`: statuses `CLEAN=0 CRASH=1 HANG=2`.
  Driver reads the pack, forks per record, `alarm()` timeout, exits early at
  the first CRASH on an index the parent marks clean (`--stop-on-kill MASKFILE`).
  `ASAN_OPTIONS=symbolize=0:detect_leaks=0:print_summary=0:malloc_context_size=0`.
- Pack, not file paths: corpora may be delta-encoded or in `seeds.zip`;
  `iter_seed_batches` streams them (memory bounded, Rule 54). Record cap =
  the fuzzer's max input size constant.
- Manifest (`manifest.pkl`: key, sites, dropped-equivalents, src hash, clang
  version) via existing pickle machinery (`state_store.safe_load`, Rule 15).

```c
/* mutant_replay.c: argv[1]=pack argv[2]=clean-mask|- ; stdout: status bytes */
static int run_one(const unsigned char *b, size_t n, unsigned timeout_s)
{
    pid_t pid = fork();
    if (pid == 0) { alarm(timeout_s); fuzz_shm_run(b, n); _exit(0); }
    int st; waitpid(pid, &st, 0);
    if (WIFSIGNALED(st)) return WTERMSIG(st) == SIGALRM ? ST_HANG : ST_CRASH;
    return WEXITSTATUS(st) ? ST_CRASH : ST_CLEAN;   /* ASAN exits 1 */
}
```

## P3 — `services/mutant_kill.py` (shipped)

Measured (fuzzgoatNoVulns, 5 stock seeds): 63 kills, 5 hangs, 0 flaky,
identical on rerun; 23.1 s at 1 job, 7.4 s at 4. Added `KillScore.flaky` and
`Confirm` (NONE/RERUN) for the kill re-run from Risks; `pack_record` in the
adapter reads the killing input back.

```python
@dataclass(frozen=True, slots=True)
class KillScore:
    kills: int
    hangs: int  # mutant HANG on a clean input; not a kill (paper)
    inputs: int
    clean_inputs: int  # original CLEAN; the only usable evidence
    killed: frozenset[int]  # mutant ids, for exploratory union diffs


class KillScorer:
    def __init__(self, kill_set: Path, jobs: int): ...  # loads manifest
    def score(self, corpus: Path) -> KillScore: ...
```

Original replay once → clean mask → mutants in a process pool, each early
stopping on first kill. Inputs that crash or hang the original never count.

## P4 — CLI (`cli/commands.py`, `cmd_replay` pattern)

```
fuzzer-tool mutants build SRC.c --wrapper targets/fuzzgoat_read.c -o DIR [--cap N] [--seed S]
fuzzer-tool mutants score DIR -d CORPUS [-j N]      # prints kills/hangs/clean_inputs
```

## P5 — bench wiring (`tools/lib/bench_paired.py`)

- `--kill-set DIR`: in `run_cell`, after `run_timed` and next to
  `_real_edges(workdir)`, before the `rmtree`: `row["kills"]`,
  `row["kill_hangs"]`, `row["clean_inputs"]`. Scored **outside** the timed
  run: `secs`/`cpu_user` unchanged (Rule 41).
- `_PLAIN_METRICS += ("kills",)`; `compare(..., metric="kills")` then works
  unchanged (higher is better).

## Statistics

- **Unit = seed.** Kill counts per (target, seed) cell → `compare` W/L/T,
  `mcnemar_exact(wins, losses)`, Wilcoxon. Mutants within one corpus are
  correlated: a per-mutant McNemar would overstate n. Per-mutant union
  differences are reported descriptively, no p.
- **Controls (Rule 46), both must pass before reading any A/B:**
  1. Determinism: same corpus scored twice → identical `killed` set.
  2. A/A: `baseline` vs its `-ctl` arm on `kills` → not significant.
- **Stop criterion, set now:** run the A/A first (20 seeds × 10k). If kill
  counts are flat across seeds (all cells equal, or spread < 1 kill) the
  metric cannot separate arms on fuzzgoat → ledger row `NULL`, stop; consider
  a bigger target. Otherwise A/B one targeting arm (cmplog) vs baseline,
  scored on both `edges` and `kills`; the interesting outcome is a `kills`
  WIN with `edges` NULL.

## Cost

Driver ≈ 4 ms/input (measured). Mutants after sampling/TCE ≈ 600–900
(estimate from site counts). Per cell worst case 900 × C clean inputs × 4 ms,
divided by `-j`; early stop cuts killed mutants to ~1 replay. Mutant build:
one clang per mutant, once, cached under the manifest key — unmeasured.

## Wiring (every touch point)

| File | Change |
|---|---|
| `src/fuzzer_tool/core/ir_mutants.py` | new: `MutOp`, `Site`, `find_sites`, `apply_site`, `sample_sites` |
| `src/fuzzer_tool/adapters/mutant_build.py` | new: `emit_ir`, `build_mutant`, `replay`, pack writer, manifest |
| `src/fuzzer_tool/adapters/mutant_replay.c` | new: fork driver + `__afl_map_edge` no-op |
| `src/fuzzer_tool/services/mutant_kill.py` | new: `KillScore`, `KillScorer` |
| `src/fuzzer_tool/cli/commands.py` | `mutants build|score` subparsers |
| `tools/build_targets.sh` | `--mutants`: build `$FUZZ_BUILD_ROOT/mutants/fuzzgoat_nv/<key>/` via the CLI (Rule 1: built artefacts come from this script) |
| `tools/lib/bench_paired.py` | `--kill-set`, `run_cell` hook, `_PLAIN_METRICS` |
| `docs/DEEP_DIVE.md` | §Mutant-kill metric |
| `docs/architecture.dot` → `docs/images/architecture.svg` | new subsystem (Rule 44) |
| `docs/AB_LEDGER.md` | A/A row, then A/B row(s) (Rule 59) |
| `docs/TODO.md` | tracking entry |

Layering: cli → services → core/adapters; bench calls the service, never the
driver (Rule 36b).

## Tests (TDD: written first, seen failing)

| File | Test | Kind |
|---|---|---|
| `tests/test_ir_mutants.py` | sites per op on a fixture `.ll`; expected counts derived by independent line scan | behaviour |
| | `apply_site` changes exactly the one line, to the expected predicate | falsification |
| | `icmp` in string constant/metadata, `br label`, `llvm.dbg`/`lifetime`, argument spills, CRLF, empty module → no sites | adversarial |
| | `sample_sites` with scripted RNG (`tests/support/scripted_rng.py`) → exact subset | Rule 39 |
| `tests/test_mutant_build.py` (skip without clang ASAN runtime) | toy `n<4`→`n<=4`: `abcd` kills, `abc` does not | falsification |
| | infinite-loop mutant → HANG, not kill; original-crashing input excluded; empty record; record > cap | adversarial |
| | mutant identical after `-O2` → dropped by TCE | behaviour |
| `tests/test_mutant_kill.py` | same corpus scored twice → identical `killed` | Rule 46 control |
| | early stop: fake adapter counts replays per killed mutant | behaviour |
| `tests/test_bench_paired_kills.py` | `_metric_arg("kills")`; `compare(rows, rows, "kills")` → p = 1 (control) before a shifted-rows case | Rule 46 |

All tests write under `tmp_path` and clean up (Rule 21). `lizard --CCN=15` on
each new `.py` (Rule 45).

## Risks

- Kill saturation on a small parser (1k LOC): handled by the stop criterion.
- Corpus size confound: more inputs → more kills. Report `inputs` per arm; if
  arms differ in corpus size, also score kills after `fuzzer-tool minimize`.
- Mutant flakiness (ASLR, timing): determinism control catches it; a kill is
  confirmed by one re-run of the killing input.
- Container needs `libclang-rt-18-dev`; `build_targets.sh`
  `check_asan_runtime` already names the package.
