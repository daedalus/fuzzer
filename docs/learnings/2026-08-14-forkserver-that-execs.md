# A forkserver that execs is not a forkserver

**Date:** 2026-08-14
**Closes:** `docs/edge-coverage-analysis.md` §1
**Commits:** `feat(shim): add a real AFL-style forkserver to afl_shim.c` and the three
that follow it.

## The trap

`adapters/forkserver.py` and `adapters/fuzz_loader.c` had existed for a long time,
were named forkserver, were described as a forkserver in their own docstrings, and
were held up by the analysis doc as *the highest-leverage change in the tree*, worth
2–10x. The doc's prescription was a deletion: `fuzz_loader.c` round-tripped a coverage
bitmap through a file while the target wrote to SHM, so drop the file path, drop the
`memmove` on the Python side, let the child inherit `__AFL_SHM_ID`, done.

All of that was correct. It was also worth nothing:

| target | posix_spawn | after the prescribed deletion | speedup |
|--------|-------------|-------------------------------|---------|
| `test_target` | 612 exec/s | 648 exec/s | 1.06x |
| ASAN + heavy static init | 93.8 exec/s | 92.6 exec/s | **0.99x** |

`run_executable()` did `fork()` **followed by `execl()`**, once per input. The entire
cost a forkserver exists to remove — ELF load, dynamic linker, libc init, ASAN runtime
init, the target's own constructors — was still paid on every single execution. The
Python-side saving (a pipe write instead of a `posix_spawn`) is noise next to that.

Two details made this easy to miss for as long as it survived:

- The name. Every reference to it in the tree, including its own comments, asserted it
  was a forkserver. Nothing said "fork, then exec".
- The doc's confidence. §1 was specific, well-argued, and correct about the bitmap
  bug, which made it read as though it had been traced end to end. It had not: the
  container had no clang, so nothing in that document had ever been run.

## What actually fixes it

The server has to live inside the target, because the exec *is* the cost. AFL puts it
there; this tree's shim had no forkserver at all — no `__AFL_INIT`, no FORKSRV
handshake, nothing. (`__AFL_INIT()` / `__AFL_LOOP()` in `png_read.c` and
`grep_read.c` are for real AFL++ builds and never ran under this shim.)

`__afl_start_forkserver()` now runs at the end of the shim's constructor, speaking
AFL's protocol on AFL's fd numbers (198/199). The target execs once; each input is a
fork from a process already past initialisation.

| target | posix_spawn | true forkserver | speedup |
|--------|-------------|-----------------|---------|
| `test_target` | 623 exec/s | 3281 exec/s | **5.27x** |
| ASAN + heavy static init | 90.6 exec/s | 125 exec/s | **1.38x** |
| full CLI, end to end | 484 eps | 1341 eps | **2.77x** |

The ASAN gap is the number to remember: forking a process carrying ASAN's shadow
mapping is expensive enough to eat most of the win, and every target in this tree is
built with ASAN by default (hard rule 9). 5.27x is the ceiling, 1.38x is closer to
what the vendored targets will see.

## The subtle bug the forkserver brings with it

The server loop is compiled into the target's own translation unit, so it is
instrumented like everything else. Left alone it:

1. wrote its own edges into the map **after** the fuzzer's per-exec reset, attributing
   the server's control flow to whichever input happened to be running; and
2. advanced `__afl_prev_loc` between forks, so children did not start from a common
   coverage state.

Symptom: `b"hello"` produced 5 edges, then 6, then a stable 6. Any novelty decision
taken during those first executions was made on noise.

The fix reuses what is already there rather than adding a hot-path branch: detach
`__afl_area` in the server parent and restore it in the child. Suppression then goes
through `if (!__afl_area) return;`, which is already the first line of
`__afl_map_edge`, so the per-edge cost is unchanged — and that check returns *before*
`prev_loc` is read or written, which fixes (2) as a side effect. Now 1 distinct edge
set over 8 runs.

## Semantics that changed on purpose

Coverage from pre-fork initialisation is recorded once, in the server parent, and then
cleared by the next reset. Children never re-record it. Before shipping that, it was
worth checking rather than assuming:

- the forkserver edge set is a strict **subset** of the spawn edge set, for every
  input tested;
- the dropped difference is **identical across all inputs** — constant init edges,
  carrying no signal;
- input discrimination is unchanged (3 distinct sets from 5 inputs, both modes);
- return codes match exactly, and end-to-end crash counts agree (38 vs 37, same unique
  signature).

Consequence to remember: **edge sets are not comparable across `--no-forkserver`**, so
a resumed session that switches modes will see its whole known edge set shift.

## Generalisable

- A name is not an implementation. `grep` for what a component *does* on its hot path
  before believing what it is called — the four lines of `run_executable()` that
  mattered were visible the whole time.
- Measure before the diff, not after. Benchmarking the doc's prescribed fix *first*
  cost one script and turned a finished-looking change into a correct one. Had it
  shipped on the strength of the doc's reasoning, the tree would have carried a
  "2–10x forkserver" that was worth 0.99x, and the next person would have had no
  reason to look.
- When a document says it has never been run, treat its conclusions as hypotheses even
  where its analysis is sound. §1 was right about the bug and wrong about the fix, and
  those are independent.
- Instrumented code that runs *between* executions is a coverage-attribution bug
  waiting to happen. Anything the fuzzer adds inside the target needs to ask whether
  it is being recorded.
