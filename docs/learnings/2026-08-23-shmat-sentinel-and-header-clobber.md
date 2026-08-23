# The half-fix that passes its own regression test

Date: 2026-08-23

Added: `tests/test_regression_shmat_sentinel_guard.py`.
Changed: `src/fuzzer_tool/adapters/inprocess.py`,
`tests/test_regression_shmat_restype.py`.

Two bugs in `adapters/inprocess.py`, found by taking the "cross-cutting
patterns" list at the end of `docs/bugreport_2026-08-21_merged.md` at its word
and checking each entry against the code.

## 1. The sentinel guard that could never fire

The ctypes-hygiene entry recorded a sibling hazard as explicitly unfixed:

> the `(void *) -1` failure sentinel does NOT compare equal to `-1` once
> `restype=c_void_p` is declared, so adding the restype without rewriting the
> `== -1` guard silences the failure path.

`inprocess.py` was in exactly that state, at all three of its attach sites. It
declared `libc.shmat.restype = ctypes.c_void_p` — so it **passed**
`test_no_module_calls_shmat_without_a_declared_restype`, the scan written to
catch this class — and then guarded the result with `if ptr and ptr != -1`.

A failed attach arrives as `0xffffffffffffffff`. Truthy. Not `-1`. The guard
admitted every failure:

| site | what a failed attach did |
|---|---|
| `read_bitmap()` | built a `c_uint8` array over the sentinel and returned it as coverage |
| `reset_bitmap()` | `memset(0xffffffffffffffff, 0, shm_size)` — a **write** through an unmapped address |
| loader script template | same, in the child |

The `memset` is the one that matters. It raises SIGSEGV in the *fuzzer*
process, which the enclosing `except Exception` cannot catch: running the
pre-fix code under pytest kills the interpreter rather than failing a test.

Worse, the sentinel was then cached in `self._shm_ptr`, whose refresh condition
was `is None`. One transient failure pinned the runner to the dead pointer for
the remainder of the campaign.

**The lesson is about the test, not the code.** A scan that checks for the
declared `restype` certifies the half of the fix that is easy to see. The
half that was actually load-bearing — the comparison — went unchecked, and the
green scan is what made the file look done. `test_regression_shmat_restype.py`
now also scans for `ptr == -1` / `ptr != -1` in any module that calls `shmat`.
`adapters/shm.py` is the form to copy: it compares against
`ctypes.c_void_p(-1).value`, never against `-1`.

Both sites now route through `adapters/libc_shm`, which returns `None` on
failure so the call-site check is a plain falsiness test that cannot be got
wrong. The loader-script template cannot import the package (it runs
standalone in the child), so it spells the sentinel out — with a test asserting
the inlined copy still agrees with `libc_shm.SHMAT_FAILED`.

## 2. Two writers of the SHM front header

Fixing the guard meant reading `reset_bitmap()` closely, which turned up a
second, larger bug. It did:

```python
ctypes.memset(self._shm_ptr, 0, self.shm_size)
```

justified by a comment: *"safe because the C shim's `__afl_map_reset()`
rewrites the header after the target executes"*.

`__afl_map_reset()` has **zero callers**. The shim's own comment at
`afl_shim.c:848` says so. Nothing rewrote the header, so this zeroed all three
fields packed into the diag word at offset 4, on every execution:

- **generation** (bits 24-31). `__afl_map_edge` reads the tag straight out of
  the diag word (`afl_shim.c:567`) and stamps it into `count`; the reader
  filters entries by it. Zeroing it pins the generation at 0 on both sides, so
  entries left over from earlier executions carry the same tag as live ones and
  read as current coverage. `reset_edge_map()` bumps the generation immediately
  before `run_one()`, and this memset undid it a moment later — the generation
  protocol, defeated by its second writer.
- **dropped-edge count** (bits 8-23), which `adapters/shm.py` describes as the
  only honest occupancy signal available. Pinned at 0, a saturated map reads as
  healthy — the exact failure that comment warns about.
- **ctx width** (bits 0-7), written once at attach (`afl_shim.c:441`).

The length was wrong too, and in the direction that hid the problem:
`shm_size` is the entry COUNT (`AFL_MAP_SIZE` convention), so the call zeroed
`shm_size` bytes of a `24 + shm_size * 8` byte segment — one eighth of the
table. The regression test catches this at index 488 of a 512-entry table:
512 bytes memset, 24 of them spent on the header.

`reset_bitmap()` now zeroes the edge table only, from
`SHM_METADATA_SIZE` for `shm_size * SIZEOF_ENTRY` bytes, importing both
constants from `adapters/shm` rather than restating them.

### Left open deliberately

On the `services/runner.py` path this reset is *redundant*:
`reset_edge_map()` already bumped the generation immediately before
`run_one()`, and `run_one` has exactly one call site. Dropping the call from
`_run_c_direct` would save a full-table memset per execution. That is a
throughput change on the hot path of the fastest execution mode, so it wants
measuring on a built target rather than arguing about — recorded in
`docs/TODO.md` instead of guessed at here.

## Measured, once clang was available

Installing clang and building instrumented targets turned this from argument
into measurement, and corrected part of the argument.

**The header clobber is real and total.** Driving `png_read.so` through
`InProcessRunner` the way `services/runner.py` does, four executions in
sequence:

| | generation at read | diag word |
|---|---|---|
| pre-fix | 0, 0, 0, 0 | `0x0` every time |
| fixed | 1, 2, 3, 4 | `0x1000008` .. `0x4000008` |

The `008` in the low byte is the ctx width the shim wrote once at attach. It
survives now; before, every execution destroyed it along with the generation
and the drop counter.

**The coverage leak it enables is real but needs a big target, which is not
what I claimed.** The prediction was that stale entries would read as live
coverage. Running the sequence longest-input-first, expecting the live set to
stop shrinking, produced identical numbers before and after the fix --
`[40, 41, 12, 12]` both ways. The reason is worth writing down: entry index IS
the edge id, assigned sequentially per module by
`__sanitizer_cov_trace_pc_guard_init`. `png_read.so` has roughly 4,600 guards,
and the old memset reached index 8,189, so every edge the target can produce
fell inside the region that got zeroed anyway. The bug was masked by the
target being small.

Planting one entry at index 20,000 -- what a target with more than 8,189 edges
leaves behind after an earlier execution -- shows the mechanism cleanly:

```
                      pre-fix   fixed
generation at read          0       1
stale edge reported      True   False
```

So the severity is conditional on edge count, not universal. Any target with
more than `(map_size - 24) / 8` live edges carries coverage forward from
previous executions. At the default 65,536-entry map that threshold is 8,189
edges -- comfortably reached by ffmpeg or a vendored grep, not by the small
targets in `targets/`. The drop counter and ctx width, by contrast, were
destroyed unconditionally on every single execution regardless of target size.

Worth being precise about which half of a prediction the evidence supports.
The mechanism was right; the blast radius was overstated, and only building
the thing showed which.

## The pattern worth carrying forward

Both bugs are the same shape as the two already recorded in this tree
(`test_hex_escape`, `test_different_stderr`): a comment that explains away a
surprising value, sitting on top of a defect.

- `ptr != -1` reads as a correct failure check and is not one.
- *"safe because `__afl_map_reset()` rewrites the header"* reads as a
  justification and cites a function nothing calls.

The check that would have caught either is the same, and it is cheap: when a
comment asserts that something else makes this safe, go and confirm that the
something else happens. `grep` for the caller. Both of these took one command.
