# Action plan: OneFuzz port (repro gate, stable stack buckets)

Source: microsoft/onefuzz agent (`check_retry_count`, `no_repro` container,
`minimized_stack_function_names_sha256`, stack-minimizer ignore list).
Survey and ranking: `docs/TODO.md` §OneFuzz port candidates.

```
 crash ──► classify_crash ──► novel? ──no──► dedup counters
                │                 │yes
       StackFilter (P2)           ▼
       sig + func_hash      repro gate (P1) ──fail──► crashes/no_repro/
                                  │pass
                                  ▼
                           save_crash ──► crashes/
```

## P1 — Repro gate (`--repro-gate N`)

**Gap.** `--replay-n` scores crashes after they are saved
(`services/stats_reporter.py:159`). Crashes that never reproduce sit in
`crashes/` and alert, just like real ones.

**Behavior.** Re-run each *novel* crash up to N times. The crash passes as
soon as one replay produces the same signature, matching OneFuzz retry
semantics. If no replay matches, write it to `crashes/no_repro/` under its own
signature map, so a later crash with that signature that does reproduce still
reaches `crashes/`. A crash in `no_repro/` sends no alert. N=0 (the default)
turns the gate off.

**Cost.** At most N execs, and only for novel crashes. Repeats of known
signatures exit in `classify_crash` before the gate runs, so no per-exec
code changes.

Sketch (`services/corpus_manager.py`, `CorpusManager`):

```python
class ReproGate(Enum):
    OFF = 0
    ON = 1

def _bind_repro(self) -> None:
    """Ifs up: choose the gate once at setup."""
    n = self.f.repro_gate
    self._reproduces = self._repro_check if n > 0 else _always_true

def _repro_check(self, data: bytes, sig: str) -> bool:
    """True when any of N replays yields *sig* (OneFuzz check_retry_count)."""
    run = self.f._runner.run_target
    for _ in range(self.f.repro_gate):
        rc, stderr = run(data)
        if _signature_of(stderr, rc) == sig:
            return True
    return False
```

In `save_crash`, after `verdict.novel` and before enrichment:

```python
if verdict.novel and not self._reproduces(data, verdict.signature):
    return self._save_no_repro(data, returncode, stderr, verdict)
```

`_save_no_repro` calls `save_crash(...)` with `crashes_dir / NO_REPRO_DIR`,
`f.no_repro_hashes` and `f.no_repro_sigs`. Both are new sets kept with the
existing pickle state machinery (Hard Rule 15).

### Wiring

| Where | Change |
|---|---|
| `cli/commands.py` fuzz parser | `--repro-gate N` (int, default 0), next to `--replay-n` |
| `services/fuzzer.py` init | `self.repro_gate`, `no_repro_hashes`, `no_repro_sigs`; persist and restore beside `crash_hashes`/`crash_sigs` |
| `services/corpus_manager.py` | `_bind_repro()` at init; gate in `save_crash`; `_save_no_repro` |
| `adapters/filesystem.py` | `NO_REPRO_DIR = "no_repro"` constant |
| `services/stats_reporter.py` | final summary: `no_repro: K` |
| `docs/DEEP_DIVE.md` | §Crash triage entry |

### Tests (`tests/test_repro_gate.py`)

- Falsification: a fake runner that is clean on replay sends the crash to
  `no_repro/`, writes nothing to `crashes/` and sends no alert.
- Adversarial: a replay crashes with a *different* signature. The gate must
  fail (the signature is compared, not the return code).
- Edge cases:
  - Runner scripted [clean, clean, match] with N=3: the crash passes. With
    N=2 it fails.
  - N=0: the runner is never called.
  - A signature seen first in `no_repro` and later reproducing lands in
    `crashes/`.
  - `--resume` restores the `no_repro` maps.
- Speed: fuzzgoat run with the gate on vs off. Execs/s must match within
  noise (Hard Rule 41).

## P2 — Configurable stack filter and function hash (shipped; `--resume` warning deferred to TODO)

**Gap.** Frame stripping uses hardcoded sets (`core/sanitizer.py:152-166`).
Project wrappers such as `av_free`, `xmalloc` and `g_free` become the
signature. Measured with a prototype on a real-shaped stack
`[free, av_free, av_freep, decode_frame, …]`:

| Stripping | Signature frames |
|---|---|
| Built-in sets only | `[av_free, av_freep, decode_frame]` |
| Plus user regex `^av_free` | `[decode_frame, avcodec_send_packet]` |

In addition, `stack_hash()` hashes PC page offsets. Any rebuild changes
those offsets, so `--crash-blocklist` silently stops matching across builds.

**Behavior.**

1. `--stack-ignore FILE`: one regex per line, `#` starts a comment. Matching
   frames are stripped in addition to the built-in sets.
2. `func_hash()`: xxh64 of the stripped function names, joined with NUL and
   cut at the harness. It survives rebuilds and ASLR. It is written to the
   `.json`/`.txt` sidecars, and `--crash-blocklist` also accepts lists tagged
   `# func-hash v1`.

**Cost.** Prototype: `signature_frames` takes 0.58 µs with no file and
1.62 µs with a 2-pattern file, once per crash classification. With no file,
the built-in predicate stays bound, so there is no cost.

Sketch (`core/sanitizer.py`):

```python
FUNC_HASH_VERSION = 1
FUNC_HASH_TAG = f"# func-hash v{FUNC_HASH_VERSION}"

def configure_stack_ignore(patterns: list[str]) -> None:
    """Bind the runtime-frame predicate once (ifs up). [] restores built-ins."""
    global _is_runtime
    if not patterns:
        _is_runtime = _is_builtin_runtime
        return
    match = re.compile("|".join(f"(?:{p})" for p in patterns)).match
    _is_runtime = lambda f: _is_builtin_runtime(f) or match(f) is not None

def func_hash(self) -> str:
    """Rebuild-stable bucket: stripped function names, harness-cut."""
    names = _target_frames(self.frames)
    if not names:
        return ""
    return f"{xxh64(NUL.join(names).encode()):016x}"
```

The module-level binding keeps all 18 `SanitizerReport.parse` call sites
unchanged. The alternative, threading a filter object through every caller,
touches far more lines (Hard Rule 36a). Tests reset the binding in a fixture
(Hard Rule 21).

**Compatibility.** The signature format is unchanged when no file is given.
With a file, signatures differ from earlier runs, so `--resume` warns once
when the hash of the ignore file stored in state differs from the current one.

### Wiring

| Where | Change |
|---|---|
| `core/sanitizer.py` | rename `_is_runtime` → `_is_builtin_runtime`; `configure_stack_ignore`; `_target_frames`; `SanitizerReport.func_hash`; `FUNC_HASH_TAG` |
| `cli/commands.py` | `--stack-ignore FILE` → parse → `configure_stack_ignore` once before the fuzzer starts; the blocklist loader accepts both tags |
| `adapters/filesystem.py` | `CrashVerdict.func_hash`; blocklist check matches `stack_h` or `func_h` |
| `core/crash_metadata.py` | `func_hash` field in `to_dict` and the `.txt` lines |
| `services/fuzzer.py` | store the ignore-file hash in state; warn on mismatch at `--resume` |
| `dictionaries/` neighbor | `stack_ignore/ffmpeg.txt` sample (`^av_(free|freep|malloc|mallocz|realloc)$`) |
| `docs/DEEP_DIVE.md` | §Crash triage entry |

### Tests (`tests/test_stack_filter.py`)

- Falsification: with no file, signature and `func_hash` match the current
  code's output on a recorded ASAN report.
- Adversarial: the pattern `.*` must not empty the signature. The innermost
  frame is kept, as `signature_frames` does today. An invalid regex fails at
  CLI parse with the line number.
- Edge cases:
  - Same functions with different PCs give the same `func_hash` (simulated
    rebuild).
  - `["ab","c"]` vs `["a","bc"]` give different hashes (NUL separator).
  - Empty stack (UBSAN) gives `""`.
  - A blocklist tagged `func-hash v1` suppresses a crash after a rebuild.
  - The fixture restores the built-in predicate.

## Order

1. P2 first: P1's gate compares signatures, so the signatures must be final
   first.
2. Write the tests first (Hard Rule 38), then the code.
3. Run the affected tests, then `ruff`.
4. Update `docs/DEEP_DIVE.md`, then commit and push.
