---
name: crash-analysis
description: >
  Systematically analyze, classify, and attribute crash artifacts from coverage-guided
  fuzzing campaigns. Use this skill whenever the user asks to analyze a crash,
  understand a crash report, reproduce a segfault/SIGFPE/SIGBUS, build a
  regression test for a crash, generate a FINDINGS.md entry, or identify the root
  cause of a signal or sanitizer error in a fuzzer target. Triggers on phrases
  like: "analyze this crash", "why did it segfault", "crash report", "SIGSEGV",
  "ASAN heap-buffer-overflow", "heap-use-after-free", "crash forensics",
  "root cause this crash", "find the mutation site", "build a regression test
  for the crash", "generate a findings doc for this bug".
---

# Crash Analysis

Reproduce, classify, and attribute crash artifacts from fuzzing campaigns using
the project's sidecar format and existing analysis tooling.

---

## When to Use This Skill

- "analyze this crash"
- "why did it segfault"
- "build a regression test for this crash"
- "generate a FINDINGS.md entry for this bug"
- "root cause this SIGFPE"
- "find the mutation site"
- "ASAN heap-buffer-overflow" / "UAF" / "UAF in libfoo"
- "crash forensics"
- "crash report" / "crash artifact"
- "minimize this crash input"

---

## Crash Artifact Format

The fuzzer writes a quintet of sidecar files per crash (via ``save_crash`` in
``adapters/filesystem.py``):

```
crashes/
  crash_<ts>_<cluster_id>_<sig>.bin   ← raw input bytes (only file counted as "crash")
  crash_<ts>_<cluster_id>_<sig>.txt   ← human-readable report (target, returncode, GDB replay)
  crash_<ts>_<cluster_id>_<sig>.json  ← same fields, machine-readable (CrashMetadata.to_dict())
  crash_<ts>_<cluster_id>_<sig>.hex   ← xxd-style hexdump + text repr
  crash_<ts>_<cluster_id>_<sig>.sh    ← base64 reproducer script
```

The ``.bin`` is the authoritative input; the ``.txt`` and ``.json`` carry the
same triage fields (sanitizer, error_type, fault_addr, frames, registers,
gdb_replay, nearest_corpus, raw_stderr) — ``.txt`` for humans, ``.json`` for
dashboards and downstream tooling. The two are written together by
``save_crash``; treat them as a single artifact.

---

## Core Methodology

### 1. Discover

Glob ``crashes/*.bin`` to enumerate inputs. Each ``.bin`` already has a
companion ``.json`` sidecar with the parsed triage fields — read those
directly:

```bash
jq '.[] | {signal, sanitizer, error_type, fault_addr}' crashes/*.json
```

The ``.json`` is written by ``save_crash`` and mirrors ``CrashMetadata.to_dict()``
(sanitizer, error_type, fault_addr, frames, registers, gdb_replay,
nearest_corpus, raw_stderr, target_sha256, returncode). No re-parse of the
``.txt`` is needed.

### 2. Classify

Every crash falls into exactly one bucket:

| Bucket | Source | Key field |
|--------|--------|-----------|
| ASAN | ``SanitizerReport.parse(stderr)`` | ``error_type`` |
| UBSAN | same | ``error_type`` |
| MSAN/TSAN/LSAN | same | ``sanitizer`` |
| Signal (no sanitizer) | ``SIGNAL_CRASH_CODES`` | ``returncode`` |
| 0-exit with diag | ``is_interesting()`` fallback | stderr scan |

**ASAN/UBSAN exploitability** is pre-computed by ``SanitizerReport`` using
the project's exploitability tables (``core/sanitizer.py``). Signal crashes
(default to ``UNKNOWN`` exploitability) require GDB replay for localization.

### 3. Reproduce

**For signal crashes (SIGSEGV/SIGFPE/SIGBUS/SIGABRT)**, the ``.sh`` reproducer
script is the fastest path. For ``.so`` targets, prefer the standalone PIE exe
when available (avoids ASAN/shim LD_PRELOAD conflicts):

```bash
printf '...' | gdb --batch -ex "file targets/ffmpeg_read" -ex "run" -ex "bt" -ex "quit"
```

**For ASAN crashes**, the ``.sh`` script with ``ASAN_OPTIONS=abort_on_error=1`` is
correct. If the ``.so`` build has an ASAN/shim conflict, build and use the
standalone exe instead.

**For ``.so`` targets with ``__afl_guarded_call``**, use the Python harness
(``adapters/inprocess.py:96-108``). The guarded call is the ONLY correct way to
invoke ``fuzz_shm_run`` from Python — bare ``ctypes.CDLL`` lets the signal
terminate Python.

### 4. Localize

Three tools, in order of what they answer:

| Tool | Answers |
|------|---------|
| ``fuzzer-tool replay <crash.bin>`` | "does it still crash?" + backtrace |
| ``fuzzer-tool root-cause <target> <crash.bin>`` | minimal byte diff + mutation site |
| ``fuzzer-tool tmin <target> <crash.bin> --lineage`` | smallest crashing input + parent chain |

For the VPK SIGFPE case: GDB backtrace at ``vpk.c:89`` points at a
divide-by-zero in ``vpk_read_packet``. Trace ``nb_channels`` upstream to find
the mutation: ``avformat_find_stream_info`` at ``demux.c:3066`` calls
``avcodec_parameters_from_context(codec, par)``, copying the zeroed decoder's
ch_layout back over the demuxer's ``par->ch_layout``.

### 5. Document

Write ``docs/FINDINGS/<bug-id>.md`` from ``docs/FINDINGS/TEMPLATE.md``.
Required sections: Target, Configuration, Bug (with Trigger Chain, Crash Input,
Crash Evidence, Exploitability), Suggested Fix, Regression Test.

---

## Sanitizer Classification Reference

```
SIGABRT  → exit 134   (SIGABRT(6))
SIGBUS   → exit 135   (SIGBUS(7))
SIGFPE   → exit 136   (SIGFPE(8))
SIGSEGV  → exit 139   (SIGSEGV(11))
SIGILL   → exit 132   (SIGILL(4))
SIGTRAP  → exit 133   (SIGTRAP(5))
```

ASAN/UBSAN typically exits 1 with a diagnostic line like
``==ERROR: AddressSanitizer: heap-buffer-overflow``.

---

## VPK SIGFPE Case Study

Input: 21 bytes ``20 4B 50 56 56 50 00 F8 04 00 3B 03 61 39 56 32 36 36 30 38 50``

- Magic ``20 4B 50 56`` = LE u32 of ``VPK `` (MKBETAG form)
- ``vpk_read_header`` sets ``nb_channels=80`` (rl32 at offset 20 reads 1 real byte + 3 EOF-zero = 0x50)
- ``avformat_find_stream_info`` then calls ``avcodec_parameters_from_context`` at ``demux.c:3066``,
  copying the freshly-allocated (zeroed) decoder's ch_layout back over the demuxer's par
- ``vpk_read_packet`` divides by the now-zero ``nb_channels`` → SIGFPE

Fix: add a defensive guard in ``vpk_read_packet`` rejecting ``nb_channels <= 0``
before the division. The header check is bypassed by ``find_stream_info``.

---

> Mistake: running ``rm -rf crashes/*`` between analysis sessions.
> This destroys coverage history and forces rediscovery from scratch.
> Always use ``--resume`` to continue.

> Mistake: using ``try/except pass`` when a sidecar is missing.
> Always surface the error explicitly with ``print("[!] ...", file=sys.stderr)``
> and return a nonzero exit code.

> Mistake: testing ``gdb`` without ``--batch`` in automation.
> Interactive gdb hangs indefinitely. Always pass ``--batch``.

> Mistake: parsing the ``.txt`` sidecar from a downstream tool.
> Read the companion ``.json`` sidecar instead — same fields, machine-readable,
> written by ``save_crash`` alongside the ``.txt``. The ``.txt`` is for humans.
