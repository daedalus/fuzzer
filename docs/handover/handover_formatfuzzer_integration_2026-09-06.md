# Handover — FormatFuzzer Integration

Original 2026-09-06 (plan). Pruned 2026-09-26 to open items; full original:
`git show 1c689e8a^:docs/handover/handover_formatfuzzer_integration_2026-09-06.md`.
Verified against `4c021daa`. Phase 1 live: `core/mutations/formatfuzzer.py`
(`ff_png`/`ff_zip`/`ff_isobmff`/`ff_jpeg`, upstream `parse`/`fuzz --decisions` CLI),
`--formatfuzzer`/`--ff-bin-dir`/`--ff-templates`, `tests/test_formatfuzzer_mutator.py`.

Paper: Dutra, Gopinath, Zeller, TOSEM 2023 (doi 10.1145/3628157); upstream
`uds-se/FormatFuzzer`. Backlog: `docs/port-backlog.md` A4.

## Open

### 1. Real upstream binaries never exercised
All tests use a fake `png-fuzzer` script (`tests/test_formatfuzzer_mutator.py:ff_bin`).
No `tools/` script builds/installs upstream generators. Add one (e.g.
`tools/vendor_formatfuzzer.sh`, per Hard Rule 1: sources under
`$FUZZ_VENDOR_ROOT/formatfuzzer/`), build `png`/`zip` at minimum, and run one
end-to-end mutate against real binaries. Blocks item 5.

### 2. Phase 2 — decision-seed mode (paper's AFL+FFGen)
Keep a side corpus of decision files per seed; add `ff_seed_havoc`: byte/bit-mutate
the stored decision file, then `<fmt>-fuzzer fuzz --decisions`. Today every
mutation re-`parse`s the input (`FormatFuzzerMutator._run_ff`, two fork+execs).
Register via `REGISTRY` only (Hard Rule 12). Memory must stay bounded (Rule 54).

### 3. Phase 3 — per-template feedback
`FormatFuzzerMutator.on_new_coverage` is a stub returning `None`. Add per
(template, generate|mutate) success counters, bias the generate-vs-mutate split
with them, and expose them in stats output.

### 4. In-process path
Cost is two fork+execs per mutation (`_SUBPROCESS_TIMEOUT = 2.0`). Upstream's
per-format `.so` (`make gif.so`, what AFL++ loads) via ctypes would remove it.
Do after item 5 shows the operator earns its cost.

### 5. Paired A/B (acceptance) — E2 in `handover_pending_2026-09-06.md`
`tools/benchmark.py paired` (`tools/lib/bench_paired.py`) has no formatfuzzer
arm. Arms: baseline vs `--formatfuzzer`, PNG/ZIP/MP4 targets, same protocol as
other paired runs. Pass: higher rare-edge or valid-coverage discovery, or a
unique crash absent from baseline; throughput regression <= 15 % (or Elo
de-prioritises the ops).

### 6. Docs
No FormatFuzzer entry in `CHANGELOG.md`, `README.md`, or `docs/DEEP_DIVE.md`
(Hard Rule 11).
