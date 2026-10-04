# Handover: paper_collection survey for binary fuzzing (2026-10-02)

Source: https://github.com/0xricksanchez/paper_collection (README, ~814 papers
across Read & Tagged, General fuzzing, Format-specific, Static binary
analysis, Harnessing, Surveys). Fuzzer baseline: HEAD `0d6c7b38`.

**Scope and method.** Analysis only; no code changed. Coverage was judged by
keyword greps over `src/`, `tools/`, `docs/` plus a skim of a few modules
(`core/state_vars.py` among them). "Not found" means no grep hit, not a proof
of absence. Papers were not read; judgement is from titles and tags in the
collection.

## 1. Already covered

| Paper family | Where in the fuzzer |
|---|---|
| REDQUEEN / CmpLog / I2S, IMPROVING CMPLOG | `core/cmplog.py`, `core/rq_encodings.py`, `services/position_arena.py` |
| WEIZZ chunk tags | `core/weizz_tags.py`, `core/wfc_chunks.py` |
| MOpt, DARWIN, SLOPT, AMSFuzz | mutator scheduling in `services/seed_picker.py`, `core/slopt.py` |
| T-Scheduler, LinUCB, bandit/UCB, AutoFuzz-style scheduling | `core/schedulers/`, `services/seed_picker.py` |
| SHAPFUZZ | `core/shapley.py` |
| Angora / gradient search | `core/gradient_cmp.py`, `core/gradient_descent.py` |
| SGFuzz-style enum state (covers the role IJON targets) | `core/state_vars.py`, `__sfuzz_state` in `afl_shim.c` |
| Dominators / superblocks (Agrawal 1994, bcov) | `core/dominators.py`, `core/mincut.py` |
| Intel PT / ptrace coverage (PTfuzz, kAFL line) | `core/intel_pt.py`, `services/ptrace_coverage.py` |
| Coverage saturation (Reachable Coverage) | saturation estimate in `services/seed_picker.py` |
| Corpus distillation, Diar-style minimization | `services/minimize.py`, `core/minimal_seeds.py` |
| AFLGo-style distance / directed (partial) | ICFG distance channel, `core/icfg.py` |
| FormatFuzzer-style format knowledge | `core/format_fsm.py`, `core/field_map.py` |

## 2. Gaps, ranked by fit (stdlib-only, target-agnostic fuzzer)

1. **Hot-byte identification** (Finch, BaSFuzz, ProFuzzer type probing, PosFuzz).
   No byte-importance map found. Restrict mutation to bytes that change
   coverage; natural extension of `services/position_arena.py`. Cheap, measurable.
2. **TaintScope-style checksum repair.** Only format-specific CRC handling
   (PNG, gzip) and int-checksum solvers found. A generic "find the check,
   re-patch after mutation" step is missing.
   *Status 2026-10-04: implemented as `--checksum-sites` (`core/checksum_sites.py`); correction: `ChecksumLearner`/`crc_learn` already recovered unknown models but patched only a trailing field.*
3. **Binary-only rewriting** (StochFuzz, E9AFL, "Same Coverage, Less Bloat",
   Breaking Through Binaries). No stripped-binary instrumentation beyond
   ptrace/PT. Largest capability gap for binary fuzzing and the costliest.
4. **Resource-guided feedback** (MemLock, HotFuzz, PerfFuzz). Peak RSS exists
   only as a stat (`services/stats.py`, `services/report.py`), not as a score.
5. **Structure synthesis without grammar** (Grimoire). Only mentioned in
   `docs/port-backlog.md`. Superion, Nautilus, Gramatron show passing
   mentions only; grammar-aware fuzzing is thin beyond format mutators.
6. **Evaluation rigor** (Evaluating Fuzz Testing, How to Compare Fuzzers,
   Magma, FixReverter). No A12 / Mann-Whitney or ground-truth bug harness
   found, which limits validating scheduler changes.

## 3. Low priority / out of scope

Directed fuzzing variants (Titan, Beacon, Hawkeye), hybrid concolic
(QSYM, SymCC, SymQEMU), kernel/firmware/IoT/hypervisor, smart-contract,
and LLM sections: mostly mismatched with a stdlib-only, target-agnostic fuzzer.
Static-binary-analysis papers (BinDiff family, angr, BAP) are reference only.

## 4. Suggested order

1. Hot-byte map (item 1), then resource feedback (item 4): both small.
2. Evaluation harness (item 6) before tuning claims.
3. Checksum repair (item 2).
4. Grimoire (item 5) and binary rewriting (item 3) as larger projects.

## 5. Open / unverified

- Grep-based; items 1, 2 and 4 should be re-checked by reading the relevant
  modules before implementing, to avoid duplicating existing partial work.
- No papers were read in full; fidelity to each paper is not assessed.
- No decision made on which item to implement.
