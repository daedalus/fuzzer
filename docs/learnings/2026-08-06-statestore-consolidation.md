# StateStore consolidation: replacing 11 JSON files with one pickle

**Date:** 2026-08-06
**Context:** fuzzer-tool `core/state_store.py` wiring — replacing 11 per-component JSON state files with a single compressed pickle

## Problem
The fuzzer persisted state across 11 separate JSON files in the corpus directory (state.json, edge_tracker.json, markov.json, mi.json, elo.json, ga.json, qea.json, sensitivity.json, crash_mi.json, length_tracker.json, seed_quality.json). Each had its own path variable, save/load logic, and key-coercion bugs (JSON turns int keys into str, requiring defensive `int(k)` reconversion in 8+ load paths). The untracked `core/state_store.py` was written but never wired in.

## Rejected
- **Modifying individual JSON files in place** — looked plausible as a minimal change; dropped because the int-key-coercion problem is systemic and only pickle (not JSON) fixes it at the serialization layer
- **Replacing component save/load APIs entirely** — looked plausible as the "clean" approach; dropped because corpus_manager.py, tmin.py, and tests call the file-path methods directly, so removing them would break the public API

## Approach
Two-layer integration: (1) Component `to_dict()`/`from_dict()` added to file-path-based components (markov, elo, edge_tracker, mi, ga, qea) — legacy `save(path)`/`load(path)` delegate to them, preserving backward compatibility; dict-based components (sensitivity, crash_mi, length_tracker, seed_quality) were already compatible. (2) StateStore intercepts at the Fuzzer/CorpusManager layer: StateStore is initialized in `Fuzzer.__init__`, loaded once when `--resume`, and `CorpusManager.save_state`/`load_state` + `cmd_fuzz` + `report.py` + `tmin.py` use `StateStore.get(section)`/`set(section, dict)` instead of individual JSON files. On first save, `cleanup_legacy()` removes the old JSON files.

## Key insight
`StateStore.set()` must set `self._loaded = True` so that a subsequent `get()` call doesn't trigger an auto-load from disk that would silently overwrite in-memory state set by `set()`. The `get()` method auto-loads on first access (for the normal `--resume` path), so without this guard, `set()` followed by `get()` would clobber the just-set data.

## Verification
15 regression tests added in `tests/test_regression_state_store.py` covering round-trip, int-key preservation, legacy JSON migration, cleanup, disabled mode, and unsafe-global rejection. Full suite: 3232 passed, 17 pre-existing failures (all `args.ptrace` AttributeError from commit a0546fb, unrelated to this change).

## Generalizes to
When introducing a caching layer with auto-load-on-access semantics, any write method (`set`) must also mark the cache as loaded to prevent the auto-load from clobbering in-memory writes. This pattern applies to any lazy-cache design where read paths trigger initialization.
