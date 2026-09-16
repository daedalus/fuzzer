# Plan: hybrid `kruskal_count` seed strategy

> **Status (2026-09-16): implemented.** Module, SeedPicker/Fuzzer/CLI/parallel wiring, state, report, docs. Open: the A/B (`docs/TODO.md`). Deviations: per-seed scores are not persisted (pure function of bytes + profile); "first coupled pair" means earliest-coupling, ties by pair index; the trajectory mask stops at the first revisited offset.
## Goal
Add an opt-in `kruskal_count` seed strategy based on Kruskal-count coupling:
- independent walkers traverse each corpus seed using value-driven jumps;
- seeds are scored by how completely and quickly walker pairs couple;
- high-scoring seeds are selected for fuzzing;
- when walkers converge, their shared trajectory guides recombination with a donor seed to synthesize a new seed;
- format hints influence jumps when available, with a byte-value fallback;
- the strategy is exposed as `--kruskal-count` and participates in Elo seed-strategy arbitration.
This is a low-blast, reversible feature. Rollback is removal of the new module plus the CLI/Fuzzer/SeedPicker wiring and state/report/docs entries.
## Confirmed integration points
- Seed-strategy dispatch and Elo eligibility: `src/fuzzer_tool/services/seed_picker.py:272-346`, `:512-543`.
- Seed-strategy registry used by Elo pre-registration: `src/fuzzer_tool/services/fuzzer.py:136-150`, `src/fuzzer_tool/core/analyzer_registry.py:524-535`.
- Fuzzer construction and persisted state: `src/fuzzer_tool/services/fuzzer.py:1293-1354`, `:1556-1566`, `:1760-1765`, `:1978-1995`, `:7227-7247`, `:7580-7625`.
- CLI parser and `--elo all`: `src/fuzzer_tool/cli/commands.py:543-600`, `:2045-2059`, `:2846-2867`, `:3061-3097`.
- CLI-to-Fuzzer and parallel wiring: `src/fuzzer_tool/cli/commands.py:400-532`, `:602-814`; `src/fuzzer_tool/services/parallel.py:571-744`.
- Target format hints: `src/fuzzer_tool/core/target_profiler.py:104-128`, `:712-810`.
- Report and live stats: `src/fuzzer_tool/services/report.py:1681-1744`, `:1929-1965`; `src/fuzzer_tool/services/stats.py:1022-1033`; startup banner `src/fuzzer_tool/services/fuzzer.py:6529-6619`.
- Existing deterministic seed-strategy tests: `tests/test_seed_picker.py:1-424`; CLI argument coverage tests are referenced at `src/fuzzer_tool/cli/commands.py:761-770`.
- Documentation and architecture conventions: `docs/DEEP_DIVE.md:49-58`, `:102-103`, `:173`, `:275-283`, `:432`, `:487`, `:707`; `docs/TODO.md`; `docs/architecture.dot` and generated `docs/images/architecture.png`/`.svg`.
## Recommended design
### 1. Add one core strategy module
Create `src/fuzzer_tool/core/schedulers/kruskal_count.py` with `KruskalCountSeedStrategy`.
Keep the implementation self-contained and pure enough for focused tests:
```python
class KruskalCountSeedStrategy:
    def score_seed(self, seed: bytes) -> float: ...
    def select(self, seeds: list[bytes]) -> bytes | None: ...
    def generate(self, anchor: bytes, donors: list[bytes]) -> bytes | None: ...
    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, data: dict, rng, profile) -> "KruskalCountSeedStrategy": ...
```
Use `RandPool` for every random draw. Do not call `random`, NumPy directly, or `hashlib` for seed identity; use the fuzzer's existing `_seed_key()` when persisting per-seed scores.
### 2. Define the walker and score
Constants should live at module scope:
- `WALKER_COUNT = 4`
- `MAX_STEPS = 256`
- a small positive selection floor such as `MIN_WEIGHT = 1e-6`
- strategy state version `STATE_VERSION = 1`
Walker initialization:
- use distinct, evenly spaced start positions when the seed is long enough;
- for short seeds, use as many distinct starts as possible;
- never start two walkers at the same position.
Jump rule:
```python
def jump_position(seed, position, profile):
    if profile.format_signature and profile.boundary_markers:
        next_boundary = first marker occurrence strictly after position
        if next_boundary is not None:
            return next_boundary
    return (position + max(1, seed[position])) % len(seed)
```
Rules:
- skip empty markers;
- never return the current position from a zero byte; zero means a one-byte jump;
- if no format boundary is available, use the byte-value wrapped jump;
- keep the fallback deterministic and independent of wall-clock time.
Coupling:
- advance all walkers synchronously;
- record the first step at which each pair occupies the same position;
- once a pair couples, treat it as coupled for the remainder of the walk;
- do not count initial duplicate starts because starts are distinct.
Score:
```text
coupling_rate = coupled_pairs / total_pairs
speed = 1 - mean(coupling_step) / max_steps
score = coupling_rate * speed
```
No-coupling seeds score `0.0`. This makes the score bounded, comparable across seed lengths, and directly aligned with the requested behavior.
### 3. Define hybrid synthesis
`generate(anchor, donors)` should:
1. run the same walker trace used for scoring;
2. select the first coupled walker pair;
3. use the shared trajectory from the coupling step through the bounded walk as a recombination mask;
4. choose a donor from the other corpus seeds, weighted by Kruskal score plus a small exploration floor;
5. copy donor bytes into the anchor at masked positions, using modulo indexing when lengths differ;
6. preserve a detected magic-byte prefix so format signatures are not immediately destroyed;
7. return `None` when there is no coupled trajectory or no donor;
8. if recombination accidentally equals the anchor, make one deterministic trajectory-position change using `RandPool` rather than retrying until different.
This is intentionally a trajectory-guided recombination, not a new grammar or parser. It reuses existing format knowledge without inventing format-specific parsers.
### 4. Wire SeedPicker
In `src/fuzzer_tool/services/seed_picker.py`:
- add `"kruskal_count"` to the Elo eligible pool when `f._kruskal_count is not None` and `f.corpus` is non-empty;
- add the strategy-map handler:
```python
"kruskal_count": lambda: self._pick_kruskal_count_seed()
```
- implement `_pick_kruskal_count_seed()`:
  - return `_format_aware_seed()` when there is no corpus;
  - score the current corpus;
  - select an anchor with `RandPool.weighted_choice()` using `score + MIN_WEIGHT`;
  - call `f._kruskal_count.generate(anchor, f.corpus)`;
  - return the generated seed when available, otherwise return the anchor;
  - keep the return type and fallback behavior consistent with existing seed pickers.
In the non-Elo `pick_seed()` fallback, place explicit Kruskal selection after QEA/GA and before Bayesian/Boltzmann/EcoFuzz. This preserves existing precedence while making the new explicit flag useful without `--elo`.
### 5. Wire Fuzzer construction, state, and diagnostics
In `src/fuzzer_tool/services/fuzzer.py`:
- append `kruskal_count=False` to the constructor signature so positional callers are not shifted;
- store `_use_kruskal_count` and create the strategy after `_profile` and `_rng` exist, before `SeedPicker`;
- load `kruskal_count` state only for resumed runs;
- persist `self._kruskal_count.to_dict()` under the `kruskal_count` state-store section at normal exit;
- add startup status output near the MCTS/AlphaBeta status block;
- include `kruskal-count` in `_selected_schedulers_str()` when enabled.
State payload must contain only safe primitives accepted by `StateStore`: version, scalar counters, and string-keyed numeric mappings. `from_dict()` must reject or ignore malformed/unversioned payloads without executing or trusting arbitrary object types.
### 6. Wire CLI and parallel runs
In `src/fuzzer_tool/cli/commands.py`:
- add `--kruskal-count` near the other seed-strategy flags;
- set `args.kruskal_count = True` under `--elo all`;
- include `"kruskal_count"` in `_HAIL_MARY_FLAGS` so the everything-on mode actually constructs it;
- pass the flag to both single-process `Fuzzer(...)` and `run_parallel(...)`;
- add the corresponding parameter to `run_parallel()` and its worker construction path.
### 7. Add reporting
In `src/fuzzer_tool/services/report.py` and `src/fuzzer_tool/services/stats.py`:
- show Kruskal state in the Fuzzing Strategy report: enabled, scored seeds, coupled pairs, generated candidates, and mean score when available;
- optionally include a compact `kruskal` field in the live stats line, guarded by attribute existence so standalone report consumers remain compatible;
- ensure Elo seed-strategy rankings automatically include `seed_kruskal_count` through the existing registry and convergence code.
### 8. Tests
Add `tests/test_kruskal_count.py` plus focused integration coverage in `tests/test_seed_picker.py` and the existing CLI kwargs regression test.
Required tests:
- byte fallback jump is exact and wraps;
- zero byte jumps by one and cannot self-loop;
- format boundary jump wins over byte fallback;
- missing/empty markers fall back safely;
- empty and one-byte seeds do not crash;
- distinct starts are enforced for short seeds;
- a known coupled trace records the expected coupling step;
- a non-coupled trace scores zero;
- faster coupling scores higher than slower coupling with the same pair count;
- synthesis uses the converged trajectory and donor bytes;
- donor/anchor length mismatch is handled by modulo indexing;
- detected magic prefix is preserved;
- no-donor and no-coupling generation returns `None`;
- `to_dict()`/`from_dict()` round-trip with a fresh strategy and deterministic RNG;
- SeedPicker marks the strategy eligible and dispatches the handler;
- non-Elo fallback dispatches Kruskal when explicitly enabled;
- CLI `--kruskal-count` reaches both single-process and parallel Fuzzer construction;
- malformed persisted state is rejected or ignored without partial unsafe state;
- adversarial constant/all-zero seeds remain deterministic and bounded;
- a self-comparison control runs the same deterministic input twice and asserts identical scores/output before any reference comparison, satisfying the oracle-control rule.
Use scripted/fake RNG or fixed `RandPool` seeds for exact assertions. Do not use retry-until-hit loops or hardcoded collection-length equality checks.
### 9. Documentation and architecture
Update:
- `docs/DEEP_DIVE.md`: feature overview, flag table, algorithm description, state section, and diagnostics;
- `README.md`: high-level feature/quick-start entry for `--kruskal-count`;
- `docs/TODO.md`: add the follow-up A/B validation item and mark implementation wiring complete only after tests pass;
- `docs/architecture.dot`: add the Kruskal strategy box and edges to SeedPicker, Elo, target profile, RandPool, and state store;
- regenerate `docs/images/architecture.png` and `.svg` with the repository's documented `dot` command.
No corpus files or build artifacts should be committed.
## Verification plan
After implementation, run in this order:
1. `pytest tests/test_kruskal_count.py tests/test_seed_picker.py tests/test_regression_cli_fuzzer_kwargs.py`
2. Add any parallel-specific regression test discovered during wiring, then run it with the above set.
3. `ruff format src/fuzzer_tool/core/schedulers/kruskal_count.py src/fuzzer_tool/services/seed_picker.py src/fuzzer_tool/services/fuzzer.py src/fuzzer_tool/services/parallel.py src/fuzzer_tool/cli/commands.py src/fuzzer_tool/services/report.py src/fuzzer_tool/services/stats.py tests/test_kruskal_count.py tests/test_seed_picker.py`
4. `ruff check` on the same touched Python files.
5. Run a short real CLI smoke campaign with a local ASAN target and `--kruskal-count --elo --iterations ...`; verify the banner, live stats, report, and `state.pkl.gz` contain Kruskal data.
6. Run the affected existing seed-strategy tests and CLI/parallel tests again after formatting.
7. Compare default-path behavior with and without the new flag using the same seed and a small fixed iteration count; the default run must not construct or draw from Kruskal state.
8. Profile the new strategy on a corpus-shaped workload. Because it is opt-in, require no default-path slowdown; if scoring becomes a hot path, vectorize the independent walker matrix after the scalar correctness tests pass and retain the scalar path as the reference.
## Risks and guardrails
- Format boundary markers are target-derived hints, not a parser. The fallback jump must remain correct when hints are absent or misleading.
- Coupling can be rare on short or constant seeds. The selection floor and no-coupling fallback prevent starvation and crashes.
- Persisted state is a security boundary. Keep payloads primitive and validate them on load.
- Generation can produce duplicates. Corpus deduplication remains authoritative; the strategy must not bypass it.
- Adding a new scheduler changes Elo competition. Keep it opt-in and include it in `--elo all` only because that mode explicitly enables all strategies.
- The strategy adds per-pick CPU work. Gate scoring to the enabled strategy, cache per-seed scores while corpus content is unchanged, and measure before optimizing.
