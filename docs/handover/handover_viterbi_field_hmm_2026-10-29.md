# Plan: Viterbi field segmentation (`--field-hmm`)

Status: implemented 2026-10-29 (DEEP_DIVE §Viterbi field segmentation);
A/B owed (`docs/AB_PENDING.md`, bench arm `field-hmm`). TODO: `docs/TODO.md` §Operators.

Deviations from the plan below:

- Spans are not stored in `seed_meta`: they are a pure function of (model,
  seed), cached per seed in the analyzer (FIFO 512) and recomputed after a
  refit or resume. No epoch, no stale-span path.
- Smoothing is fixed (1.0 transition, 0.1 emission), not `dm_alpha`: a
  refit alpha would break the non-decreasing objective.
- `_LIGHTWEIGHT_OFF` untouched: it lists default-on features only.
- Decoder bound once per model: pure Python for K ≤ 4, numpy for K > 4
  (§7 numbers in DEEP_DIVE). Training is batched across seeds.
- §8 open questions unchanged: K is a flag (no BIC), raw-byte emissions only,
  Weizz tags still not persisted.

## 1. Goal

Label each corpus byte with a hidden field state (magic, length, payload, ...)
with no cmplog, colorize or format knowledge. Mutators then work on whole
fields instead of flat bytes.

```
seed bytes   89 50 4E 47 | 00 00 00 0D | 49 48 44 52 | ...
Viterbi path  s0 s0 s0 s0 | s1 s1 s1 s1 | s0 s0 s0 s0 | s2 s2 ...
spans        [(0,4,s0), (4,8,s1), (8,12,s0), (12,..,s2)]
                   │
   hmm_segment op ─┴─► havoc / splice / dup / delete one span
```

## 2. Why not an existing labeller

| Labeller | Needs | Gap the HMM fills |
|---|---|---|
| Weizz `StructureMap` (`core/weizz_tags.py`) | cmplog (+ colorize) | targets without cmplog; tags lost on `--resume` |
| `FormatLearner` (`analyzer_format_learner.py`) | per-signature rule `_classify` | fixed-width rules, no contiguous segmentation |
| `field_map.py` | known formats | unknown formats |

The HMM reuses their labels where present (init and state naming, §4.3)
rather than adding a parallel taxonomy.

**Hard limit:** states are anonymous. Byte-value emissions separate
"zero-heavy / ASCII / high-entropy" well; they do not separate a length
field from payload. Naming comes from overlap with Weizz/FormatLearner
labels only; otherwise a state is `FieldKind.UNKNOWN`, still usable for
segmentation and state-aware splicing.

## 3. Model

- K states (default 8), first-order transitions, 256-way categorical
  emission per state. Params in log space: `log_pi[K]`, `log_A[K,K]`,
  `log_B[K,256]` (float64, ~17 KB).
- Decode: Viterbi, O(n·K²), n capped at `DECODE_MAX = 4096` bytes
  (Hard Rule 54); tail beyond cap gets no spans.
- Train: hard EM (Viterbi training). Decode sample, recount, smooth with
  `core/dirichlet.dm_alpha`. One code path; no Baum-Welch forward-backward
  needed. Retrain every `RETRAIN_EVERY = 64` admits on at most
  `TRAIN_SEEDS = 256` corpus entries.
- Init deterministic (no RNG): from Weizz/FormatLearner labels when present,
  else byte-class heuristic (zero, ASCII, high-entropy, other).

## 4. Code sketches

### 4.1 `core/field_hmm.py` (pure model, no fuzzer imports)

```python
DECODE_MAX = 4096


class FieldKind(Enum):
    UNKNOWN = 0
    MAGIC = 1
    LENGTH = 2
    CHECKSUM = 3
    PAYLOAD = 4


class FieldHMM:
    """K-state byte HMM: Viterbi decode, hard-EM training."""

    def __init__(self, n_states: int = 8):
        self._k = n_states
        self._log_pi, self._log_A, self._log_B = _uniform(n_states)
        self._kinds = [FieldKind.UNKNOWN] * n_states

    def decode(self, data: bytes) -> np.ndarray:
        """Most likely state per byte (uint8), first DECODE_MAX bytes."""
        obs = np.frombuffer(data[:DECODE_MAX], np.uint8)
        return _viterbi(obs, self._log_pi, self._log_A, self._log_B)

    def spans(self, data: bytes) -> list[tuple[int, int, int]]:
        """Run-length (start, end, state) of decode()."""

    def fit(self, seeds: Sequence[bytes], iters: int = 4) -> float:
        """Hard EM; returns final log-likelihood."""

    def name_states(self, labelled: Iterable[tuple[np.ndarray, list[FieldKind]]]) -> None:
        """Majority-vote FieldKind per state from external labels."""

    def to_state(self) -> dict: ...  # plain dict + ndarray (StateStore allowlist)
    def load_state(self, d: dict) -> None: ...


def _viterbi(obs, log_pi, log_A, log_B) -> np.ndarray:
    n, k = obs.size, log_pi.size
    if n == 0:
        return np.empty(0, np.uint8)

    # Gather emissions once (fors down): emit[t, s] = log_B[s, obs[t]].
    emit = log_B[:, obs].T
    back = np.empty((n, k), np.uint8)
    delta = log_pi + emit[0]
    cols = np.arange(k)

    # Forward pass: best predecessor per state.
    for t in range(1, n):
        scores = delta[:, None] + log_A
        back[t] = scores.argmax(axis=0)
        delta = scores[back[t], cols] + emit[t]

    # Backtrack.
    path = np.empty(n, np.uint8)
    path[-1] = delta.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path
```

Hard Rule 14: also write a pure-Python `array`-based `_viterbi` for K ≤ 8;
benchmark both at n ∈ {64, 512, 4096}; keep the faster (likely numpy at
large n, Python at small n — if split, pick once per call by n, not per step).

### 4.2 `core/analyzers/analyzer_field_hmm.py` (lifecycle)

```python
class FieldHmmAnalyzer:
    """Owns FieldHMM; labels admitted seeds; periodic refit."""

    def __init__(self, f, n_states: int): ...
    def on_admit(self, data: bytes) -> None:
        # Label seed; refit every RETRAIN_EVERY admits.
    def spans(self, data: bytes) -> list[tuple[int, int, int]]:
        # From seed_meta["hmm_spans_rle"]; [] if absent.
    def same_state(self, state: int, exclude: bytes) -> bytes | None:
        # A span of `state` from another seed, for splicing.
```

After a refit, stored spans are stale: bump `self._epoch`, store epoch with
spans, relabel lazily on read (Hard Rule 57 "cache safely").

### 4.3 Op `hmm_segment` (`services/operators.py`)

```python
class _SegAction(Enum):
    HAVOC = 0
    SPLICE = 1
    DUP = 2
    DELETE = 3


def _op_hmm_segment(self, buf, byte_idx, data):
    hmm = getattr(self.f, "_field_hmm", None)
    spans = hmm.spans(data) if hmm else []
    if not spans or len(buf) != len(data):
        return self._op_declined("hmm_segment", buf)

    rng = self.ctx._rng
    start, end, state = spans[rng.choice(range(len(spans)))]
    action = _SegAction(rng.randint(0, len(_SegAction) - 1))
    return _SEG_ACTIONS[action](self, buf, start, end, state)
```

Add to `_DELOCALISED_OPS`. Action table built once (ifs up).

## 5. Wiring (full)

| # | File | Change |
|---|---|---|
| 1 | `core/field_hmm.py` | new: `FieldHMM`, `_viterbi`, `FieldKind` |
| 2 | `core/analyzers/analyzer_field_hmm.py` | new: `FieldHmmAnalyzer` |
| 3 | `core/analyzer_registry.py` | `AnalyzerSpec("field_hmm", available=lambda f: f._use_field_hmm, activate=...)` → builds `f._field_hmm`, restores from `f._state_store.get("field_hmm")` (pattern: length_tracker `:300`) |
| 4 | `cli/commands.py` | `--field-hmm` (store_true), `--field-hmm-states` (int, 8); pass in `Fuzzer(...)` (`:637`); add to `_EXCLUDED_OPT_IN` with "unmeasured" comment; add to `_LIGHTWEIGHT_OFF` |
| 5 | `services/fuzzer.py` | `__init__` kwargs `field_hmm=False`, `field_hmm_states=8` → `self._use_field_hmm`, `self._field_hmm_states`; end-of-run save `self._state_store.set("field_hmm", self._field_hmm.to_state())` next to markov save (`:10506`) |
| 6 | `services/corpus_manager.py` | `_after_admit` (`:1387`): call bound `self._hmm_admit(data)` (no-op lambda when off, bound at setup); add `"hmm_spans_rle"`, `"hmm_epoch"` to persisted seed_meta keys (`:622-646`) and `_restore_seed_meta` (`:768`) |
| 7 | `core/operator_registry.py` | `"hmm_segment"` in `_CATEGORIES["adaptive"]`; `_AVAILABLE["hmm_segment"] = lambda f, d: bool(getattr(f, "_field_hmm", None)) and f._field_hmm.has_spans(d)` |
| 8 | `services/operators.py` | `_op_hmm_segment`, `_SEG_ACTIONS`; `"hmm_segment"` in `_DELOCALISED_OPS` |
| 9 | phase 2 | position proposer: span boundaries in `_side_positions` (`:6812`) — only if §7 A/B of the op is positive |
| 10 | docs | `DEEP_DIVE.md` §Viterbi field segmentation; `AB_PENDING.md` entry; `architecture.dot` + png/svg (Hard Rule 44) |

No scheduler changes: op discovered via `REGISTRY` (Hard Rule 12).

## 6. Tests (write first, observe red — Hard Rule 38)

`tests/test_field_hmm.py`:

- **Falsification:** `_viterbi` == brute-force argmax over all Kⁿ paths
  (K=3, n=6, params from seeded `RandPool`). Control first (Hard Rule 46):
  brute force vs a second brute-force run on the same params.
- **Falsification:** synthetic `b"\x00"*40 + b"\xff"*40` with 2-state
  hand-set params → boundary exactly at 40.
- **Property:** hard-EM log-likelihood non-decreasing across `fit` iters.
- **Adversarial:** empty input, 1 byte, all-identical bytes, input >
  `DECODE_MAX` (spans stop at cap), `log_A` with `-inf` entries (forbidden
  transitions never in path), `n_states=1`, `n_states=256` (uint8 backptr limit).
- **Persistence:** `to_state` round-trips through `StateStore` save/load
  (allowlist enforced by `_verify_readable`).
- **Stale spans:** refit bumps epoch; old spans relabelled on read.

`tests/test_regression_operator_registry.py`: category placement, gate
off without `_field_hmm`, gate off for seed without spans.

Op test (pattern `tests/test_byte_interaction.py:213`): `SimpleNamespace`
engine + `ScriptedRng(choice_idxs=..., randints=...)`; assert exact output per
`_SegAction`; decline when `len(buf) != len(data)`. No retry loops (Hard Rule 39).

Auto-covered: `test_regression_cli_fuzzer_kwargs.py`,
`test_regression_hail_mary_gates.py`. `lizard --CCN=15` on both new files.

## 7. Performance and measurement

- Decode cost per admit: §4.1 sketch measured 14.7 ms at n=4096, K=8
  (scratch microbench; matched brute force on 200 random K=3, n=6 cases). Admits are rare vs execs; refit bounded by `TRAIN_SEEDS`.
  Measure with `--profile-hotpath` on fuzzgoat; off-path cost must be zero
  (bound no-op, Hard Rule 57).
- Memory: backptr ≤ 4096·K bytes transient; per-seed spans RLE.
- A/B: paired fuzzgoat, clang build (Hard Rule 52), `--field-hmm` on vs off,
  base-vs-base control first. Then a structured target (png).
- **Kill criterion:** on a png corpus, if decoded boundaries agree with
  Weizz magic/length span edges no better than a random segmentation with
  the same span count, drop the feature and record the null result.

## 8. Open questions

1. K: fixed 8, or pick by BIC on first fit?
2. Emissions on raw bytes only, or add a byte-delta feature to help separate
   length/counter fields (phase 2)?
3. Persist Weizz tags too (same `seed_meta` keys path) — separate TODO.
