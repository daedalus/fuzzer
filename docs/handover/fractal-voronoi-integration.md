# Handover: Fractal Jittered Voronoi Integration into fuzzer-tool

**Date:** 2026-09-01
**Author:** Integration Team
**Status:** Ready for review → implementation → Elo scheduler tuning
**Scope:** `src/fuzzer_tool/core/mutations/fractal_voronoi.py`, registry hook, tests

---

## 1. Background & Motivation

We identified a procedural-generation algorithm — **Fractal Jittered Voronoi Partitions** (Boris the Brave, 2026-08-29) — that can be repurposed as a **spatial meta-mutation operator** for the fuzzer. The algorithm partitions a 2D plane into cells with fractal, coastline-like boundaries. When mapped onto an input buffer, it creates multi-scale mutation zones where different sub-operators interact at boundaries — exactly the kind of "mixed" region that triggers parser state-machine transitions.

### Why This Fits fuzzer-tool

- The fuzzer already has **147 operators** and an **Elo arbitration scheduler**.
- The tool supports **class-based mutators** via `MutatorBase` (see `core/mutator_interface.py`).
- Fractal Voronoi adds **spatial composition** of existing operators without requiring new low-level mutations.
- The "coastline" boundaries between cells are edge-case generators — two operators meeting at a fractal boundary produce inputs neither would generate alone.

---

## 2. Algorithm Summary (from Source)

**Source:** Boris the Brave — *Fractal Jittered Voronoi Partitions* (2026-08-29)
**URL:** https://www.boristhebrave.com/2026/08/29/fractal-jittered-voronoi-partitions/

### Core Procedure

1. **Layer 0 (Root):** Infinite grid of squares. Each square gets one random site via PRNG.
2. **Layer 1:** Grid with squares half the size. Each new site finds its *parent* in Layer 0 (nearest site).
3. **Layer 2:** Halve again. Each site finds parent in Layer 1, tracing back to Layer 0.
4. **Repeat indefinitely.**

The **partition** a point belongs to is the root Layer-0 site reached by tracing parents upward.

### Computational Properties

- **Local computation:** For any point *p*, only a **5×5 neighborhood** at each layer is needed to find the nearest site.
- **Deterministic:** `hash2(layer, cell)` produces reproducible offsets.
- **Adaptive early stopping:** If *p* is close enough to a site that all deeper descendants share the same root, recursion can stop early.

### Visual Character

- Large regions reminiscent of Voronoi cells.
- Boundaries are infinitely wrinkled, fractal, coastline-like.
- Self-similar detail at every scale.
- Shader-friendly (point-by-point evaluation).

---

## 3. Integration Strategy: Three Approaches

### Approach A — Spatial Mutation Operator (`structural` category) ⭐ RECOMMENDED

**What:** Treat the input buffer as a 2D plane. Partition it with fractal Voronoi. Each cell gets a different sub-operator. Boundaries get blended mutations.

**Why:** Requires no changes to coverage/scheduling core. Just a new mutation file + registry entry. Synergizes with existing 147-operator pool.

**File:** `src/fuzzer_tool/core/mutations/fractal_voronoi.py`

**Implementation sketch:**
- Map 1D buffer to roughly-square 2D grid.
- For each byte position, compute its fractal Voronoi cell root.
- Use root hash to deterministically select a sub-operator from the registry.
- Apply operator to that byte/region.
- Coarse roots → block mutations; fine roots → bit/byte mutations.

**Registry integration:**
```python
from fuzzer_tool.core.mutations.fractal_voronoi import FractalVoronoiMutator
REGISTRY.register_mutator(FractalVoronoiMutator(
    max_depth=4,
    cell_ops=REGISTRY.get_operators(['bit', 'byte', 'block'])
))
```

**Weight:** Start at `0.05` (low), let Elo scheduler tune upward if it finds coverage.

---

### Approach B — Fractal Coverage-Guided Seed Selection (`adaptive` category)

**What:** Map the 64KB AFL coverage bitmap into 256×256 space. Use fractal Voronoi to partition coverage space. Prioritize seeds on the "coastline" between covered and uncovered regions.

**Why:** Adds spatial novelty bias. "Coastlines" in coverage space are where control flow is most sensitive to input changes.

**File:** `src/fuzzer_tool/core/schedulers/fractal_boundary.py` (new scheduler arm)

**Implementation sketch:**
- For each seed, compute boundary proximity score across its touched edges.
- Seeds near fractal boundaries get higher energy in the power schedule.
- Add as a new arm in the Elo arbitration meta-scheduler.

**Effort:** High — requires changes to `core/seed_selection.py` or scheduler internals.

---

### Approach C — Fractal Corpus Partition for Parallel Fuzzing

**What:** When running with `-j N` workers, use fractal Voronoi to partition the input hash space. Each worker owns a fractal region.

**Why:** Implicit work stealing. Workers at coarse boundaries naturally explore cross-region inputs.

**File:** `src/fuzzer_tool/core/parallel_fractal_partition.py`

**Implementation sketch:**
- Hash seed → 2D point.
- Compute root cell at depth 3.
- Assign root cells to workers via `hash(root) % N`.
- Sync interval: share seeds that crossed a fractal boundary.

---

## 4. Implementation Deliverables

This handover includes a **complete implementation of Approach A** plus registry hook and tests.

### Files Added

| File | Purpose |
|------|---------|
| `src/fuzzer_tool/core/mutations/fractal_voronoi.py` | Core `FractalVoronoiMutator` class implementing the spatial meta-operator |
| `tests/test_fractal_voronoi_mutator.py` | Unit tests for the mutator (determinism, boundary detection, decline behavior) |
| `docs/handover/fractal-voronoi-integration.md` | This document |

### Files Modified

| File | Change |
|------|--------|
| `src/fuzzer_tool/core/mutations/__init__.py` | Added import for `fractal_voronoi` module so registration runs on import |

---

## 5. Technical Details: `FractalVoronoiMutator`

### Class Signature

```python
class FractalVoronoiMutator(MutatorBase):
    name = "fractal_voronoi"
    category = "structural"

    def __init__(self, max_depth: int = 4, cell_ops: list[Callable] | None = None)
    def mutate(self, data, rng, max_len=0, *, context=None, **ctx) -> bytes | None
    def is_available(self, context, data) -> bool
```

### Key Design Decisions

1. **1D→2D mapping:** Buffer mapped to roughly-square grid via `side = int(sqrt(len(data)))`. Remainder bytes wrap to last partial row.
2. **Determinism:** Cell offsets use `hashlib.sha256(f"{layer}:{cell}").digest()` — no external PRNG state needed. Same `(layer, cell)` always maps to same operator.
3. **Operator selection:** Root cell hash modulo `len(cell_ops)` selects the sub-operator. The hash also controls mutation strength (coarse vs fine).
4. **Boundary awareness:** A byte is considered "on a boundary" if any of its 8 neighbors at the same depth maps to a different root. Boundary bytes get XOR with a blended hash — creating the "coastal" edge cases.
5. **Decline behavior:** Returns `None` for inputs < 16 bytes (too small for meaningful spatial partition).
6. **No in-place mutation:** Returns new `bytearray` per `MutatorBase` contract.

### Performance Characteristics

- **Time:** O(n × max_depth) per mutation, where n = input length. With `max_depth=4` and typical inputs < 4KB, this is ~16K operations — acceptable for a structural meta-operator.
- **Space:** O(1) auxiliary (no grid allocation; computes per-byte).
- **Cache:** `@lru_cache` on `_site()` and `_root()` within the instance. Cache size bounded by `25^max_depth` worst-case, but in practice much smaller due to spatial locality.

---

## 6. Testing Strategy

### Unit Tests (`tests/test_fractal_voronoi_mutator.py`)

| Test | What It Verifies |
|------|-----------------|
| `test_name_and_category` | Mutator registers with correct metadata |
| `test_mutate_determinism` | Same input → same output (given same rng seed) |
| `test_mutate_changes_data` | Output differs from input for non-trivial data |
| `test_declines_small_input` | Returns `None` for inputs < 16 bytes |
| `test_respects_max_len` | Output never exceeds `max_len` (registry adapter also enforces) |
| `test_is_available_always` | No external dependencies (dictionary, cmplog, etc.) |
| `test_boundary_bytes_differ` | Bytes near fractal boundaries get different treatment than interior bytes |
| `test_different_depths_produce_different_outputs` | `max_depth` parameter actually changes behavior |

### Integration Tests (recommended, not included)

- Run a 5-minute fuzzing campaign on `libpng` or `libjpeg-turbo` with and without `--fractal-voronoi`.
- Compare: unique edges found, crashes found, Elo weight evolution.
- Expected: modest unique-edge increase (5-15%) on structured targets; neutral or slight decrease on raw binary targets.

---

## 7. Registry & Scheduler Integration

### Current Hook

The mutator self-registers on module import via:

```python
def _register() -> None:
    from fuzzer_tool.core.operator_registry import REGISTRY
    m = FractalVoronoiMutator()
    if m.name not in REGISTRY.names():
        REGISTRY.register_mutator(m)

_register()
```

This mirrors the pattern in `weizz_structural.py`.

### Elo Arbitration

The mutator starts with default weight. The Elo scheduler will:
1. Shadow-run it alongside existing operators.
2. Track coverage-per-mutation attribution (via `on_new_coverage()` if needed).
3. Adjust weight based on edge-discovery rate.

**Tuning note:** If the mutator is too slow (O(n × depth)), consider reducing `max_depth` to 3 or adding an adaptive depth based on input size: `depth = min(4, int(log2(len(data)/4)))`.

---

## 8. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Performance regression (slow mutation) | Medium | Medium | Start with weight 0.05; Elo will suppress if slow. Add adaptive depth. |
| Determinism issues (non-reproducible crashes) | Low | High | SHA-256 hashing is deterministic. Cache is per-instance. Verify with `test_mutate_determinism`. |
| No coverage improvement on target | High | Low | Elo arbitration will naturally deprioritize. No harm to existing operators. |
| Buffer mapping artifacts (1D→2D) | Medium | Low | Use toroidal wrap for continuity. Test with non-square inputs. |
| Registry import order issues | Low | Medium | Registration deferred to `_register()` called at module load, not class definition. |

---

## 9. Next Steps

1. **Review** this handover and the attached patch.
2. **Apply patch:** `git am fractal-voronoi-integration.patch`
3. **Run tests:** `python -m pytest tests/test_fractal_voronoi_mutator.py -v`
4. **Smoke test:** `fuzzer-tool fuzz ./target --mutations fractal_voronoi` (verify it runs without error)
5. **A/B campaign:** Run 10-minute campaigns on a structured target (PNG parser, JSON parser) with and without the operator enabled. Compare edge counts.
6. **Tune depth:** If performance is acceptable but coverage is low, try `max_depth=5` or `max_depth=3`.
7. **Approach B/C:** If Approach A shows promise, consider implementing coverage-space prioritization (B) or parallel partitioning (C).

---

## 10. References

- Boris the Brave. *Fractal Jittered Voronoi Partitions.* 2026-08-29. https://www.boristhebrave.com/2026/08/29/fractal-jittered-voronoi-partitions/
- `daedalus/fuzzer` repository: https://github.com/daedalus/fuzzer
- `docs/handover/handover_weizz_structure_aware_port_2026-08-31.md` — prior art for class-based mutator registration
- `src/fuzzer_tool/core/mutator_interface.py` — `MutatorBase` contract
- `src/fuzzer_tool/core/operator_registry.py` — `REGISTRY.register_mutator()` API

---

*End of handover document.*
