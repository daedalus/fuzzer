# Exec-Budget Allocation Spike — Findings

## Current Architecture

The fuzzer uses two independent mechanisms to control per-seed CPU budget:

1. **Seed selection probability** (`seed_picker.py` `_compute_weights()`): Pareto-weighted multi-signal score → softmax → selection probability. Controls WHICH seed gets mutated next. Weights incorporate fuzz count, coverage, age, and temperature (annealing).

2. **Per-selection mutation count** (`schedules.py` `SeedScorer.score()`): After a seed is selected, this produces a multiplier on `mutations_per_input` (default 8, base). Factors: speed (fast seeds get more), bitmap size (more edges = more), depth, and schedule-specific frequency adjustment (AFLFast/COE/RARE/etc). **NOTE:** This multiplier doesn't actually scale the mutation count in the current code — `mutations_per_input` is fixed at 8 (line 247 of fuzzer.py) and was never wired to use the SeedScorer's output.

3. **Mutation attempts per input** (`operators.py` line 1032): `n_mutations = f.mutations_per_input` — hard-coded to 8, with a bump to 16 during stall recovery. This is the actual exec budget per seed pick.

## Gap

The current approach is two-stage (pick → mutate N times) with no explicit "we have E execs this cycle, here's the fairest allocation" framing. The `SeedScorer` computes a multiplier that is never consumed. The actual `mutations_per_input` is a flat count, not adjusted per seed.

## Proposed: Fractional-Knapsack Budget Allocator

```python
def allocate_budget(seeds, total_execs, cycle_execs):
    """Greedy fractional-knapsack allocation of exec budget.

    Args:
        seeds: list of (score, exec_time_per_mutation, n_fuzz)
        total_execs: total execs so far
        cycle_execs: execs to allocate this cycle (e.g. 1000)

    Returns:
        dict[seed_idx, int] — mutation attempts per seed
    """
    # Value density = score * diminishing_returns_factor / exec_time
    scored = []
    for i, (score, time_per_mut, n_fuzz) in enumerate(seeds):
        # Diminishing returns: 1/sqrt(n_fuzz+1)
        density = score * (1.0 / math.sqrt(n_fuzz + 1)) / max(time_per_mut, 1e-9)
        scored.append((density, i))
    scored.sort(key=lambda x: -x[0])

    # Greedy allocation up to cycle_execs
    budget = {}
    remaining = cycle_execs
    for density, i in scored:
        time_per_mut = seeds[i][1]
        max_for_seed = min(remaining, int(cycle_execs * 0.2))  # cap at 20%
        alloc = min(max_for_seed, int(remaining * density / (density + 1e-9)))
        if alloc > 0:
            budget[i] = max(1, alloc)
            remaining -= budget[i]
    # Distribute leftovers round-robin
    idx = 0
    while remaining > 0 and scored:
        i = scored[idx % len(scored)][1]
        budget[i] = budget.get(i, 0) + 1
        remaining -= 1
        idx += 1
    return budget
```

## Comparison

| Aspect | Current (multiplier) | Proposed (knapsack) |
|--------|---------------------|---------------------|
| Control variable | Selection probability | Explicit exec budget per cycle |
| Adjusts for exec time | No (all seeds equal) | Yes (fast seeds get more) |
| Diminishing returns | Via fuzz_count in weight | Explicit 1/sqrt(n_fuzz) |
| Hard ceiling | None (seeds can dominate) | Capped at 20% of cycle |
| Starvation protection | Via annealing temperature | Via round-robin leftovers |

## Recommendation

Do not implement the knapsack allocator as a replacement. The current approach is already a reasonable approximation and adding explicit budget accounting would add complexity without clear benefit. However, **wire the SeedScorer multiplier into `mutations_per_input`** — this is a one-line change that already has the infrastructure:

In `operators.py` `mutate()`:
```python
n_mutations = f.mutations_per_input
# Apply seed-level energy multiplier if available
if f._seed_scorer:
    perf = f._seed_scorer.score(...)  # already computed in pick_seed
    n_mutations = max(1, int(n_mutations * perf / 100.0))
```

This uses the existing AFLFast/COE/RARE schedules without adding a new allocation framework. Estimated effort: ~0.5 days (need to thread the scores through the pick→mutate pipeline).
