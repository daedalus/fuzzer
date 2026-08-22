# ExhaustivePool: enumerating the operator table, and two max_len escapes

Date: 2026-08-22

Added: `src/fuzzer_tool/core/exhaustive_pool.py`, `tests/test_exhaustive_pool.py`,
`tests/support/operator_env.py`.
Changed: `src/fuzzer_tool/services/operators.py`, `tests/test_new_operators.py`.

Port item P1-5 from `docs/tigerbeetle_four_fuzzers_port.md`.

## The trick

`core/rand_pool.py` was already the abstraction the port needed: every
discrete draw carries an explicit bound, and operators receive the pool as
`self.f._rand_pool` rather than reaching for a module-level `random`.
Substituting an object with the same method names turns "run this operator
once with random draws" into "run it once per reachable combination of
draws" — with no enumeration logic written per operator.

`ExhaustivePool` is an odometer of `[value, bound]` positions in draw
order. Each run reads values at its positions, appending `[0, bound]` where
it goes deeper than before; between runs the last incrementable position
advances and everything after it is discarded, so the next run replays an
identical prefix and then diverges.

The property that makes it work on real operators is that positions are
defined by *draw order*, not call site. Operators branch constantly — on
buffer length, on whether a candidate list came back empty, on which byte
was picked — and the tree handles it, because a replayed prefix is
byte-identical and therefore takes the identical path.

## What it found

**Two operators exceeded `max_len` on reachable paths.** Both mutate in
place and return `None`, so neither reaches `mutate()`'s post-operator
`f.max_len` clamp — that clamp only runs in the `result is not None`
branch. `_op_fuse_this`'s docstring already warned about exactly this after
the same class of bug cost an unbounded-growth incident; these two were
live instances of it that nobody had walked into.

| operator | with `max_len=8` | shape |
|---|---|---|
| `regex_bomb` | 13 | extends the buffer to the pattern length unconditionally, so a cap below the longest bomb was simply ignored |
| `utf8_widen` | 9 | grows by exactly +1 per application |

`utf8_widen` is the more interesting one. A one-byte overrun looks like
rounding — but the operator can be selected again on its own output, and
nothing downstream caps this path. It is `_op_fuse_this` at a slower rate.

Every *other* operator respects `max_len` at caps of 1, 2, 4, 8 and 16, so
the two were not following a different convention. They had simply never
been run down the right path: random testing had not reached either in the
life of the project.

Both fixed to follow the convention `_op_clone_fixed` already uses —
decline rather than truncate. A truncated overlong UTF-8 sequence is not an
overlong sequence and a truncated backtracking bomb is not a bomb, so
emitting either would make the operator look like it had fired when it had
not.

**Nothing draws entropy the pool does not own.** `NondeterministicDrawError`
fires when a replayed prefix requests a different bound, which is how a
module-level `random`, a clock, or a set iteration order would surface.
Across the whole table, none does. A negative result, and worth recording:
it means `--fuzz-seed` genuinely controls operator behaviour, which had
been assumed rather than checked.

## Census: 70 of 134 operators fully enumerable

On an 8-byte buffer with `max_depth=16`, `max_runs=4000`:

| status | count | why |
|---|---|---|
| enumerated | 70 | the space was walked |
| continuous | 21 | drew `random()` / `gauss()` / … |
| bulk | 14 | drew `randbytes` or a `*_list` variant |
| over budget | 25 | finite but above 4000 runs at this size |
| too deep | 4 | more than 16 bounded draws in one run |

The **21 continuous** are the actionable number. Most are not continuous in
any real sense — they are coin flips written `rng.random() < 0.5` instead
of `rng.randint(0, 1)`. `_op_arithmetic` is typical: it picks a width, an
index and a delta through bounded draws, then negates the delta with
`rng.random() < 0.5`. That single line is the only thing keeping it out of
the enumerable set. Converting them is mechanical, does not change the
distribution, and would move most of that 21 across; `ContinuousDrawError`
says so in its message, since the person who hits it is the person who can
fix it.

The **25 over budget** shrink with a smaller input, not with more budget:
at `max_len=1` the enumerable count rises to 102.

## Design decisions worth keeping

**Refuse, never guess.** Continuous, bulk-without-opt-in, and
depth-exceeded all raise. The alternative — return something plausible and
carry on — is worse than useless here, because the walk would still report
`exhausted`. An enumeration that quietly skips a draw is a test suite that
lies about its own coverage.

**`exhausted` is separate from the run count.** Reaching `max_runs` sets
`budget_exhausted` and leaves `exhausted` false, so a test asserting
`pool.exhausted` cannot pass on a partial walk.

**A bound of 1 is not recorded.** Operators call `randint(0, 0)` constantly,
any time a computed span leaves one legal position. Recording those would
multiply the run count by 1 while doubling depth — and depth is the budget
that runs out first.

**Degenerate inputs mirror `RandPool` exactly.** It returns 0 for a
non-positive `randrange` and `a` for an inverted `randint` rather than
raising. An enumeration that raised instead would report operators as
broken that are not.

**`weighted_choice` ignores magnitudes but not zeros.** A 1-in-1000 branch
is visited as often as the other 999 — that is the point of enumerating.
Zero weight is different in kind, being unreachable, so it is excluded.

## A note on the harness

`tests/support/operator_env.py` is the 95-line `Fuzzer` mock, moved
verbatim out of `test_new_operators.py` so both suites share it, and given
a `pool=` parameter as the injection seam.

Its existence is the argument for P1-4 restated: 29 attributes, mostly
`None` or `False`, present only because `OperatorEngine`'s declared
contract is the whole `Fuzzer`. `test_operator_smoke.py` takes the other
road and constructs a real one, which is why it needs a compiled target
binary on disk to call `_op_bit_flip`.

## Verification

- 46 tests, 2.7s. Against the unfixed operators, 4 fail — the three
  targeted regressions plus the whole-table `max_len` sweep.
- Cardinalities checked in closed form: products, `n!` for n=1..5,
  `n!/(n-k)!`, and `256**2` under `allow_bulk`.
- Full suite: 4903 passed, 180 skipped, 1 xfailed.
- `ruff check` and `ruff format --check` clean.

## Not done

`ExhaustivePool` is not wired into any campaign path and should not be — it
is a test-time substitute. The natural follow-ons are converting the 21
`random() < p` coin flips to bounded draws, which is cheap and would roughly
double the enumerable set, and P2-6's negative-space generator for
`core/state_store.py`, which wants this pool to drive its corruption
choices.
