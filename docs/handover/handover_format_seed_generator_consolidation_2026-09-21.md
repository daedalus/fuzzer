# Format-learner seed generation: consolidating two independent implementations

2026-09-21, on top of `c5c62989` (feat(mutations): add covering_array_ihdr).

## Background

Two implementations of "turn `FormatLearner`'s field hypotheses back into a
seed" landed independently and in parallel:

1. `core/format_seed_generator.py` (`FormatSeedGenerator`, commit
   `29b9a8e5`) — stdlib-only, no live `Fuzzer`/`OperatorEngine` dependency,
   given a **base seed** it emits up to N field-targeted *variants*: length
   fields get width-appropriate boundary sweeps in both endiannesses, crc
   fields get algorithm-agnostic stress fills, data/unknown fields replay
   whichever mutation operators the learner saw move coverage there. Used
   by the offline `tools/gen_format_seeds.py` CLI.

2. `SeedPicker._format_learner_seed()` (commit `0d2c1870`) — lives inside
   the live fuzzer's cold-start path (`_format_aware_seed`), builds **one**
   seed from nothing by filling each confident field with its
   `most_common_value` (a per-position byte-frequency histogram
   `FormatLearner._track_values` added in the same commit), with a 30%
   per-field chance of substituting a type-specific stress default instead.

These aren't actually redundant — one needs a base seed and produces many
variants, the other builds a single seed with nothing to start from — but
they duplicated the confidence-filtering, type-default, and field-iteration
logic, and disagreed on where the "which byte fills the field" model lives
(the `sensitive_ops`-replay approach vs. the `value_counts` histogram).

## What changed

Moved `SeedPicker._format_learner_seed`'s algorithm into
`core/format_seed_generator.py` as a new function, `cold_start_seed()`,
byte-for-byte identical to the original (same confidence threshold, same
type-default table, same 30%-override roll, same offset-ascending
iteration). `SeedPicker._format_learner_seed()` is now a ~10-line wrapper
that calls it. No behavior change at either call site — all 25 pre-existing
`TestFormatLearnerSeed` cases in `tests/test_seed_picker.py` pass unchanged,
including the `Random(42)`-seeded determinism ones (the 30%-override roll
only fires when a field type has a non-empty default list, so a plain
`random.Random` vs. `RandPool` gives the same result for `"unknown"`-typed
fields — confirmed the delegation doesn't disturb that).

`_FieldSpec` (the internal representation `FormatSeedGenerator` and now
`cold_start_seed()` both build from) gained a `most_common_value` field,
populated either straight from a `get_format_summary()` dict or reduced
from a `FieldHypothesis.value_counts` histogram via a new
`_most_common_value_from_counts()` helper — same reduction
`FormatLearner.get_format_summary()` itself does. `_coerce_fields()` (the
existing FieldHypothesis-objects-or-dicts adapter) now populates it for
both input shapes, so `cold_start_seed()` accepts the same three input
shapes `FormatSeedGenerator` already did: live `FormatLearner.hypotheses`,
`get_format_summary()["fields"]` dicts, or `get_state()["hypotheses"]`
dicts.

Net effect: one field-hypothesis → seed-bytes model, shared by the live
fuzzer's cold-start path and the offline CLI, instead of two that could
silently drift apart. `cold_start_seed()`'s output can now also be handed
to `FormatSeedGenerator(fields).generate(cold_start_seed(...), n_seeds=...)`
as its base seed — previously cold-start had no path into the
variant-generator at all, since it only ran inside `SeedPicker` and never
touched `core/format_seed_generator.py`.

## What didn't change

- `FormatSeedGenerator.generate()` and its field-type strategies
  (`_length_candidates`, `_crc_candidates`, `_SENSITIVE_OP_TABLE` replay) —
  untouched.
- `tools/gen_format_seeds.py` — untouched, still imports
  `FormatSeedGenerator` directly.
- `SeedPicker._format_aware_seed()`'s call site and fallback chain
  (learner seed → `MINIMAL_SEEDS` → generated-format mutator → random
  buffer) — untouched, still calls `self._format_learner_seed()` first.

## Testing

- `tests/test_seed_picker.py::TestFormatLearnerSeed` (25 cases, unchanged):
  passes against the delegating implementation.
- `tests/test_format_seed_generator.py` gained `TestColdStartSeed` (6 new
  cases): confidence-bar rejection, missing-value rejection,
  most-common-value fill, `max_len` truncation, `get_format_summary()`-dict
  input, and an end-to-end case that feeds one live `FormatLearner`
  through both `get_format_summary()["fields"]` and raw `.hypotheses` and
  asserts identical output — the two shapes `SeedPicker` and the offline
  CLI respectively hand in.
- `ruff check` clean on both touched files.
- Full relevant suite: 88 passed (`test_seed_picker.py` +
  `test_format_seed_generator.py` + `test_format_learner.py`).
