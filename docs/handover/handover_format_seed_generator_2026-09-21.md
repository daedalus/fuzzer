# Handover: seed generator from FormatLearner's inferred fields (2026-09-21)

## Ask

Build a seed generator from the "Inferred format fields" table
`FormatLearner`/`_format_learning` (`core/analyzers/analyzer_format_learner.py`,
printed by `services/report.py::_format_learning`) already produces:

```
Offset  Width  Type  Conf  Obs  Edges  Sensitive ops
```

## What was built

`core/format_seed_generator.py`: `FormatSeedGenerator`, stdlib-only, no
dependency on a live `Fuzzer`/`OperatorEngine` — takes a list of field
hypotheses (accepts `FieldHypothesis` objects, `get_format_summary()`
dicts, or `get_state()["hypotheses"]` dicts interchangeably) plus a base
seed, and returns a list of `GeneratedSeed(data, offset, width,
field_type, strategy)`.

Per-field-type strategy, one field isolated per output seed (so an effect
in a later A/B or a `suggest_discriminating_mutation`-style comparison can
be attributed to a single field, mirroring the learner's own
discriminating-mutation philosophy rather than confounding several field
edits in one seed):

- `magic` / `padding` — never touched. `magic` is presumed required for
  parser acceptance; `padding` is already-confirmed-inert
  (`record_liveness`), so spending budget there is wasted.
- `length` — swept through width-sized boundary values (0, 1, max-for-width,
  current-payload-length ± 1, doubled) in **both** endiannesses, plus the
  existing `INTERESTING_8/16/32` AFL-style tables from
  `core/mutations/generic.py` reused rather than re-invented. The model
  doesn't record which endianness the format uses, so both are tried.
- `crc` — four cheap, algorithm-agnostic stress fills (0x00, 0xFF, 0x01,
  0x80) repeated across the field width. No attempt to recompute a valid
  checksum — the learner never records which polynomial/algorithm a field
  uses, only that many op types move coverage there, so recomputation
  isn't possible from the model's own data.
- `data` / `unknown` — replays whichever operators `sensitive_ops` says
  the learner actually saw move coverage at that offset, via a small
  local approximation table (`bit_flip`, `byte_flip`, `xor_byte`,
  `havoc_arith`) rather than calling into the live
  `OperatorEngine`/`REGISTRY.dispatch` machinery, which is built around a
  running `Fuzzer` instance (`self.f`) and isn't practical to drive
  standalone/offline. Unmapped op names fall back to a generic
  `INTERESTING_8`-based byte stress (`byte_stress`).

Fields are visited in descending
`confidence * (1 + controlled_edges) * log1p(observations)` order — same
signal `suggest_discriminating_mutation` already uses for "how much do we
trust this, and how much does it matter" — so a capped `n_seeds` budget is
spent on the highest-value fields first.

`tools/gen_format_seeds.py`: offline CLI. There's no existing in-fuzzer
hook that serializes `FormatLearner.get_state()` to disk (it's dead code
for persistence today per `core/live_bit_mask.py`'s note), so the tool
takes a JSON dump of it (documented at the top of the tool how to produce
one ad hoc from a live fuzzer instance) plus a base seed file, and writes
`id_<hash>_<field-provenance-label>.bin` files under `<out>/seeds/`,
matching the layout `adapters/filesystem.py::load_corpus` reads.

## Verified

- 22 new tests in `tests/test_format_seed_generator.py`: type-skipping
  (magic/padding), length boundary + endianness coverage, crc stress
  values + width isolation, sensitive-ops replay + fallback, score-based
  prioritization under a tight budget, out-of-range field handled without
  crashing, all three input shapes (`FieldHypothesis`,
  `get_format_summary()` dicts, `get_state()` dicts) accepted, RNG
  determinism.
- End-to-end smoke test of `tools/gen_format_seeds.py` against a
  synthetic 4-field state (magic/length/crc/data) and a hand-built base
  seed: magic correctly never mutated, other three field types produced
  the expected strategy labels and stayed within their own byte range.
- `ruff check`/`ruff format` clean on the three new files.
- `pytest tests/test_format_seed_generator.py tests/test_format_learner.py`:
  52/52 pass.

## Not done / open

- `data`/`unknown` strategies are a local approximation of a handful of
  operator names, not the real `OperatorEngine` dispatch — if the sensitive
  op recorded for a field isn't one of `bit_flip`/`byte_flip`/`xor_byte`/
  `havoc_arith`, generation falls back to generic byte stress rather than
  actually replaying that specific operator's real behavior.
- ~~No live wiring into `Fuzzer.run()`~~ — DONE 2026-09-24:
  `Fuzzer._refill_format_seeds` (stats tick, every 5000 execs, 32 seeds)
  queues variants; `OperatorEngine.mutate` drains one per round.
  Tests: `tests/test_regression_format_seed_wiring.py`. A/B owed (TODO.md).
- `crc` stress values don't attempt checksum recomputation even when a
  format's algorithm could plausibly be guessed (e.g. width=4 near a
  `zip`/`png`-classified seed) — out of scope for this pass, flagged as a
  possible future improvement if it proves valuable.
