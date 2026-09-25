# Handover — seed_overhead (RSS/seed) diagnostic — 2026-09-25

## Request
Add a diagnostic field: peak RSS divided by the number of live corpus
seeds ("seed_overhead"), reported as MB/seed. Explicitly diagnostic only
— nothing acts on it. Run only the tests affected by the change.

## Base
`git pull` on `master` (repo: https://Github.com/daedalus/fuzzer) left
`HEAD` unchanged at `75b3f2f6` (`feat(ops): lz4_chunk_mutate and
rar_chunk_mutate`) — no new upstream commits since the last recorded
state (`e77573d2` was the previous known HEAD; the intervening history
was already on this clone).

## What changed
`src/fuzzer_tool/services/stats.py`:

- New `StatsReporter._print_stats_seed_overhead_str(self, f)`, placed
  next to `_print_stats_seed_energy_gini_str` /
  `_print_stats_op_gini_str` (same diagnostic family, same
  degrade-to-`""` convention so a partially-mocked or not-yet-warmed-up
  `Fuzzer` never takes down `print_stats`):
  - Reads `f._peak_rss` (KB, same field the existing `rss:` status-line
    entry already uses) and `len(f.corpus)`.
  - Returns `""` if RSS is falsy/zero, or the corpus is empty/absent —
    avoids a division by zero during startup, before any seed is
    loaded.
  - Otherwise: `mb_per_seed = (rss_kb / 1024.0) / len(corpus)`, printed
    as `" | seed-ovh: X.XXMB/seed"`.
- Wired into the existing `dr_str` fragment chain inside `print_stats`
  (same chain that already carries `seed-gini`/`op-gini`), so it shows
  up in the status line right after those two, e.g.:
  `... | seed-gini: 0.59 | op-gini: 0.64 | seed-ovh: 2.79MB/seed | ...`
  (worked from the status line in the request: 2646MB / 949 seeds ≈
  2.79MB/seed).

Nothing reads this value back — no scheduler, scorer, or admission
control touches it. It is exactly as inert as the existing seed-gini/
op-gini fields it sits next to.

## Tests
New file `tests/test_regression_stats_seed_overhead_fragment.py`,
mirroring the pattern in `tests/test_regression_stats_gini_fragments.py`
(same hazard: a `MagicMock` that answers every attribute, or a bare
`object()`, must still degrade to `""` rather than raise). 6 cases:
zero RSS, absent RSS (`object()` stand-in), empty corpus, absent
corpus, the worked 2646MB/949-seed example (→ `2.79MB/seed`), and a
single-seed corpus.

Ran only the affected slice (`pytest tests/ -k stats`, plus the new
file and `test_regression_stats_gini_fragments.py` /
`test_stats_reporter.py` explicitly): **274 passed, 6 skipped**
(skips are pre-existing/unrelated — `test_smt_solver.py`'s 4 skips).
Same result reproduced from scratch on an independent fresh clone
after `git am`. `ruff check` clean on both touched files (the ~72
pre-existing findings elsewhere in the repo are untouched by this
change).

## Verification
Patch generated with `git format-patch -1` on commit `5b966cb7`
(parent `75b3f2f6`), applied with `git am` on an independent fresh
clone of `master` at `75b3f2f6` — applied clean, tests re-run there
and passed identically.

## Delivered
Single patch (`0001-diag-stats-add-seed_overhead-RSS-seed-to-print_stats.patch`)
zipped together with this handover doc.

## Open / not done
Nothing follow-on identified for this task — it was scoped as a single
diagnostic field, delivered as such.
