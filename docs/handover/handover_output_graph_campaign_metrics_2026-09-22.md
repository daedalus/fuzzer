# `--output-graph`: multi-section PNG of the running campaign

## Background

Every stats tick already assembles a large set of metrics twice: once as
the human `[*] execs: ... | corpus: ...` line, and once as a JSON record
via `--log-json` (`StatsReporter._emit_json_stats`). Neither is a graph a
person can glance at. `--plot-graph` (pre-existing) renders an HTML/SVG
report, but it reads back a coverage-only CSV (`elapsed, exec_count,
cumulative_edges, corpus_size, crash_count[, novel_input_count]`) written
by `--coverage-log`, not the richer per-tick set.

This adds `--output-graph FILE.png`: a single PNG with one panel per
metric family (throughput, coverage/corpus, crashes/timeouts, memory),
rendered once at the end of the run from an in-memory history collected
on every tick.

## Design

- `StatsReporter._record_graph_snapshot(elapsed, eps)` (`services/stats.py`),
  called from the tail of `print_stats()` right after `_emit_json_stats`,
  so it fires on the exact same cadence as the existing JSON telemetry.
  It appends one dict per tick to `fuzzer._campaign_graph_history` and is
  a no-op whenever that attribute is `None` — i.e. whenever
  `--output-graph` was not passed — so a normal run pays nothing.
  Wrapped in `try/except Exception` (`pragma: no cover`) like
  `_emit_json_stats`: telemetry must not be able to kill a campaign that
  is otherwise healthy.
- `core/campaign_graph.py` (new): `render_campaign_graph(history,
  output_path, target_label)`. Lazily imports `matplotlib` (`Agg`
  backend) the same way `cmd_ppmd_stats`'s pre-existing `--graph` already
  does in `cli/commands.py`, and degrades to a printed warning rather
  than a traceback if it isn't installed, or if the history is empty.
  Four stacked subplots sharing the x-axis (elapsed seconds):
  1. Throughput — eps (raw + Kalman-filtered) on the left axis,
     cumulative execs on a twin right axis.
  2. Coverage & corpus growth — corpus size and novel-input count on the
     left axis, cumulative edges on a twin right axis (edges are only
     plotted when `shm_cov` was active, since not every run has it).
  3. Crashes & timeouts — crash count, unique signature count, and
     timeout count as step plots (these only change in jumps).
  4. Memory — peak RSS in MB.
- `cli/commands.py`: new `--output-graph FILE.png` argument on the `fuzz`
  subcommand. `cmd_fuzz` sets `fuzzer._campaign_graph_history = []`
  before the run only when the flag is present, and renders the graph
  inside the existing `finally:` block (alongside the `--log-json`
  handle close), so an interrupted or errored run still gets a graph of
  however far it got.
- **Deliberately excluded from `--hail-mary`**: `output_graph` is not in
  `_HAIL_MARY_FLAGS`, and `_apply_hail_mary` only force-enables flags at
  their argparse default anyway — a path-valued option like this one was
  never going to be swept by that loop, but this is called out explicitly
  since gating it correctly (opt-in only, never implied by kitchen-sink
  mode) was one of the stated requirements.

## Verification

Built a small stdin target, ran a short live campaign through the
installed `fuzzer-tool` entry point with `-n 5000 --no-shm
--output-graph campaign.png`, and confirmed:

- A valid PNG (1650x2100, RGBA) was written.
- All four sections render with correct labels/legends.
- The graph is still written when the run is interrupted (`finally:`
  path), not only on clean completion.

No new automated tests were added for the rendering path itself since it
is thin glue over matplotlib (already the project's precedent for
optional graph output, see `cmd_ppmd_stats`'s `--graph`); the metric
collection (`_record_graph_snapshot`) is exercised indirectly by any
existing `print_stats()` test once `_campaign_graph_history` is set.

## Files changed

- `src/fuzzer_tool/services/stats.py` — `_record_graph_snapshot` +
  call site.
- `src/fuzzer_tool/core/campaign_graph.py` — new.
- `src/fuzzer_tool/cli/commands.py` — `--output-graph` flag, history
  init, render-at-teardown.

## Pending / not done

- No test file was added; the project's regression suite
  (`test_regression_cli_fuzzer_kwargs.py`,
  `test_regression_hail_mary_gates.py`) was not re-run in this
  environment (no `pytest` installed) after this patch — please run it
  before merging.
- Panel selection is a fixed set of four; if a specific scheduler's
  diagnostics (e.g. Kuramoto sync `r(t)`) should get its own panel, that
  would be a follow-up, not done here to keep this patch reviewable.
