"""Tests for --log-json structured stats output.

The record is built from the fuzzer's own attributes rather than by parsing
the human stats line, because that line is assembled from ~30 conditional
fragments whose presence depends on which strategies are enabled -- scraping
it is exactly the fragility this flag removes. These tests therefore assert
on the record's keys and values, and on the stream discipline (strict JSON
Lines, append-on-resume, never mixed into stdout).
"""

from __future__ import annotations

import io
import json
from types import SimpleNamespace

from fuzzer_tool.services.stats import StatsReporter


def _fuzzer(**overrides):
    f = SimpleNamespace(
        exec_count=1000,
        crash_count=3,
        crash_sigs={"a": 2, "b": 1},
        timeout_count=7,
        corpus=[b"x", b"y"],
        dictionary=[b"IHDR", b"IDAT"],
        shm_cov=None,
        _cmplog=None,
        _peak_rss=45092,
        _eps_filtered=812.5,
        _log_json_fh=None,
    )
    for k, v in overrides.items():
        setattr(f, k, v)
    return f


def _emit(f, elapsed=2.0, eps=500.0):
    r = StatsReporter.__new__(StatsReporter)
    r.f = f
    r._emit_json_stats(elapsed, eps)


class TestRecordContents:
    def test_core_fields(self):
        buf = io.StringIO()
        f = _fuzzer(_log_json_fh=buf)
        _emit(f, elapsed=2.0, eps=500.0)
        rec = json.loads(buf.getvalue())
        assert rec["execs"] == 1000
        assert rec["crashes"] == 3
        assert rec["crash_sigs"] == 2
        assert rec["timeouts"] == 7
        assert rec["corpus"] == 2
        assert rec["dict_tokens"] == 2
        assert rec["elapsed"] == 2.0
        assert rec["eps"] == 500.0
        assert rec["eps_filtered"] == 812.5
        assert rec["peak_rss_kb"] == 45092
        assert isinstance(rec["ts"], float)

    def test_coverage_fields_only_when_shm_present(self):
        buf = io.StringIO()
        _emit(_fuzzer(_log_json_fh=buf))
        assert "edges" not in json.loads(buf.getvalue())

        buf = io.StringIO()
        shm = SimpleNamespace(cumulative_edges=1234, read_dropped_edges=lambda: 56)
        _emit(_fuzzer(_log_json_fh=buf, shm_cov=shm))
        rec = json.loads(buf.getvalue())
        assert rec["edges"] == 1234
        assert rec["dropped_edges"] == 56

    def test_cmplog_fields_only_when_enabled(self):
        buf = io.StringIO()
        _emit(_fuzzer(_log_json_fh=buf))
        assert "cmplog_tokens" not in json.loads(buf.getvalue())

        buf = io.StringIO()
        cmplog = SimpleNamespace(
            tokens=[1, 2, 3],
            pairs=[1, 2],
            total_comparisons=lambda: (900, 17),
        )
        _emit(_fuzzer(_log_json_fh=buf, _cmplog=cmplog))
        rec = json.loads(buf.getvalue())
        assert rec["cmplog_tokens"] == 3
        assert rec["cmplog_pairs"] == 2

    def test_comparison_counters_recorded(self):
        """Fired/asserted totals ride alongside the token and pair counts.

        They measure different things: tokens and pairs are what survived
        dedup, these are what the target actually executed.
        """
        buf = io.StringIO()
        cmplog = SimpleNamespace(
            tokens=[1, 2, 3],
            pairs=[1, 2],
            total_comparisons=lambda: (900, 17),
        )
        _emit(_fuzzer(_log_json_fh=buf, _cmplog=cmplog))
        rec = json.loads(buf.getvalue())
        assert rec["cmplog_cmp_fired"] == 900
        assert rec["cmplog_cmp_asserted"] == 17

    def test_no_dictionary_reports_zero_not_crash(self):
        buf = io.StringIO()
        _emit(_fuzzer(_log_json_fh=buf, dictionary=None))
        assert json.loads(buf.getvalue())["dict_tokens"] == 0

    def test_novel_inputs_defaults_to_zero_when_absent(self):
        """A fuzzer instance that predates the counter must not crash the
        emitter -- getattr fallback, not a hard attribute access."""
        buf = io.StringIO()
        _emit(_fuzzer(_log_json_fh=buf))
        assert json.loads(buf.getvalue())["novel_inputs"] == 0

    def test_novel_inputs_reported_when_present(self):
        buf = io.StringIO()
        _emit(_fuzzer(_log_json_fh=buf, _novel_input_count=42))
        assert json.loads(buf.getvalue())["novel_inputs"] == 42


class TestStreamDiscipline:
    def test_disabled_by_default_writes_nothing(self):
        f = _fuzzer()  # _log_json_fh is None
        _emit(f)  # must not raise

    def test_one_line_per_tick_and_strict_jsonl(self):
        buf = io.StringIO()
        f = _fuzzer(_log_json_fh=buf)
        for i in range(3):
            f.exec_count = 1000 * (i + 1)
            _emit(f)
        lines = buf.getvalue().splitlines()
        assert len(lines) == 3
        assert [json.loads(ln)["execs"] for ln in lines] == [1000, 2000, 3000]

    def test_a_broken_stream_never_aborts_the_run(self):
        """Telemetry must not be able to kill a campaign."""

        class _Exploding:
            def write(self, _):
                raise OSError("disk full")

            def flush(self):
                pass

        _emit(_fuzzer(_log_json_fh=_Exploding()))  # must not raise

    def test_unserialisable_value_is_swallowed(self):
        buf = io.StringIO()
        f = _fuzzer(_log_json_fh=buf, _peak_rss=object())
        _emit(f)  # must not raise
        assert buf.getvalue() == ""
