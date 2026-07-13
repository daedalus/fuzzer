"""Tests for the plotting module — SVG chart generation and HTML report."""

import types

from fuzzer_tool.core.plotting import (
    _derive_exec_rate,
    _scale,
    _svg_bar_chart,
    _svg_line_chart,
    generate_html_report,
    read_coverage_log,
)


class TestReadCoverageLog:
    def test_missing_file_returns_empty(self, tmp_path):
        assert read_coverage_log(tmp_path / "nope.csv") == []

    def test_parses_valid_rows(self, tmp_path):
        p = tmp_path / "cov.csv"
        p.write_text("1.0,10,5,2,0\n2.0,20,8,3,1\n")
        rows = read_coverage_log(p)
        assert len(rows) == 2
        assert rows[0] == {
            "elapsed": 1.0,
            "exec_count": 10,
            "cumulative_edges": 5,
            "corpus_size": 2,
            "crash_count": 0,
        }
        assert rows[1]["crash_count"] == 1

    def test_skips_malformed_rows(self, tmp_path):
        p = tmp_path / "cov.csv"
        p.write_text("1.0,10,5,2,0\nnot,a,valid,row\n3.0,30,9,4,1\n")
        rows = read_coverage_log(p)
        # Malformed row (wrong types) is skipped, valid ones kept.
        assert len(rows) == 2
        assert rows[-1]["exec_count"] == 30

    def test_skips_rows_with_wrong_column_count(self, tmp_path):
        p = tmp_path / "cov.csv"
        p.write_text("1.0,10,5,2,0\n1.0,10,5\n")
        rows = read_coverage_log(p)
        assert len(rows) == 1


class TestDeriveExecRate:
    def test_empty_input(self):
        assert _derive_exec_rate([]) == []

    def test_single_row_produces_no_points(self):
        rows = [{"elapsed": 1.0, "exec_count": 10}]
        assert _derive_exec_rate(rows) == []

    def test_computes_rate_between_rows(self):
        rows = [
            {"elapsed": 0.0, "exec_count": 0},
            {"elapsed": 2.0, "exec_count": 20},
        ]
        points = _derive_exec_rate(rows)
        assert len(points) == 1
        elapsed, rate = points[0]
        assert elapsed == 2.0
        assert rate == 10.0  # 20 execs / 2 seconds

    def test_skips_non_positive_time_deltas(self):
        rows = [
            {"elapsed": 5.0, "exec_count": 10},
            {"elapsed": 5.0, "exec_count": 20},  # dt == 0, would divide by zero
            {"elapsed": 4.0, "exec_count": 30},  # dt < 0, out of order
        ]
        # Should not raise, and should skip both bad deltas.
        assert _derive_exec_rate(rows) == []


class TestScale:
    def test_maps_midpoint(self):
        assert _scale(5, 0, 10, 0, 100) == 50

    def test_degenerate_range_returns_output_midpoint(self):
        # lo == hi would otherwise divide by zero.
        assert _scale(5, 5, 5, 0, 100) == 50


class TestSvgLineChart:
    def test_empty_points_renders_placeholder(self):
        svg = _svg_line_chart("Edges", "elapsed", "edges", [])
        assert "<svg" in svg
        assert "no data yet" in svg

    def test_nonempty_points_renders_path(self):
        svg = _svg_line_chart("Edges", "elapsed", "edges", [(0.0, 1.0), (1.0, 5.0), (2.0, 3.0)])
        assert "<svg" in svg
        assert "<path" in svg
        assert "Edges" in svg

    def test_single_point_does_not_crash(self):
        # A single point means x_lo == x_hi; must not raise ZeroDivisionError.
        svg = _svg_line_chart("Edges", "elapsed", "edges", [(1.0, 1.0)])
        assert "<svg" in svg

    def test_flat_line_does_not_crash(self):
        # All same y value; y_hi - y_lo == 0, must not raise.
        svg = _svg_line_chart("Edges", "elapsed", "edges", [(0.0, 5.0), (1.0, 5.0), (2.0, 5.0)])
        assert "<svg" in svg

    def test_escapes_title(self):
        svg = _svg_line_chart("<script>alert(1)</script>", "x", "y", [])
        assert "<script>" not in svg
        assert "&lt;script&gt;" in svg


class TestSvgBarChart:
    def test_empty_bars_renders_placeholder(self):
        svg = _svg_bar_chart("Ops", [])
        assert "no data yet" in svg

    def test_nonempty_bars_renders_rects(self):
        svg = _svg_bar_chart("Ops", [("bit_flip", 0.1), ("havoc", 0.0)])
        assert svg.count("<rect") >= 2  # background + at least one bar
        assert "bit_flip" in svg
        assert "10.0%" in svg

    def test_zero_value_bar_does_not_crash(self):
        svg = _svg_bar_chart("Ops", [("dead_op", 0.0)])
        assert "<svg" in svg

    def test_escapes_label(self):
        svg = _svg_bar_chart("Ops", [("<b>x</b>", 0.5)])
        assert "<b>x</b>" not in svg


class TestGenerateHtmlReport:
    def test_writes_report_with_no_log(self, tmp_path):
        fuzzer = types.SimpleNamespace(
            op_counts={}, op_success={}, exec_count=0, crash_count=0, corpus=[]
        )
        out = generate_html_report(fuzzer, tmp_path / "missing.csv", tmp_path / "report.html")
        assert (tmp_path / "report.html").exists()
        content = (tmp_path / "report.html").read_text()
        assert "<html>" in content
        assert out == str(tmp_path / "report.html")

    def test_writes_report_with_data(self, tmp_path):
        log = tmp_path / "cov.csv"
        log.write_text("1.0,10,5,2,0\n2.0,20,9,3,1\n")
        fuzzer = types.SimpleNamespace(
            op_counts={"bit_flip": 100, "havoc": 50},
            op_success={"bit_flip": 10, "havoc": 0},
            exec_count=20,
            crash_count=1,
            corpus=[b"a", b"b", b"c"],
        )
        generate_html_report(fuzzer, log, tmp_path / "out.html")
        content = (tmp_path / "out.html").read_text()
        assert "bit_flip" in content
        assert "samples logged: 2" in content

    def test_creates_parent_directories(self, tmp_path):
        fuzzer = types.SimpleNamespace(
            op_counts={}, op_success={}, exec_count=0, crash_count=0, corpus=[]
        )
        nested = tmp_path / "a" / "b" / "report.html"
        generate_html_report(fuzzer, tmp_path / "missing.csv", nested)
        assert nested.exists()

    def test_handles_missing_fuzzer_attrs_gracefully(self, tmp_path):
        # A bare object with none of the expected attributes should not crash;
        # getattr defaults should cover it.
        fuzzer = types.SimpleNamespace()
        generate_html_report(fuzzer, tmp_path / "missing.csv", tmp_path / "out.html")
        assert (tmp_path / "out.html").exists()
