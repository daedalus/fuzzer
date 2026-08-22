"""SVG chart generation and self-contained HTML report for fuzzer runs."""

import logging
from html import escape
from pathlib import Path

log = logging.getLogger(__name__)


def _esc(value) -> str:
    """HTML-escape a report string, tolerating non-str values."""
    return escape(str(value))


def read_coverage_log(path):
    """Read coverage log CSV: elapsed,exec_count,cumulative_edges,corpus_size,crash_count."""
    rows = []
    p = Path(path)
    if not p.exists():
        log.debug("read_coverage_log: file missing %s", path)
        return rows
    with p.open() as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 5:
                continue
            try:
                rows.append(
                    {
                        "elapsed": float(parts[0]),
                        "exec_count": int(parts[1]),
                        "cumulative_edges": int(parts[2]),
                        "corpus_size": int(parts[3]),
                        "crash_count": int(parts[4]),
                    }
                )
            except (ValueError, IndexError):
                continue
    log.debug("read_coverage_log: parsed %d rows from %s", len(rows), path)
    return rows


def _derive_exec_rate(rows):
    """Compute (elapsed, execs/sec) between consecutive rows."""
    log.debug("_derive_exec_rate: input rows=%d", len(rows))
    points = []
    for i in range(1, len(rows)):
        dt = rows[i]["elapsed"] - rows[i - 1]["elapsed"]
        de = rows[i]["exec_count"] - rows[i - 1]["exec_count"]
        if dt > 0 and de >= 0:
            points.append((rows[i]["elapsed"], de / dt))
    log.debug("_derive_exec_rate: produced %d points", len(points))
    return points


def _scale(value, in_lo, in_hi, out_lo, out_hi):
    """Linear interpolation from [in_lo, in_hi] to [out_lo, out_hi]."""
    if in_hi == in_lo:
        return (out_lo + out_hi) / 2
    return out_lo + (value - in_lo) / (in_hi - in_lo) * (out_hi - out_lo)


def _svg_line_chart(title, x_label, y_label, points, width=700, height=200):
    """Render an inline SVG line chart. points = [(x, y), ...]."""
    log.debug("_svg_line_chart: title=%r points=%d", title, len(points))
    pad_l, pad_r, pad_t, pad_b = 60, 20, 30, 40
    iw = width - pad_l - pad_r
    ih = height - pad_t - pad_b

    if not points:
        return (
            f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">\n'
            f'  <text x="{width / 2}" y="{height / 2}" text-anchor="middle" '
            f'fill="#888" font-size="14">{escape(title)}: no data yet</text>\n</svg>'
        )

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_lo, x_hi = min(xs), max(xs)
    y_lo, y_hi = 0, max(ys) * 1.1 if max(ys) > 0 else 1

    coords = []
    for x, y in points:
        cx = pad_l + _scale(x, x_lo, x_hi, 0, iw)
        cy = pad_t + ih - _scale(y, y_lo, y_hi, 0, ih)
        coords.append(f"{cx:.1f},{cy:.1f}")
    path_d = "M" + " L".join(coords)

    return (
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">\n'
        f'  <text x="{width / 2}" y="18" text-anchor="middle" font-size="14" '
        f'font-weight="bold">{escape(title)}</text>\n'
        f'  <path d="{path_d}" fill="none" stroke="#2563eb" stroke-width="2"/>\n'
        f'  <text x="{pad_l - 5}" y="{pad_t + ih + 20}" font-size="11" '
        f'text-anchor="middle">{escape(x_label)}</text>\n'
        f'  <text x="12" y="{pad_t + ih / 2}" font-size="11" text-anchor="middle" '
        f'transform="rotate(-90,12,{pad_t + ih / 2})">{escape(y_label)}</text>\n'
        f'  <line x1="{pad_l}" y1="{pad_t + ih}" x2="{pad_l + iw}" '
        f'y2="{pad_t + ih}" stroke="#ccc" stroke-width="1"/>\n'
        f'  <line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" '
        f'y2="{pad_t + ih}" stroke="#ccc" stroke-width="1"/>\n'
        f"</svg>"
    )


def _svg_bar_chart(title, bars, width=700, height=200):
    """Render an inline SVG horizontal bar chart. bars = [(label, value), ...]."""
    log.debug("_svg_bar_chart: title=%r bars=%d", title, len(bars))
    pad_l, pad_r, pad_t, pad_b = 120, 60, 30, 20
    iw = width - pad_l - pad_r
    ih = height - pad_t - pad_b

    if not bars:
        return (
            f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">\n'
            f'  <text x="{width / 2}" y="{height / 2}" text-anchor="middle" '
            f'fill="#888" font-size="14">no data yet</text>\n</svg>'
        )

    max_val = max((v for _, v in bars), default=1) or 1
    bar_h = min(ih / len(bars) * 0.7, 18)
    gap = ih / len(bars) * 0.3

    rects = []
    for idx, (label, val) in enumerate(bars):
        y = pad_t + idx * (bar_h + gap)
        bw = _scale(val, 0, max_val, 0, iw)
        pct = val * 100
        rects.append(
            f'  <rect x="{pad_l}" y="{y:.1f}" width="{bw:.1f}" height="{bar_h:.1f}" '
            f'fill="#2563eb"/>\n'
            f'  <text x="{pad_l - 5}" y="{y + bar_h / 2 + 4:.1f}" text-anchor="end" '
            f'font-size="11">{escape(label)}</text>\n'
            f'  <text x="{pad_l + bw + 5:.1f}" y="{y + bar_h / 2 + 4:.1f}" '
            f'font-size="11">{pct:.1f}%</text>'
        )

    return (
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">\n'
        f'  <text x="{width / 2}" y="18" text-anchor="middle" font-size="14" '
        f'font-weight="bold">{escape(title)}</text>\n' + "\n".join(rects) + "\n</svg>"
    )


def generate_html_report(fuzzer, coverage_log_path, output_path):
    """Generate a self-contained HTML report with SVG charts."""
    log.debug("generate_html_report: start path=%s output=%s", coverage_log_path, output_path)
    rows = read_coverage_log(coverage_log_path)

    # Exec rate
    rate_points = _derive_exec_rate(rows)

    # Extract series
    edge_points = [(r["elapsed"], r["cumulative_edges"]) for r in rows]
    corpus_points = [(r["elapsed"], r["corpus_size"]) for r in rows]
    crash_points = [(r["elapsed"], r["crash_count"]) for r in rows]

    # Operator success rates
    op_counts = getattr(fuzzer, "op_counts", {})
    op_success = getattr(fuzzer, "op_success", {})
    op_bars = []
    for op in sorted(op_counts, key=op_counts.get, reverse=True)[:15]:
        count = op_counts[op]
        success = op_success.get(op, 0)
        rate = success / count if count > 0 else 0
        op_bars.append((op, rate))

    # Operator usage counts (bar chart of raw counts)
    op_usage_bars = []
    if op_counts:
        total_ops = sum(op_counts.values()) or 1
        for op in sorted(op_counts, key=op_counts.get, reverse=True)[:15]:
            op_usage_bars.append((op, op_counts[op] / total_ops))

    # Build HTML
    charts = []
    if edge_points:
        charts.append(_svg_line_chart("Edges Discovered", "time (s)", "edges", edge_points))
    if rate_points:
        charts.append(_svg_line_chart("Execution Rate", "time (s)", "execs/sec", rate_points))
    if corpus_points:
        charts.append(_svg_line_chart("Corpus Size", "time (s)", "seeds", corpus_points))
    if crash_points:
        charts.append(_svg_line_chart("Crashes", "time (s)", "crashes", crash_points))
    if op_bars:
        charts.append(_svg_bar_chart("Operator Success Rate", op_bars))
    if op_usage_bars:
        charts.append(_svg_bar_chart("Operator Usage", op_usage_bars))

    exec_count = getattr(fuzzer, "exec_count", 0)
    crash_count = getattr(fuzzer, "crash_count", 0)
    corpus_size = len(getattr(fuzzer, "corpus", []))

    target = getattr(fuzzer, "target", "unknown")
    target_name = str(target).rsplit("/", 1)[-1]

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Fuzzer Report — {_esc(target_name)}</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 800px; margin: 2em auto; padding: 0 1em; }}
h1 {{ border-bottom: 2px solid #2563eb; padding-bottom: 0.3em; }}
.summary {{ background: #f8fafc; padding: 1em; border-radius: 8px; margin: 1em 0; }}
.details {{ background: #f8fafc; padding: 1em; border-radius: 8px; margin: 1em 0; }}
.details code {{ display: block; white-space: pre-wrap; word-break: break-all; font-size: 0.9em; }}
.chart {{ margin: 1.5em 0; }}
svg {{ border: 1px solid #e2e8f0; border-radius: 4px; }}
footer {{ color: #888; font-size: 0.85em; margin-top: 2em; }}
</style></head><body>
<h1>Fuzzer Report — {_esc(target_name)}</h1>
<div class="summary">
  <b>Executions:</b> {exec_count:,} &nbsp;|&nbsp;
  <b>Crashes:</b> {crash_count} &nbsp;|&nbsp;
  <b>Corpus:</b> {corpus_size} &nbsp;|&nbsp;
  samples logged: {len(rows)}
</div>
<div class="details">
"""
    exec_line = getattr(fuzzer, "exec_line", None)
    if exec_line is None:
        args = list(getattr(fuzzer, "target_args", None) or [])
        if getattr(fuzzer, "file_mode", False):
            parts = [str(target)] + ([a.replace("{file}", "@@") for a in args] if args else ["@@"])
        else:
            parts = [str(target)] + args
        exec_line = " ".join(parts)
    html += f"  <b>Target:</b> {_esc(target)}<br>\n"
    html += f"  <b>Exec line:</b> <code>{_esc(exec_line)}</code><br>\n"
    inv = getattr(fuzzer, "invocation", "")
    if inv:
        html += f"  <b>Invocation:</b> <code>{_esc(inv)}</code><br>\n"
    orig = getattr(fuzzer, "original_invocation", "")
    if orig and orig != inv:
        html += f"  <b>Started as:</b> <code>{_esc(orig)}</code><br>\n"
    mode = "file" if getattr(fuzzer, "file_mode", False) else "stdin"
    html += f"  <b>Input mode:</b> {_esc(mode)}<br>\n"
    cov_mode = (
        "SHM bitmap"
        if getattr(fuzzer, "shm_cov", None)
        else "ptrace"
        if getattr(fuzzer, "ptrace_cov", None)
        else "none"
    )
    html += f"  <b>Coverage:</b> {_esc(cov_mode)} &nbsp;|&nbsp; "
    html += f"<b>Max len:</b> {getattr(fuzzer, 'max_len', 0)} &nbsp;|&nbsp; "
    html += f"<b>Timeout:</b> {getattr(fuzzer, 'timeout', 0)}s<br>\n"
    cmplog = getattr(fuzzer, "_cmplog", None)
    html += f"  <b>Cmplog:</b> {'enabled' if cmplog is not None else 'disabled'}<br>\n"
    html += "</div>\n"
    for chart in charts:
        html += f'<div class="chart">{chart}</div>\n'

    html += "<footer>Generated by fuzzer-tool</footer>\n</body></html>"

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    log.debug("generate_html_report: wrote %s (%d bytes)", out, len(html))
    return str(out)
