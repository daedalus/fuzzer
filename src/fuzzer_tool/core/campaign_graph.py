"""Render a fuzzing campaign's running metrics to a single multi-section PNG.

Fed by StatsReporter._record_graph_snapshot(), which appends one row per
print_stats() tick to Fuzzer._campaign_graph_history whenever --output-graph
is set (see cli/commands.py). This module only turns that history into a
figure; it never touches the fuzzer loop itself.

Optional dependency (matplotlib), following the same pattern already used
for --graph in cmd_ppmd_stats: import lazily, degrade to a clear warning
rather than a traceback if it isn't installed.
"""

import logging

log = logging.getLogger(__name__)


def render_campaign_graph(history: list[dict], output_path: str, target_label: str = "") -> bool:
    """Render `history` (list of per-tick metric dicts) to `output_path`.

    Returns True on success, False if matplotlib isn't available or the
    history is empty -- either way this must never raise into the caller,
    since it runs during teardown after the campaign itself already
    finished.
    """
    if not history:
        print("[*] --output-graph: no stats ticks were recorded, skipping graph")
        return False

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  Warning: matplotlib not installed, skipping --output-graph")
        return False

    try:
        elapsed = [row["elapsed"] for row in history]
        execs = [row["execs"] for row in history]
        eps = [row["eps"] for row in history]
        eps_filtered = [row["eps_filtered"] for row in history]
        corpus = [row["corpus"] for row in history]
        edges = [row["edges"] for row in history if row["edges"] is not None]
        edges_elapsed = [row["elapsed"] for row in history if row["edges"] is not None]
        crashes = [row["crashes"] for row in history]
        crash_sigs = [row["crash_sigs"] for row in history]
        timeouts = [row["timeouts"] for row in history]
        rss_mb = [row["peak_rss_kb"] / 1024 for row in history]
        novel = [row["novel_inputs"] for row in history]

        fig, axes = plt.subplots(4, 1, figsize=(11, 14), sharex=True)
        title = f"Fuzzing campaign — {target_label}" if target_label else "Fuzzing campaign"
        fig.suptitle(f"{title} ({len(history)} ticks, {elapsed[-1]:.0f}s elapsed)", fontsize=14)

        # Section 1: throughput
        ax = axes[0]
        ax.plot(elapsed, eps, color="#90A4AE", linewidth=1, alpha=0.6, label="eps (raw)")
        ax.plot(elapsed, eps_filtered, color="#2196F3", linewidth=2, label="eps (filtered)")
        ax.set_ylabel("execs / sec")
        ax.set_title("Throughput", fontsize=11, loc="left")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax_execs = ax.twinx()
        ax_execs.plot(elapsed, execs, color="#455A64", linewidth=1, linestyle="--", alpha=0.5)
        ax_execs.set_ylabel("cumulative execs", color="#455A64")

        # Section 2: coverage
        ax = axes[1]
        ax.plot(elapsed, corpus, color="#4CAF50", linewidth=2, label="corpus size")
        ax.plot(elapsed, novel, color="#8BC34A", linewidth=1, alpha=0.6, label="novel inputs")
        ax.set_ylabel("seeds", color="#4CAF50")
        ax.set_title("Coverage & corpus growth", fontsize=11, loc="left")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)
        if edges:
            ax_edges = ax.twinx()
            ax_edges.plot(edges_elapsed, edges, color="#FF9800", linewidth=2, label="edges")
            ax_edges.set_ylabel("cumulative edges", color="#FF9800")

        # Section 3: crashes & timeouts
        ax = axes[2]
        ax.step(elapsed, crashes, where="post", color="#F44336", linewidth=2, label="crashes")
        ax.step(
            elapsed,
            crash_sigs,
            where="post",
            color="#C62828",
            linewidth=1,
            linestyle="--",
            label="unique signatures",
        )
        ax.step(
            elapsed, timeouts, where="post", color="#9C27B0", linewidth=1, alpha=0.7,
            label="timeouts",
        )
        ax.set_ylabel("count")
        ax.set_title("Crashes & timeouts", fontsize=11, loc="left")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

        # Section 4: memory
        ax = axes[3]
        ax.plot(elapsed, rss_mb, color="#607D8B", linewidth=2, label="peak RSS")
        ax.set_ylabel("MB")
        ax.set_xlabel("elapsed (s)")
        ax.set_title("Memory", fontsize=11, loc="left")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout(rect=(0, 0, 1, 0.97))
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"[*] Campaign graph saved to: {output_path}")
        return True
    except Exception as e:  # pragma: no cover - rendering must not crash teardown
        log.debug("failed to render campaign graph", exc_info=True)
        print(f"\n  Error generating --output-graph: {e}")
        return False
