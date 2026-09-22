"""Live synchronization-transition monitor over `OpKuramotoScheduler`.

`core/kuramoto.py`'s module docstring frames the Kuramoto incoherent-to-
synchronized transition as a genuine second-order phase transition with
"the same critical-slowing-down precursor (rising variance/autocorrelation
just below K_c) that `analyzer_critical_slowing.py` already looks for in
the discovery-rate series." `core/schedulers/op_kuramoto.py` (which wires
that diagnostic machinery -- `order_parameter`, `spectral_radius`,
`critical_coupling` -- against its own live operator-transition graph via
`diagnostics()`) explicitly leaves that cross-check for later: its
docstring calls `diagnostics()` "exposed for the A/B run ... still
outstanding, never consumed by `select_op` itself."

This module is that follow-up, and nothing more: it samples
`OpKuramotoScheduler.diagnostics()` over time and reuses
`CriticalSlowingDown` -- unmodified -- to watch the order parameter r(t)
for the same rising-variance/autocorrelation precursor already trusted for
the discovery-rate series. Read-only. It does not feed back into
`select_op` or any other scheduling decision, mirroring `diagnostics()`'s
own discipline: this is an observation surface, not a controller.

See docs/handover/handover_kuramoto_sync_monitor_2026-09-21.md.
"""

from __future__ import annotations

from fuzzer_tool.core.analyzers.analyzer_critical_slowing import CriticalSlowingDown


class KuramotoSyncMonitor:
    """Track `OpKuramotoScheduler.diagnostics()` snapshots over time and
    flag critical slowing down in the order parameter r(t).

    Args:
        window_size: Passed through to the internal `CriticalSlowingDown`.
        rise_threshold: Passed through to the internal `CriticalSlowingDown`.
        min_observations: Passed through to the internal
            `CriticalSlowingDown`.
    """

    def __init__(
        self,
        window_size: int = 50,
        rise_threshold: float = 1.5,
        min_observations: int = 20,
    ):
        self.window_size = window_size
        self.rise_threshold = rise_threshold
        self.min_observations = min_observations
        self._csd = CriticalSlowingDown(
            window_size=window_size,
            rise_threshold=rise_threshold,
            min_observations=min_observations,
        )
        self._last_r: float = 0.0
        self._last_psi: float = 0.0
        self._last_critical_coupling: float = float("inf")
        self._last_spectral_radius: float = 0.0
        self._last_n_arms: int = 0
        self.n_observations: int = 0

    def observe(self, diagnostics: dict) -> None:
        """Ingest one `OpKuramotoScheduler.diagnostics()` snapshot.

        Args:
            diagnostics: The dict returned by
                `OpKuramotoScheduler.diagnostics()` -- expects at least
                `"r"`; `"psi"`, `"critical_coupling"`, `"spectral_radius"`,
                and `"n_arms"` are recorded when present but not required,
                so a partial or future-shaped dict degrades gracefully
                rather than raising.
        """
        self._last_r = float(diagnostics.get("r", 0.0))
        self._last_psi = float(diagnostics.get("psi", 0.0))
        self._last_critical_coupling = float(diagnostics.get("critical_coupling", float("inf")))
        self._last_spectral_radius = float(diagnostics.get("spectral_radius", 0.0))
        self._last_n_arms = int(diagnostics.get("n_arms", 0))
        self.n_observations += 1
        self._csd.observe(self._last_r)

    def status(self) -> tuple[bool, str]:
        """Check whether r(t) shows critical slowing down.

        Returns:
            Tuple of (detected, reason_string) -- same contract as
            `CriticalSlowingDown.is_approaching_transition()`, applied to
            the order-parameter series instead of the discovery rate.
        """
        if self._last_n_arms < 2:
            return False, f"need >=2 arms (have {self._last_n_arms})"
        return self._csd.is_approaching_transition()

    @property
    def last_r(self) -> float:
        return self._last_r

    @property
    def last_critical_coupling(self) -> float:
        return self._last_critical_coupling

    @property
    def last_spectral_radius(self) -> float:
        return self._last_spectral_radius

    def reset(self) -> None:
        """Reset monitor state."""
        self._csd.reset()
        self._last_r = 0.0
        self._last_psi = 0.0
        self._last_critical_coupling = float("inf")
        self._last_spectral_radius = 0.0
        self._last_n_arms = 0
        self.n_observations = 0

    def save(self) -> dict:
        """Serialize state."""
        return {
            "csd": self._csd.save(),
            "last_r": self._last_r,
            "last_psi": self._last_psi,
            "last_critical_coupling": self._last_critical_coupling,
            "last_spectral_radius": self._last_spectral_radius,
            "last_n_arms": self._last_n_arms,
            "n_observations": self.n_observations,
            "window_size": self.window_size,
            "rise_threshold": self.rise_threshold,
            "min_observations": self.min_observations,
        }

    def load(self, data: dict) -> None:
        """Restore state."""
        if "csd" in data:
            self._csd.load(data["csd"])
        self._last_r = data.get("last_r", 0.0)
        self._last_psi = data.get("last_psi", 0.0)
        self._last_critical_coupling = data.get("last_critical_coupling", float("inf"))
        self._last_spectral_radius = data.get("last_spectral_radius", 0.0)
        self._last_n_arms = data.get("last_n_arms", 0)
        self.n_observations = data.get("n_observations", 0)
        self.window_size = data.get("window_size", self.window_size)
        self.rise_threshold = data.get("rise_threshold", self.rise_threshold)
        self.min_observations = data.get("min_observations", self.min_observations)
