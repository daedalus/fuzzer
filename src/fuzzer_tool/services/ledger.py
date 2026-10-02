"""--preset-ledger orchestration: one target's cross-campaign memory.

Binds :class:`~fuzzer_tool.core.campaign_ledger.CampaignLedger` to a target
binary and a finished Fuzzer, so the CLI deals in presets and thresholds only.

  cli main()                          cli cmd_fuzz() end
  ──────────                          ──────────────────
  LedgerSession(target)               LedgerSession(target)
    .choose(presets)  -> preset         .finish(fuzzer, preset)
    .stall_prior(1000) -> --stall         score = cumulative edges
                                          max_gap = longest broken silence
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from fuzzer_tool.core.campaign_ledger import CampaignLedger
from fuzzer_tool.core.cfg_cache import binary_digest

if TYPE_CHECKING:
    from fuzzer_tool.services.fuzzer import Fuzzer

LEDGER_SUBDIR = "ledger"
UNKNOWN_BUILD = "unknown"
BUILD_KEY_LEN = 16

# History value: longest run of execs without a new edge that still ended in one.
MAX_GAP_KEY = "max_gap"

# Campaigns scanned for the stall prior, and the minimum before it replaces
# the flag default (Stardust uses 20 / 0..3 for its timing priors).
HISTORY_WINDOW = 20
MIN_HISTORY = 3

# --stall bounds: below the floor recovery thrashes; above the ceiling it never fires.
STALL_FLOOR = 100
STALL_CEIL = 1_000_000


def ledger_dir(target: str) -> Path:
    """``~/fuzzing/<target>/ledger``, next to the default corpus (see ``_get_dirs``)."""
    return Path.home() / "fuzzing" / Path(target).resolve().name / LEDGER_SUBDIR


def build_key(target: str) -> str:
    """Short build identity; the same binary maps to the same key across runs."""
    try:
        return binary_digest(target)[:BUILD_KEY_LEN]
    except OSError:
        return UNKNOWN_BUILD


class LedgerSession:
    """One target's ledger, loaded once per CLI phase."""

    def __init__(self, target: str, root: Path | None = None):
        self._build = build_key(target)
        self._ledger = CampaignLedger(root if root is not None else ledger_dir(target))
        self._ledger.load()

    @property
    def build(self) -> str:
        return self._build

    def choose(self, presets: list[str]) -> str:
        """Preset for this campaign (UCB1 over past campaigns on this target)."""
        return self._ledger.select(presets, self._build)

    def stall_prior(self, default: int) -> int:
        """Shortest of the recent campaigns' longest broken silences, clamped.

        Every recent campaign broke a silence at least this long without
        help, so stall recovery should not fire before it (and needs not
        wait much past it). Falls back to ``default`` until enough history.
        """
        gap = self._ledger.min_value(MAX_GAP_KEY, default, HISTORY_WINDOW, MIN_HISTORY)
        return min(max(gap, STALL_FLOOR), STALL_CEIL)

    def finish(self, fuzzer: Fuzzer, preset: str) -> bool:
        """Record the campaign; False when it has no comparable edge count."""
        cov = getattr(fuzzer, "shm_cov", None)
        if cov is None:
            return False

        # No discovery means no broken silence: omit the key, which ends the
        # prior's scan, so it falls back to the default instead of the floor.
        gap = int(getattr(fuzzer, "_max_edge_gap", 0))
        values = {MAX_GAP_KEY: gap} if gap > 0 else {}
        self._ledger.record(preset, int(cov.cumulative_edges), self._build, values)
        return self._ledger.save()
