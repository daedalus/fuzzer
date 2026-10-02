"""Per-target cross-campaign ledger: preset choice and history priors.

Port of Stardust's ``Opponent::selectOpeningUCB1`` and
``Opponent::minValueInPreviousGames`` (github.com/bmnielsen/Stardust,
``src/Opponent.cpp``, MIT). Opponent -> target, opening -> config preset,
game result -> campaign score, map -> target build.

  campaign N                              campaign N+1
  ──────────                              ────────────
  select(presets, build) ──► preset        select() sees N's record
  run ... record(preset, edges, build,     min_value("max_gap") ──► prior
                 {"max_gap": G})
  save() ──► ~/fuzzing/<target>/ledger/state.pkl.gz

Differences from Stardust: scores are continuous (normalised by the best in
history) instead of win/loss, and "never lost" becomes "fewer than
``min_trials`` campaigns", which is safe under noisy outcomes.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from fuzzer_tool.core.state_store import StateStore

LEDGER_SECTION = "campaign_ledger"
LEDGER_VERSION = 1

DEFAULT_MAX_RECORDS = 256
DEFAULT_DECAY = 0.05
DEFAULT_MIN_TRIALS = 1
SAME_BUILD_WEIGHT = 2.0

# Record fields; everything else in a record is a caller-supplied history value.
_PRESET = "preset"
_SCORE = "score"
_BUILD = "build"
_RESERVED = frozenset((_PRESET, _SCORE, _BUILD))

# One campaign: preset/score/build plus caller-supplied int history values.
Record = dict[str, Any]


def _clean(rec: Any) -> Record | None:
    """Return a sanitised record, or None for junk (tampered or older shapes)."""
    if not isinstance(rec, dict):
        return None

    preset, score, build = rec.get(_PRESET), rec.get(_SCORE), rec.get(_BUILD)
    if not isinstance(preset, str) or not isinstance(build, str):
        return None
    if isinstance(score, bool) or not isinstance(score, int | float):
        return None
    if not math.isfinite(score):
        return None

    out: Record = {k: v for k, v in rec.items() if k not in _RESERVED and isinstance(v, int)}
    out.update({_PRESET: preset, _SCORE: max(0.0, float(score)), _BUILD: build})
    return out


class CampaignLedger:
    """Bounded per-target history of campaign outcomes, persisted as a pickle."""

    def __init__(self, ledger_dir: str | Path, max_records: int = DEFAULT_MAX_RECORDS):
        self._store = StateStore(ledger_dir)
        self._max = max_records
        self._records: list[Record] = []

    def __len__(self) -> int:
        return len(self._records)

    # ── persistence ──────────────────────────────────────────────────

    def load(self) -> None:
        """Read history; junk rows are dropped, a junk section reads as empty."""
        section = self._store.get(LEDGER_SECTION)
        rows = section.get("records") if isinstance(section, dict) else None
        if not isinstance(rows, list):
            rows = []

        cleaned = (_clean(r) for r in rows)
        self._records = [r for r in cleaned if r is not None][-self._max :]

    def save(self) -> bool:
        """Write history atomically (StateStore temp-file + rename)."""
        self._store.set(LEDGER_SECTION, {"version": LEDGER_VERSION, "records": self._records})
        return self._store.save()

    def record(
        self, preset: str, score: float, build: str, values: dict[str, int] | None = None
    ) -> None:
        """Append one campaign's outcome; the oldest record falls off past the cap."""
        rec = _clean({**(values or {}), _PRESET: preset, _SCORE: score, _BUILD: build})
        if rec is None:
            raise ValueError(f"invalid ledger record: {preset!r} {score!r} {build!r}")

        self._records.append(rec)
        del self._records[: -self._max]

    # ── item 1: preset choice ────────────────────────────────────────

    def _weigh(
        self, presets: list[str], build: str, decay: float
    ) -> tuple[dict[str, int], dict[str, float], dict[str, float]]:
        """Decayed, build-weighted (trials, reward, potential) per offered preset.

        Newest record has age 0 and weight exp(-decay); same build counts x2.
        Scores are normalised by the best score in history, so reward is in [0, 1].
        """
        best = max((r[_SCORE] for r in self._records), default=0.0) or 1.0
        trials = dict.fromkeys(presets, 0)
        reward = dict.fromkeys(presets, 0.0)
        potential = dict.fromkeys(presets, 0.0)

        for age, rec in enumerate(reversed(self._records)):
            name = rec[_PRESET]
            if name not in trials:
                continue

            w = math.exp(-decay * (age + 1))
            if rec[_BUILD] == build:
                w *= SAME_BUILD_WEIGHT
            trials[name] += 1
            potential[name] += w
            reward[name] += w * rec[_SCORE] / best

        return trials, reward, potential

    def select(
        self,
        presets: list[str],
        build: str,
        decay: float = DEFAULT_DECAY,
        min_trials: int = DEFAULT_MIN_TRIALS,
    ) -> str:
        """Pick the preset for the next campaign.

        1. Any preset with fewer than ``min_trials`` campaigns: fewest first,
           ties in ballot order (Stardust: untried / never-lost first).
        2. Otherwise UCB1 on decayed potentials:
           ``reward/potential + sqrt(2 ln(max(total, 1)) / potential)``.
        Deterministic: no RNG, ties go to ballot order.
        """
        if not presets:
            raise ValueError("no presets to choose from")
        if len(presets) == 1:
            return presets[0]

        trials, reward, potential = self._weigh(presets, build, decay)

        under = [p for p in presets if trials[p] < min_trials]
        if under:
            return min(under, key=trials.__getitem__)

        # Heavy decay can push total potential below 1; clamp so the
        # exploration term stays real (Stardust gets NaN here).
        log_total = math.log(max(sum(potential.values()), 1.0))
        best, best_score = presets[0], -math.inf
        for p in presets:
            score = reward[p] / potential[p] + math.sqrt(2.0 * log_total / potential[p])
            if score > best_score:
                best, best_score = p, score

        return best

    # ── item 2: history priors ───────────────────────────────────────

    def min_value(self, key: str, default: int, max_count: int, min_count: int) -> int:
        """Minimum of ``key`` over the newest ``max_count`` campaigns.

        A record without ``key`` was written before the value existed and ends
        the scan (Stardust's version boundary). Fewer than ``min_count``
        values found returns ``default``.
        """
        result, count = None, 0
        for rec in reversed(self._records):
            if count >= max_count or key not in rec:
                break

            value = rec[key]
            result = value if result is None else min(result, value)
            count += 1

        if result is None or count < min_count:
            return default
        return int(result)
