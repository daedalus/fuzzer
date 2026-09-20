"""Per-site phase offsets for the fuzzer's periodic subsystems.

Nineteen subsystems do periodic work keyed on a counter — CRPS sampling
every 8 executions, the Monte-Carlo draw refresh every 16, state snapshots
every 20, crash-eta / Katz / Markov every 50, five separate things every
100, a GARCH refit at 128, the seed-picker weight recompute at 200, corpus
history at 500, three at 1000 and two at 2000.

Measured over 20,000 executions: of the ten distinct periods, **zero of the
forty-five pairs are coprime**, and the co-firing distribution is bimodal —
nothing between four and nine subsystems firing together, then spikes of
eighteen and nineteen.  A weight recompute alone is ~11.6 ms against ~0.8 ms
for an ffmpeg execution, so those iterations cost order 100x a normal one.

The periods are not the problem and this module does not touch them.  Each
subsystem picked its rate for a reason, and retuning nineteen of them to be
pairwise coprime would change nineteen behaviours to fix a scheduling
artifact.  What is wrong is that they all measure their period from the same
origin, so their firings pile onto the same counters.  Giving each site a
fixed offset within its own period leaves every rate exactly as it was and
only moves *when* in the cycle the work lands::

    period 100, three sites, bare modulo        phase-offset
    c: 0 . . . 100 . . . 200        c: 43 . 71 . . 6 . 143 . 171 . 206
       ^^^ all three                   ^    ^      ^   spread
              three at once

The offset is derived from the site name with CRC-32, not from ``hash()``:
the builtin string hash is salted per interpreter run, so a resumed campaign
would fire on a different schedule than the one it saved.

Use :func:`due` for ``counter % period == 0`` gates and :func:`bucket` for
cache keys of the ``counter // period`` shape.  The two stay consistent —
``bucket`` advances on exactly the counters where ``due`` is true — so a
cache and the recompute that fills it can be gated on either.
"""

from __future__ import annotations

import zlib

# Cache of site -> CRC, so the digest is computed once per call site rather
# than on every execution of the hot loop.
_SITE_CRC: dict[str, int] = {}


def _crc(site: str) -> int:
    crc = _SITE_CRC.get(site)
    if crc is None:
        crc = zlib.crc32(site.encode())
        _SITE_CRC[site] = crc

    return crc


def phase_of(site: str, period: int) -> int:
    """Stable offset in ``[0, period)`` for a named call site.

    Deterministic across processes, runs and resumes.

    Raises:
        ValueError: if *period* is not positive.
    """
    if period <= 0:
        raise ValueError(f"period must be positive, got {period}")

    return _crc(site) % period


def due(counter: int, period: int, site: str) -> bool:
    """True on one counter in every *period*, offset by the site's phase.

    Drop-in for ``counter % period == 0``. The rate is identical; only the
    firing instants move.
    """
    return (counter + phase_of(site, period)) % period == 0


def bucket(counter: int, period: int, site: str) -> int:
    """Index of the period *counter* falls in, offset by the site's phase.

    Drop-in for ``counter // period`` as a cache key. Non-negative for
    ``counter >= 0``, so the bucket covering execution 0 cannot collide with
    the one after the first wrap.
    """
    return (counter + phase_of(site, period)) // period
