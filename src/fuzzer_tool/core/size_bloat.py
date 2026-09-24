"""Corpus bloat verdict from recent seed sizes: location, not shape.

Why not skewness
----------------
The previous gate was ``skewness > 2.0`` over the last 200 seed sizes.
Skewness is scale-free and describes the *shape* of a heavy-tailed size
distribution, not whether sizes are growing, and it gets both directions
wrong on real corpora:

* Seed sizes are close to lognormal (Phi on log-size is near exact, see
  ``core/gaussian.py``). A lognormal's population skewness passes 2 once
  the log-size stddev passes ~0.6, and the sample skewness of a window is
  governed by its few largest members (for tail index < 3 it does not even
  converge -- the generalized CLT regime). Replaying the old gate on
  *stationary* lognormal sizes, i.e. no growth at all, it fired on 15% of
  checks at log-sigma 0.5, 95% at 1.0 and 99.9% at 1.5: a minimisation
  every 500 execs for the whole run.
* Sizes drifting up until they pin at a fixed ``max_len`` pile mass against
  the cap and drive skewness *negative*: 0% of checks fired. A real
  png_read campaign with log-sigma 1.5 and most seeds near the 4096 cap
  read skewness between -0.29 and +0.05 and never fired.

What this measures instead
--------------------------
Two location signals, either of which is bloat:

* ``growth``: the median of the newest ``SEGMENT`` additions is at least
  ``GROWTH_FACTOR`` times the median of the oldest ``SEGMENT`` in the
  window (up to ``WINDOW`` additions apart). Medians, because a mean of
  lognormal sizes is itself dominated by its largest members. With
  log-sigma 1.5 and 100 seeds per segment a doubling is ~2.6 standard
  errors of the log-median difference, so a stationary corpus rarely
  crosses it (see tests/test_size_bloat.py for the measured rates). This is
  the signal that matters under the adaptive ``max_len``
  (corpus_manager: ``2 * p90``, floored at the configured value), which
  moves the cap away from seeds that grow into it.
* ``at_cap``: at least ``AT_CAP_SHARE`` of the newest ``SEGMENT`` sit at or
  above ``CAP_FRACTION * max_len``. The adaptive cap normally prevents
  this; it happens when growth cannot be absorbed -- the 65536 ceiling --
  or when ``max_len`` is fixed.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence

#: Most recent corpus additions the verdict looks at: everything
#: corpus_manager retains in ``_corpus_size_history`` (it trims 1000 -> 500),
#: so the two compared segments sit 500 to 1000 additions apart once the
#: history has filled.
WINDOW = 1000

#: Additions per compared segment (newest vs oldest in the window).
SEGMENT = 100

#: Additions before a verdict is attempted: two disjoint segments.
MIN_SAMPLES = 2 * SEGMENT

#: A seed counts as "at the cap" at or above this fraction of max_len.
CAP_FRACTION = 0.9

#: Share of the newest segment at the cap that counts as bloat.
AT_CAP_SHARE = 0.5

#: Newest-segment median over oldest-segment median that counts as bloat.
GROWTH_FACTOR = 2.0


def seed_size_bloat(sizes: Sequence[int], max_len: int) -> str | None:
    """Return a short reason string when *sizes* show bloat, else None.

    *sizes* are corpus additions in insertion order; only the last
    ``WINDOW`` are read.
    """
    sizes = list(sizes)[-WINDOW:]
    n = len(sizes)
    if n < MIN_SAMPLES:
        return None

    newest = sizes[-SEGMENT:]
    if max_len > 0:
        cap = CAP_FRACTION * max_len
        at_cap = sum(1 for s in newest if s >= cap)
        if at_cap >= AT_CAP_SHARE * SEGMENT:
            return (
                f"{at_cap}/{SEGMENT} newest seeds at >= {CAP_FRACTION:.0%} of max_len ({max_len}B)"
            )

    older = statistics.median(sizes[:SEGMENT])
    newer = statistics.median(newest)
    if older > 0 and newer >= GROWTH_FACTOR * older:
        return f"median seed size {older:.0f}B -> {newer:.0f}B across the last {n} additions"

    return None
