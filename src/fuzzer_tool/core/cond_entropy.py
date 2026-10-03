"""Conditional entropy of coverage given input structure (handover 7.3).

``H(path | context)`` separates input regions that fully determine the path
(magic numbers, checksums: H ~ 0) from regions that still carry entropy, which
is where novel edges are likely. ``context`` is any hashable (``--weizz-tags``
tag, a k-byte prefix, a field name). ``path`` is any hashable signature of the
execution, normally ``frozenset(edges)``.

Estimator: Miller-Madow corrected per-context entropy, weighted by context
frequency. Contexts with fewer than ``min_n`` samples are reported as
``undetermined`` (no evidence), never as determined.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Hashable, Iterable


def _h(counts: Counter, miller_madow: bool) -> float:
    n = sum(counts.values())
    if n == 0:
        return 0.0
    h = -sum((c / n) * math.log2(c / n) for c in counts.values())
    if miller_madow:
        h += (len(counts) - 1) / (2.0 * n * math.log(2))
    return h


def group_by_context(
    pairs: Iterable[tuple[Hashable, Hashable]],
) -> dict[Hashable, Counter]:
    groups: dict[Hashable, Counter] = defaultdict(Counter)
    for ctx, path in pairs:
        groups[ctx][path] += 1
    return dict(groups)


def conditional_entropy(
    pairs: Iterable[tuple[Hashable, Hashable]], miller_madow: bool = True
) -> float:
    """``H(path | context)`` in bits, frequency weighted."""
    groups = group_by_context(pairs)
    total = sum(sum(c.values()) for c in groups.values())
    if total == 0:
        return 0.0
    return sum(sum(c.values()) / total * _h(c, miller_madow) for c in groups.values())


def context_report(
    pairs: Iterable[tuple[Hashable, Hashable]],
    min_n: int = 8,
    determined_below: float = 0.05,
) -> dict[Hashable, dict]:
    """Per-context ``{n, h, status}``; status in determined/open/undetermined.

    ``determined`` needs ``n >= min_n`` and plug-in entropy below
    ``determined_below`` bits (one path dominating). The plug-in value is used
    for the threshold because Miller-Madow adds a floor for every extra path.
    """
    out = {}
    for ctx, counts in group_by_context(pairs).items():
        n = sum(counts.values())
        h = _h(counts, True)
        if n < min_n:
            status = "undetermined"
        elif _h(counts, False) < determined_below:
            status = "determined"
        else:
            status = "open"
        out[ctx] = {"n": n, "h": h, "status": status}
    return out


def energy_weights(report: dict[Hashable, dict], floor: float = 0.05) -> dict:
    """Weight per context: determined regions get ``floor``, others ``1 + h``."""
    return {
        c: (floor if r["status"] == "determined" else 1.0 + r["h"])
        for c, r in report.items()
    }


def prefix_context(data: bytes, k: int = 4) -> bytes:
    return bytes(data[:k])
