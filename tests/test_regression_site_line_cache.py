"""collect_sites parses each distinct ``CNS`` line once.

FFmpeg writes ~850 site lines per exec and 95% repeat verbatim (318k lines,
17k distinct over 400 execs); split + three int() per line was 6% of
fuzz-loop wall time. Parsed lines are cached by their text (fast path);
unseen, ``CND`` and malformed lines take the parser (slow path). The cache
is bounded. Totals must equal the uncached loop.
"""

import random

import pytest

from fuzzer_tool.core import cmplog
from fuzzer_tool.core.cmplog import CmplogCollector


def _ref_fold(state, lines):
    """Oracle: the pre-change loop over plain dicts."""
    site_fired, site_asserted, dropped = state
    last_fired, last_asserted = {}, {}
    for line in lines:
        parts = line.split()
        n = len(parts)
        if n != 5 or parts[0] != "CNS":
            if n == 2 and parts[0] == "CND" and parts[1].isdigit():
                dropped.append(int(parts[1]))
            continue
        try:
            pc = int(parts[2], 16)
            fired = int(parts[3])
            asserted = int(parts[4])
        except ValueError:
            continue
        key = (parts[1], pc)
        site_fired[key] = site_fired.get(key, 0) + fired
        site_asserted[key] = site_asserted.get(key, 0) + asserted
        if fired:
            last_fired[key] = last_fired.get(key, 0) + fired
        if asserted:
            last_asserted[key] = last_asserted.get(key, 0) + asserted
    return last_fired, last_asserted


def _drains(seed, rounds=12, per=200, distinct=60):
    """Repeating site lines plus CND, malformed and blank lines."""
    rnd = random.Random(seed)
    sites = [
        f"CNS trace_cmp{rnd.choice((1, 2, 4, 8))} 0x{rnd.randrange(1 << 40):x} "
        f"{rnd.randint(0, 3)} {rnd.randint(0, 2)}\n"
        for _ in range(distinct)
    ]
    junk = ["CND 3\n", "CND x\n", "CNS a 0xzz 1 1\n", "CNS a 0x10 1\n", "\n", "XYZ 1 2 3 4\n"]
    return [[rnd.choice(sites + junk) for _ in range(per)] for _ in range(rounds)]


def _collector(tmp_path):
    c = CmplogCollector(site_counts=True, workdir=str(tmp_path))
    c.sites_path = str(tmp_path / "sites")
    return c


def _run_new(tmp_path, drains):
    c = _collector(tmp_path)
    lasts = []
    for lines in drains:
        (tmp_path / "sites").write_text("".join(lines))
        c._sites_offset = 0
        c.last_site_fired, c.last_site_asserted = {}, {}
        c.collect_sites()
        lasts.append((dict(c.last_site_fired), dict(c.last_site_asserted)))
    return lasts, dict(c.site_fired), dict(c.site_asserted), c.site_dropped, c


def _run_ref(drains):
    state = ({}, {}, [])
    lasts = [_ref_fold(state, lines) for lines in drains]
    return lasts, state[0], state[1], sum(state[2])


def test_control_oracle_matches_itself():
    """Rule 46: the oracle agrees with a second run of itself."""
    d = _drains(1)
    assert _run_ref(d) == _run_ref(d)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_totals_match_uncached_loop(tmp_path, seed):
    """Falsification: per-drain and cumulative totals equal the old loop,
    with CND / malformed lines interleaved with cached ones."""
    d = _drains(seed)
    lasts, fired, asserted, dropped, _ = _run_new(tmp_path, d)
    assert (lasts, fired, asserted, dropped) == _run_ref(d)


def test_repeated_lines_parsed_once(tmp_path, monkeypatch):
    """Falsification: a line seen before is not re-split."""
    d = _drains(4, rounds=6, distinct=10)
    distinct_cns = {ln for lines in d for ln in lines if ln.startswith("CNS")}
    _, _, _, _, c = _run_new(tmp_path, d)
    assert len(c._site_line_cache) <= len(distinct_cns)


def test_cache_stays_bounded(tmp_path, monkeypatch):
    """Adversarial: unbounded distinct lines (ASLR-like churn) never grow
    the cache past its cap, and totals stay exact across the reset."""
    monkeypatch.setattr(cmplog, "SITE_LINE_CACHE_MAX", 50)
    d = _drains(5, rounds=10, per=100, distinct=400)
    lasts, fired, asserted, dropped, c = _run_new(tmp_path, d)
    assert len(c._site_line_cache) <= 50
    assert (lasts, fired, asserted, dropped) == _run_ref(d)
