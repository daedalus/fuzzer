"""collect_sites folds a drain with numpy instead of per-line dict updates.

Real FFmpeg drains: 1022 lines each, 57k distinct lines over 1.44M, 4037
sites. The text cache still left four dict upserts per line (527 us per
drain). ``_SiteCounters`` maps each cached line to a slot (site id, fired,
asserted) and folds a drain with ``np.add.at``; ``last_site_fired`` (never
read on the hot path) is built on read. 272 us per drain on the same
drains. Every dict must equal the per-line fold.
"""

import random

import pytest

from fuzzer_tool.core import cmplog
from fuzzer_tool.core.cmplog import CmplogCollector
from tests.test_regression_site_line_cache import _ref_fold


def _drains(seed, rounds=12, per=200, distinct=60, dup_sites=True):
    """Repeating lines, zero counts, the same site with two count pairs in
    one drain, CND / malformed / blank lines."""
    rnd = random.Random(seed)
    pcs = [rnd.randrange(1 << 40) for _ in range(distinct // 2 if dup_sites else distinct)]
    sites = [
        f"CNS trace_cmp{rnd.choice((1, 2, 4, 8))} 0x{rnd.choice(pcs):x} "
        f"{rnd.randint(0, 3)} {rnd.randint(0, 2)}\n"
        for _ in range(distinct)
    ]
    junk = ["CND 3\n", "CND x\n", "CNS a 0xzz 1 1\n", "CNS a 0x10 1\n", "\n", "XYZ 1 2 3 4\n"]
    return [[rnd.choice(sites + junk) for _ in range(per)] for _ in range(rounds)]


def _collector(tmp_path):
    c = CmplogCollector(site_counts=True, workdir=str(tmp_path))
    c.sites_path = str(tmp_path / "sites")
    (tmp_path / "sites").write_text("")
    return c


def _feed(c, lines):
    it = iter([lines])
    c._read_site_lines = lambda: next(it)
    c.collect_sites()


def _run_new(tmp_path, drains):
    c = _collector(tmp_path)
    lasts = []
    for lines in drains:
        c.last_site_fired, c.last_site_asserted = {}, {}
        _feed(c, lines)
        lasts.append((dict(c.last_site_fired), dict(c.last_site_asserted)))
    return lasts, dict(c.site_fired), dict(c.site_asserted), c.site_dropped


def _run_ref(drains):
    state = ({}, {}, [])
    lasts = [_ref_fold(state, lines) for lines in drains]
    return lasts, state[0], state[1], sum(state[2])


def test_control_oracle_matches_itself():
    """Rule 46: the oracle agrees with a second run of itself."""
    d = _drains(1)
    assert _run_ref(d) == _run_ref(d)


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("cache_max", [8, 65_536])
def test_matches_per_line_fold(tmp_path, monkeypatch, seed, cache_max):
    """Falsification: per-drain and total dicts (zeros, duplicate sites in
    one drain, junk) equal the per-line fold, with and without cache resets."""
    monkeypatch.setattr(cmplog, "SITE_LINE_CACHE_MAX", cache_max)
    d = _drains(seed)
    assert _run_new(tmp_path, d) == _run_ref(d)


def test_two_drains_without_reset_accumulate(tmp_path):
    """Adversarial: last_site_* accumulate until collect_counts resets them."""
    d = _drains(4, rounds=2)
    c = _collector(tmp_path)
    c.last_site_fired, c.last_site_asserted = {}, {}
    _feed(c, d[0])
    _feed(c, d[1])
    state = ({}, {}, [])
    lf, la = {}, {}
    for lines in d:
        f, a = _ref_fold(state, lines)
        for k, v in f.items():
            lf[k] = lf.get(k, 0) + v
        for k, v in a.items():
            la[k] = la.get(k, 0) + v
    assert (dict(c.last_site_fired), dict(c.last_site_asserted)) == (lf, la)


def test_reset_assignments_clear(tmp_path):
    """Adversarial: ``= {}`` resets (as collect_counts does) empty the views."""
    c = _collector(tmp_path)
    _feed(c, _drains(5, rounds=1)[0])
    c.last_site_fired, c.last_site_asserted = {}, {}
    assert dict(c.last_site_fired) == {} and dict(c.last_site_asserted) == {}
    c.site_fired, c.site_asserted = {}, {}
    assert dict(c.site_fired) == {} and dict(c.site_asserted) == {}


def test_many_sites_grow_arrays(tmp_path):
    """Adversarial: far more sites than the initial array capacity."""
    d = _drains(6, rounds=4, per=3000, distinct=6000, dup_sites=False)
    assert _run_new(tmp_path, d) == _run_ref(d)


def test_last_fired_built_on_read(tmp_path):
    """Falsification: the fold does not build last_site_fired (no hot-path
    reader); the first read does, and equals the per-line fold."""
    d = _drains(7, rounds=1)
    c = _collector(tmp_path)
    c.last_site_fired, c.last_site_asserted = {}, {}
    _feed(c, d[0])
    assert c._sites.last_fired_cached() is False
    assert dict(c.last_site_fired) == _ref_fold(({}, {}, []), d[0])[0]
    assert c._sites.last_fired_cached() is True
