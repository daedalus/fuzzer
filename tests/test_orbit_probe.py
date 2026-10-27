"""Orbit probe: run a folded repeat against its orbit minimum, flag wrong declarations."""

from fuzzer_tool.core.orbit_dedup import (
    OrbitProbe,
    Verdict,
    canonical,
    judge,
    parse_orbit_spec,
    run_probe,
)
from tests.test_orbit_dedup import HEAD, SWAP3, _Stub


class _Rng:
    """Scripted RNG: returns the queued values in order."""

    def __init__(self, *vals):
        self._vals = list(vals)

    def random(self):
        return self._vals.pop(0)


def _runner(table):
    """Fake target: edge set looked up by exact input bytes."""
    calls = []

    def run(data):
        calls.append(data)
        return table(data)

    run.calls = calls
    return run


def test_judge_same_diff_flaky():
    assert judge({1, 2}, {1, 2}, {1, 2}) is Verdict.SAME
    assert judge({1, 2}, {1, 2}, {1, 3}) is Verdict.DIFF
    assert judge({1, 2}, {1, 4}, {1, 3}) is Verdict.FLAKY


def test_judge_control_never_diff_against_itself():
    assert judge({5}, {5}, {5}) is not Verdict.DIFF


def test_due_period_one_in_n():
    probe = OrbitProbe(period=4)
    assert probe.due(_Rng(0.0)) is True
    assert probe.due(_Rng(0.26)) is False


def test_due_period_zero_is_off():
    assert OrbitProbe(period=0).due(_Rng(0.0)) is False


def test_run_probe_symmetric_target_is_same():
    sym = lambda d: {1, 2} if sorted([d[2:4], d[4:6], d[6:8]]) else set()  # noqa: E731
    x = HEAD + b"ccaabb"

    assert run_probe(x, SWAP3, _runner(sym)) is Verdict.SAME


def test_run_probe_wrong_declaration_is_diff():
    ordered = lambda d: {hash(d[2:8]) & 0xFFFF}  # noqa: E731
    x = HEAD + b"ccaabb"

    assert run_probe(x, SWAP3, _runner(ordered)) is Verdict.DIFF


def test_run_probe_runs_input_twice_and_minimum_once():
    run = _runner(lambda d: {1})
    x = HEAD + b"ccaabb"
    run_probe(x, SWAP3, run)

    assert run.calls == [x, x, canonical(x, SWAP3)]


def test_run_probe_flaky_target_never_diff():
    n = iter(range(100))
    flaky = lambda d: {next(n)}  # noqa: E731

    assert run_probe(HEAD + b"ccaabb", SWAP3, _runner(flaky)) is Verdict.FLAKY


def test_adversarial_minimum_input_not_probed():
    x = canonical(HEAD + b"ccaabb", SWAP3)
    f = _Stub(SWAP3)
    f._orbit_probe = OrbitProbe(period=1)
    parent = HEAD + b"xxyyzz"
    f._is_repeat(x, parent)

    assert f._is_repeat(x, parent) is True
    assert f._orbit_probe_pending is False


def test_is_repeat_probe_lets_folded_repeat_through():
    f = _Stub(SWAP3)
    f._orbit_probe = OrbitProbe(period=1)
    f._orbit_probe_rng = _Rng(0.0)
    parent = HEAD + b"xxyyzz"
    f._is_repeat(HEAD + b"bbaacc", parent)

    assert f._is_repeat(HEAD + b"ccaabb", parent) is False
    assert f._orbit_probe_pending is True


def test_falsification_probe_off_keeps_dedup():
    f = _Stub(SWAP3)
    parent = HEAD + b"xxyyzz"
    f._is_repeat(HEAD + b"bbaacc", parent)

    assert f._is_repeat(HEAD + b"ccaabb", parent) is True


def test_record_counts():
    probe = OrbitProbe(period=1)
    for v in (Verdict.SAME, Verdict.DIFF, Verdict.FLAKY, Verdict.DIFF):
        probe.record(v)

    assert (probe.probes, probe.diffs, probe.flaky) == (4, 2, 1)


def test_suspect_needs_min_probes():
    probe = OrbitProbe(period=1)
    probe.record(Verdict.DIFF)

    assert probe.suspect() is False
    for _ in range(OrbitProbe.MIN_PROBES):
        probe.record(Verdict.DIFF)

    assert probe.suspect() is True


def test_suspect_false_when_all_same():
    probe = OrbitProbe(period=1)
    for _ in range(50):
        probe.record(Verdict.SAME)

    assert probe.suspect() is False


def test_parse_spec_unchanged():
    assert parse_orbit_spec("swap@2:2x3")


class _Shm:
    shm_cov = True


def _round(f, mutated, table):
    from fuzzer_tool.services.fuzz_round import FuzzRound

    class _Round(FuzzRound):
        def _probe_run(self, candidate):
            return table(candidate)

    r = _Round.__new__(_Round)
    r._f = f
    r._mutated = mutated
    return r


def _holder(specs, period=1):
    f = _Shm()
    f._orbit_specs = specs
    f._orbit_probe = OrbitProbe(period)
    f._orbit_probe_pending = True
    return f


def test_round_probe_records_diff_and_warns_once(capsys):
    f = _holder(SWAP3)
    r = _round(f, HEAD + b"ccaabb", lambda d: {hash(d)})
    for _ in range(OrbitProbe.MIN_PROBES + 2):
        r._probe_orbit()

    assert f._orbit_probe.diffs == OrbitProbe.MIN_PROBES + 2
    assert capsys.readouterr().out.count("declaration likely wrong") == 1


def test_round_probe_skips_when_not_pending():
    f = _holder(SWAP3)
    f._orbit_probe_pending = False
    r = _round(f, HEAD + b"ccaabb", lambda d: {1})
    r._probe_orbit()

    assert f._orbit_probe.probes == 0


def test_adversarial_round_probe_without_shm_does_nothing():
    f = _holder(SWAP3)
    f.shm_cov = None
    r = _round(f, HEAD + b"ccaabb", lambda d: {1})
    r._probe_orbit()

    assert f._orbit_probe.probes == 0


def test_falsification_round_probe_correct_spec_stays_quiet(capsys):
    f = _holder(SWAP3)
    r = _round(f, HEAD + b"ccaabb", lambda d: {hash(canonical(d, SWAP3))})
    for _ in range(OrbitProbe.MIN_PROBES + 2):
        r._probe_orbit()

    assert f._orbit_probe.diffs == 0
    assert "likely wrong" not in capsys.readouterr().out
