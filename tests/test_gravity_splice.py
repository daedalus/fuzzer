"""Wiring of --splice-donor gravity: EdgeTracker → OperatorEngine → Fuzzer → CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

from fuzzer_tool.core.edge_tracker import EdgeTracker
from fuzzer_tool.core.gravity import DONOR_CANDIDATES, GravityModel, SpliceDonor
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.services.corpus_manager import seed_key
from fuzzer_tool.services.operators import OperatorEngine
from fuzzer_tool.services.stats import StatsReporter
from tests.support.operator_env import make_minimal_fuzzer
from tests.support.scripted_rng import ScriptedRng
from tests.test_dirichlet_wiring import _TARGET, _fuzzer_call_kwargs, _parse, build  # noqa: F401
from tests.test_regression_enabled_features_entropy_deviation import _make_fake_fuzzer

# Seeds and their edge sets. CLONE covers exactly what BASE covers (inert
# donor); FAR shares half of BASE's edges and owns 25 of its own.
BASE = b"base-seed-" + b"A" * 30
CLONE = b"clone-seed" + b"B" * 30
FAR = b"far-seed--" + b"C" * 30
_EDGES = {
    BASE: set(range(1, 51)),
    CLONE: set(range(1, 51)),
    FAR: set(range(1, 26)) | set(range(100, 125)),
}


def _tracker(seeds) -> EdgeTracker:
    et = EdgeTracker(map_size=1024)
    for s in seeds:
        et.record_edges(seed_key(s), _EDGES[s])
    return et


def _engine(corpus, rng, gravity: GravityModel | None) -> OperatorEngine:
    f = make_minimal_fuzzer(pool=rng)
    f.corpus = list(corpus)
    f._edge_tracker = _tracker(corpus)
    f._gravity = gravity
    f._seed_key = seed_key
    return OperatorEngine(f)


class TestSeedOverlaps:
    def test_unknown_seed_is_none(self):
        et = _tracker([BASE])
        assert et.seed_overlaps(seed_key(BASE), ["0" * 16]) == [None]
        assert et.seed_overlaps("0" * 16, [seed_key(BASE)]) == [None]

    def test_identical_sets_estimate_one(self):
        et = _tracker([BASE, CLONE])
        assert et.seed_overlaps(seed_key(BASE), [seed_key(CLONE)]) == [(50, 50, 1.0)]

    def test_partial_overlap_estimate(self):
        # |∩|=25, |∪|=75 → J=1/3; 64 permutations resolve that to ~±0.15.
        et = _tracker([BASE, FAR])
        [(size_a, size_b, jac)] = et.seed_overlaps(seed_key(BASE), [seed_key(FAR)])
        assert (size_a, size_b) == (50, 50)
        assert jac == pytest.approx(1 / 3, abs=0.15)

    def test_batch_matches_scalar_estimate(self):
        # Oracle: the tracker's existing scalar MinHash Jaccard, pair by pair.
        et = _tracker([BASE, CLONE, FAR])
        keys = [seed_key(s) for s in (CLONE, FAR, BASE)]
        got = [jac for _, _, jac in et.seed_overlaps(seed_key(BASE), keys)]
        want = [et._minhash.approximate_jaccard(seed_key(BASE), k) for k in keys]
        assert got == want

    def test_adversarial_mixed_known_unknown_keeps_positions(self):
        et = _tracker([BASE, FAR])
        out = et.seed_overlaps(seed_key(BASE), ["0" * 16, seed_key(FAR), "1" * 16])
        assert out[0] is None and out[2] is None
        assert out[1] is not None and out[1][:2] == (50, 50)


class TestDonor:
    def test_uniform_is_one_choice_draw(self):
        # Bit-identical to the pre-gravity call: exactly one choice(), nothing else.
        eng = _engine([BASE, CLONE, FAR], ScriptedRng(choice_idxs=(2,)), None)
        assert eng._donor(BASE, [BASE, CLONE, FAR], eng.ctx._rng) == FAR

    def test_falsification_clone_never_picked(self):
        # CLONE weighs 0, so every CDF draw lands on FAR — r at both ends.
        for r in (0.0, 0.999):
            g = GravityModel()
            eng = _engine([BASE, CLONE, FAR], ScriptedRng(randoms=(r,)), g)
            assert eng._donor(BASE, [CLONE, FAR], eng.ctx._rng) == FAR
            assert g.pending == 1

    def test_adversarial_all_inert_falls_back_to_uniform(self):
        g = GravityModel()
        eng = _engine([BASE, CLONE], ScriptedRng(choice_idxs=(0,)), g)
        assert eng._donor(BASE, [CLONE], eng.ctx._rng) == CLONE
        # Nothing staged: a fallback pick is not a gravity observation.
        assert g.pending == 0

    def test_adversarial_unknown_base_falls_back(self):
        g = GravityModel()
        eng = _engine([CLONE, FAR], ScriptedRng(choice_idxs=(1,)), g)
        assert eng._donor(b"never-executed", [CLONE, FAR], eng.ctx._rng) == FAR
        assert g.pending == 0

    def test_large_pool_samples_k_candidates(self):
        # Pool > k: k scripted randint draws pick candidates, then one CDF draw.
        pool = [CLONE] * (DONOR_CANDIDATES + 3) + [FAR]
        picks = [0] * (DONOR_CANDIDATES - 1) + [len(pool) - 1]
        g = GravityModel()
        eng = _engine([BASE, CLONE, FAR], ScriptedRng(randints=picks, randoms=(0.5,)), g)
        assert eng._donor(BASE, pool, eng.ctx._rng) == FAR


_SPLICE_OPS = (
    "_op_splice",
    "_op_splice_diff_located",
    "_op_splice_common_prefix",
    "_op_insert_range_from_other",
    "_op_crossover",
    "_op_fuse_next",
)


@pytest.mark.parametrize("op", _SPLICE_OPS)
def test_splice_family_routes_through_gravity(op):
    g = GravityModel()
    eng = _engine([BASE, CLONE, FAR], RandPool(seed=7), g)
    getattr(eng, op)(bytearray(BASE), 0, BASE)
    assert g.pending >= 1


class TestFuzzerWiring:
    def test_off_by_default(self, build):  # noqa: F811
        assert build()._gravity is None

    def test_gravity_builds_model(self, build):  # noqa: F811
        assert isinstance(build(splice_donor=SpliceDonor.GRAVITY)._gravity, GravityModel)

    def test_mutate_discards_stale_pairs(self, build):  # noqa: F811
        f = build(splice_donor=SpliceDonor.GRAVITY)
        f._gravity.stage(1, 2, 0.5)
        f.mutate(b"GET / HTTP/1.1\r\n\r\n")
        assert f._gravity.pending == 0

    def test_cli_default_and_value(self, monkeypatch):
        assert _parse(monkeypatch).splice_donor == SpliceDonor.UNIFORM.value
        assert _parse(monkeypatch, "--splice-donor", "gravity").splice_donor == "gravity"

    def test_adversarial_cli_rejects_unknown(self, monkeypatch):
        with pytest.raises(SystemExit):
            _parse(monkeypatch, "--splice-donor", "newton")

    def test_hail_mary_enables(self, monkeypatch):
        assert _parse(monkeypatch, "--hail-mary").splice_donor == SpliceDonor.GRAVITY.value

    def test_cmd_fuzz_forwards(self):
        assert all("splice_donor" in k for k in _fuzzer_call_kwargs())

    def test_banner(self, capsys):
        from fuzzer_tool.services.fuzzer import Fuzzer

        Fuzzer._print_enabled_features(_make_fake_fuzzer())
        assert "splice-donor" not in capsys.readouterr().out
        Fuzzer._print_enabled_features(_make_fake_fuzzer(_gravity=GravityModel()))
        assert "splice-donor=gravity" in capsys.readouterr().out

    def test_resume_round_trip(self, build):  # noqa: F811
        f = build(splice_donor=SpliceDonor.GRAVITY)
        f._gravity.stage(1, 2, 0.5)
        f._gravity.observe(3)
        f._save_learned()
        g = build(splice_donor=SpliceDonor.GRAVITY)
        g.resume = True
        g._state_store = f._state_store
        g._load_learned()
        assert g._gravity.summary() == f._gravity.summary()

    def test_summary_line(self, capsys):
        g = GravityModel()
        g.stage(1, 2, 0.5)
        g.observe(4)
        fake = _make_fake_fuzzer(_gravity=g)
        StatsReporter(fake)._print_summary_gravity(fake)
        out = capsys.readouterr().out
        assert "Gravity splice" in out and "gamma" in out
        fake = _make_fake_fuzzer()
        StatsReporter(fake)._print_summary_gravity(fake)
        assert capsys.readouterr().out == ""


@pytest.mark.skipif(not Path(_TARGET).exists(), reason="targets/test_target not built")
def test_fuzz_one_observes_every_splice_round(build):  # noqa: F811
    """Each executed round is one observation; an unconfirmed round credits 0.

    The coverage verdict is scripted (Hard Rule 39) so both branches run.
    """
    f = build(splice_donor=SpliceDonor.GRAVITY, use_coverage=True)
    # (verdict, confirmed edge set) per round: 3 fresh edges, then nothing.
    fresh = {101, 102, 103}
    script = [(True, fresh), (False, None)]
    verdicts = iter(script)
    f._confirm_new_coverage = lambda *a, **k: next(verdicts)
    # SHM path: the confirmed hit counts are the same scripted edges.
    f._only_confirmed = lambda _counts: dict.fromkeys(fresh, 1)

    # Stand-in for a splice op: stage one pair inside the round's mutate().
    real_mutate = f._operators.mutate

    def mutate(data):
        out = real_mutate(data)
        f._gravity.stage(1, 2, 0.5)
        return out

    f._operators.mutate = mutate
    new_edges = []
    for i in range(len(script)):
        f.fuzz_one(b"GET /" + bytes([65 + i]) * 8)
        new_edges.append(f._last_new_edge_count)

    ys = f._gravity.state_dict()["ys"]
    assert f._gravity.pending == 0
    # The confirmed round credits the edges record_edges just counted, not
    # the zero _last_new_edge_count holds before recording runs.
    assert new_edges == [len(fresh), 0]
    assert ys == [len(fresh), 0]


class TestBenchArm:
    """tools/bench_paired.py arm: baseline plus exactly --splice-donor gravity."""

    @pytest.fixture(autouse=True)
    def _tools_path(self, monkeypatch):
        monkeypatch.syspath_prepend(str(Path(__file__).resolve().parent.parent / "tools"))

    def test_arm_pairs_against_baseline(self):
        from bench_paired import ARM_BASELINES, ARMS

        assert ARM_BASELINES["splice-gravity"] == "baseline"
        assert ARMS["splice-gravity"] == [*ARMS["baseline"], "--splice-donor", "gravity"]

    def test_arm_flags_reach_the_real_parser(self, monkeypatch):
        from bench_paired import ARMS

        assert _parse(monkeypatch, *ARMS["splice-gravity"]).splice_donor == "gravity"
        assert _parse(monkeypatch, *ARMS["baseline"]).splice_donor == "uniform"
