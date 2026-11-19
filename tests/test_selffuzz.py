"""tools/fuzz.py + tools/lib/fuzzkit: robustness self-fuzzer for fuzzer-tool's own parsers."""

from __future__ import annotations

import argparse
import importlib.util
import pickle
import sys
from pathlib import Path

import pytest

_TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(_TOOLS / "lib"))

from fuzzkit import campaign, corpus, mutators, regress, targets  # noqa: E402

from fuzzer_tool.core.analyzers.analyzer_format_learner import FormatCluster  # noqa: E402
from fuzzer_tool.core.rand_pool import RandPool  # noqa: E402


def _load_driver():
    spec = importlib.util.spec_from_file_location("selffuzz_driver", _TOOLS / "fuzz.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _opts(tmp_path: Path, **kw) -> argparse.Namespace:
    base = {
        "iterations": 40,
        "seed": 1234,
        "time_budget": 0,
        "crash_dir": str(tmp_path / "crashes"),
        "kinds": ",".join(campaign.KIND_WEIGHTS),
    }
    base.update(kw)
    return argparse.Namespace(**base)


# ── mutators ─────────────────────────────────────────────────────────────────


class TestMutators:
    def test_same_seed_same_output(self):
        data = bytes(range(64))
        a = mutators.mutate_bytes(RandPool(7), data)
        b = mutators.mutate_bytes(RandPool(7), data)

        assert a == b

    def test_falsify_seeds_diverge(self):
        data = bytes(range(64))
        outs = {mutators.mutate_bytes(RandPool(s), data) for s in range(8)}

        assert len(outs) > 1

    def test_adversarial_empty_input(self):
        out = mutators.mutate_bytes(RandPool(1), b"")

        assert isinstance(out, bytes)
        assert len(out) <= mutators.MAX_MUTANT_LEN

    def test_obj_mutation_stays_picklable(self):
        obj = {"a": [1, 2, {"b": "x"}], "c": b"\x00"}
        out = mutators.mutate_obj(RandPool(3), obj)

        assert pickle.loads(pickle.dumps(out)) == out


# ── targets / oracles ────────────────────────────────────────────────────────


class TestTargets:
    def test_garbage_elf_is_handled(self):
        assert targets.run_target("elf_bytes", b"\x7fELF" + b"\xff" * 200) is None

    def test_state_load_never_raises(self):
        assert targets.run_target("state", b"not a pickle") is None

    def test_falsify_unhandled_exception_reported(self, monkeypatch):
        def boom(_data: bytes) -> None:
            raise IndexError("planted")

        monkeypatch.setitem(targets.TARGETS, "planted", targets.Target("planted", "bin", boom))

        detail = targets.run_target("planted", b"x")

        assert detail is not None
        assert "IndexError" in detail

    def test_slow_call_reported(self, monkeypatch):
        monkeypatch.setattr(corpus, "SLOW_SECONDS", -1.0)

        detail = targets.run_target("state", b"")

        assert detail is not None
        assert detail.startswith("slow")

    def test_ops_mock_gap_is_skipped(self, monkeypatch):
        # A handler reading a mock-absent attribute is a harness gap, not a bug.
        name = targets.op_names()[0]

        def gap(_buf, _idx, _data):
            raise AttributeError(f"'{targets.MockFuzzer.__name__}' object has no attribute 'x'")

        monkeypatch.setitem(targets._op_table(), name, gap)

        assert targets.run_target(f"op.{name}", b"abc") is None

    def test_adversarial_op_bad_return_type(self, monkeypatch):
        name = targets.op_names()[0]
        monkeypatch.setitem(targets._op_table(), name, lambda _b, _i, _d: 42)

        detail = targets.run_target(f"op.{name}", b"abc")

        assert detail is not None
        assert "int" in detail

    def test_decode_failure_is_not_a_finding(self):
        # Input that is not a pickle never reaches from_dict: skip, not fail.
        assert targets.run_target("markov", b"\x00garbage") is None


# ── crash save / replay ──────────────────────────────────────────────────────


class TestReplay:
    def test_roundtrip_failing_and_passing(self, tmp_path, monkeypatch):
        def boom(data: bytes) -> None:
            if data == b"bad":
                raise KeyError("planted")

        monkeypatch.setitem(targets.TARGETS, "planted", targets.Target("planted", "bin", boom))

        bad = mutators.save_crash(tmp_path, "planted", "bin", b"bad")
        good = mutators.save_crash(tmp_path, "planted", "bin", b"good")

        assert campaign.replay(bad) == 1
        assert campaign.replay(good) == 0

    def test_adversarial_unknown_target_file(self, tmp_path):
        path = tmp_path / "nosuch__0123.bin"
        path.write_bytes(b"x")

        assert campaign.replay(path) == 2


# ── campaign ─────────────────────────────────────────────────────────────────


class TestCampaign:
    def test_control_same_seed_same_cases(self, tmp_path):
        # Hard Rule 46: the campaign against itself must match exactly.
        a = campaign.case_labels(_opts(tmp_path, seed=99), tmp_path / "a")
        b = campaign.case_labels(_opts(tmp_path, seed=99), tmp_path / "b")

        assert a == b

    def test_falsify_other_seed_other_cases(self, tmp_path):
        a = campaign.case_labels(_opts(tmp_path, seed=1), tmp_path / "a")
        b = campaign.case_labels(_opts(tmp_path, seed=2), tmp_path / "b")

        assert a != b

    def test_mutate_clean_on_current_tree(self, tmp_path):
        fails = campaign.fuzz_mutate(tmp_path / "w", _opts(tmp_path, kinds="ops,regress"))

        assert fails == []
        assert not (tmp_path / "crashes").exists()

    def test_planted_bug_found_and_saved(self, tmp_path, monkeypatch):
        def boom(_data: bytes) -> None:
            raise ZeroDivisionError("planted")

        monkeypatch.setitem(targets.TARGETS, "state", targets.Target("state", "pkl", boom))
        # Only the planted target: section targets carry real findings.
        monkeypatch.setitem(campaign._BUILDERS, "state", lambda rng: ("state", rng.randbytes(8)))

        fails = campaign.fuzz_mutate(tmp_path / "w", _opts(tmp_path, kinds="state"))

        assert len(fails) == 1  # deduplicated by crash key
        assert list((tmp_path / "crashes").glob("state__*.pkl"))

    def test_adversarial_bad_kinds(self):
        with pytest.raises(ValueError, match="bogus"):
            campaign.parse_kinds("ops,bogus")


# ── regress ──────────────────────────────────────────────────────────────────


class TestRegress:
    @pytest.mark.parametrize("name", sorted(regress.REGRESSION_MUTATORS))
    def test_control_passes_on_current_tree(self, name, tmp_path):
        for seed in range(5):
            assert regress.run_regression(name, RandPool(seed), tmp_path) is None

    def test_falsify_none_offset_bug_detected(self, monkeypatch, tmp_path):
        # Reintroduce the 2026-11-19 crash: span-scan a None offset.
        def buggy(self, entry):
            for h in self.hypotheses:
                if h.offset <= entry.mutation_offset < h.offset + h.width:
                    return True
            return None

        monkeypatch.setattr(FormatCluster, "_predict_effect", buggy)

        details = [regress.run_regression("none_offset", RandPool(s), tmp_path) for s in range(5)]

        assert any(d and "TypeError" in d for d in details)

    def test_falsify_flood_bound_detected(self, monkeypatch, tmp_path):
        from fuzzer_tool.core.analyzers import analyzer_format_learner as afl

        # Cap shrunk below what the oracle reads -> bound appears violated.
        monkeypatch.setattr(regress, "_flood_cap", lambda: afl.MAX_TRACKED_SIGNATURES - 1)
        monkeypatch.setattr(afl, "MAX_TRACKED_SIGNATURES", 4)

        detail = regress.run_regression("mg_flood", RandPool(0), tmp_path)

        assert detail is not None


# ── driver ───────────────────────────────────────────────────────────────────


class TestDriver:
    def test_unknown_suite_rejected(self):
        drv = _load_driver()

        with pytest.raises(SystemExit):
            drv.main(["nosuch"])

    def test_adversarial_unknown_kind_rejected(self):
        drv = _load_driver()

        with pytest.raises(SystemExit):
            drv.main(["mutate", "--kinds", "bogus"])

    def test_fixed_suites_clean(self, tmp_path, capsys):
        drv = _load_driver()

        rc = drv.main(["elf", "ops", "--verbose", "--crash-dir", str(tmp_path / "c")])

        assert rc == 0
        assert "TOTAL: 0 failure(s)" in capsys.readouterr().out
