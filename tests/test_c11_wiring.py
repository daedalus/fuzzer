"""--c11 wiring: CLI -> Fuzzer -> C11Channel -> seed_meta -> seed weight."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from fuzzer_tool.adapters.sancov_tables import PC_FLAG_ENTRY, SancovKind, SancovRecord
from fuzzer_tool.core.c11 import c11_weight
from fuzzer_tool.services.c11_channel import READY_ATTEMPTS, C11Channel
from fuzzer_tool.services.fuzzer import Fuzzer
from fuzzer_tool.services.seed_picker import SeedPicker

ENTRY_PC = 0x1000
OTHER_PC = 0x1040
FIRST_GUARD = 4
LOCALS = 7


class _Shm:
    def __init__(self, value):
        self.value = value
        self.filled = None

    def fill(self, table):
        self.filled = table.copy()

    def read(self, gen):
        return self.value

    def cleanup(self):
        pass


class _Tables:
    """pc-table dump: absent until *ready_after* calls."""

    def __init__(self, ready_after=0):
        self.calls = 0
        self.ready_after = ready_after

    def records(self):
        self.calls += 1
        if self.calls <= self.ready_after:
            return []
        words = np.array([ENTRY_PC, PC_FLAG_ENTRY, OTHER_PC, 0], dtype=np.uint64)
        return [SancovRecord(kind=SancovKind.PCS, first=FIRST_GUARD, bias=0, words=words)]


def _channel(tables, shm, locals_by_pc=None):
    counts = {ENTRY_PC: LOCALS} if locals_by_pc is None else locals_by_pc
    return C11Channel(shm, tables, lambda: counts)


class TestChannel:
    def test_fills_table_once_dump_exists(self):
        shm, tables = _Shm(LOCALS), _Tables(ready_after=1)
        ch = _channel(tables, shm)

        assert ch.measure(gen=1) == 0, "no dump yet: unmeasured"
        assert ch.measure(gen=2) == LOCALS
        expected = np.zeros(FIRST_GUARD + 1, dtype=np.uint8)  # sized to the last entry
        expected[FIRST_GUARD] = LOCALS  # the non-entry block at OTHER_PC stays out
        assert np.array_equal(shm.filled, expected)

    def test_gives_up_without_pc_table(self, capsys):
        """Adversarial: a target built without pc-table never stalls admissions."""
        tables = _Tables(ready_after=10**9)
        ch = _channel(tables, _Shm(LOCALS))
        for gen in range(READY_ATTEMPTS + 5):
            assert ch.measure(gen) == 0

        assert tables.calls == READY_ATTEMPTS
        assert "pc-table" in capsys.readouterr().out

    def test_no_debug_info_gives_up(self, capsys):
        """Falsification: no DWARF locals -> nothing measured, one warning."""
        ch = _channel(_Tables(), _Shm(LOCALS), locals_by_pc={})
        assert ch.measure(1) == 0
        assert ch.measure(2) == 0
        assert capsys.readouterr().out.count("--c11") == 1


class TestFuzzerWiring:
    def test_constructor_takes_c11(self):
        assert "c11" in inspect.signature(Fuzzer.__init__).parameters

    def test_note_writes_meta(self):
        seed = b"seed"
        f = SimpleNamespace(
            _c11=SimpleNamespace(measure=lambda gen: LOCALS),
            shm_cov=SimpleNamespace(read_generation=lambda: 3),
            seed_meta={seed: {}},
        )
        Fuzzer._note_c11(f, seed)
        assert f.seed_meta[seed]["c11"] == LOCALS

    def test_note_off_is_noop(self):
        seed = b"seed"
        f = SimpleNamespace(_c11=None, seed_meta={seed: {}})
        Fuzzer._note_c11(f, seed)
        assert "c11" not in f.seed_meta[seed]


class TestSeedWeight:
    def _picker(self, metas, stalled=False, c11_on=True):
        f = SimpleNamespace(
            _c11=object() if c11_on else None,
            _stall_recovery_active=stalled,
            seed_meta={bytes([i]): m for i, m in enumerate(metas)},
        )
        p = SeedPicker.__new__(SeedPicker)
        p.f = f
        return p, f

    def test_factor_applied(self):
        metas = [{"c11": 2}, {"c11": 10}, {}]
        p, f = self._picker(metas)
        rng = p._c11_range(f)

        assert rng == (2, 10)
        assert p._weight_c11(metas[1], 1.0, rng) == pytest.approx(c11_weight(10, 2, 10))
        assert p._weight_c11(metas[2], 1.0, rng) == 1.0

    def test_off_while_stalled(self):
        """AFL++ drops the factor while starved; so does stall recovery."""
        p, f = self._picker([{"c11": 2}, {"c11": 10}], stalled=True)
        assert p._c11_range(f) is None

    def test_off_without_flag(self):
        p, f = self._picker([{"c11": 2}, {"c11": 10}], c11_on=False)
        assert p._c11_range(f) is None
        assert p._weight_c11({"c11": 10}, 3.0, None) == 3.0


def test_cli_flag(monkeypatch):
    from tests.test_lightweight_mode import _cli

    args, _ = _cli(monkeypatch, "--c11")
    assert args.c11 is True


class TestResume:
    """seed_meta persists through a key whitelist: c11 must be on it."""

    SEED = b"c11-resume-seed"

    @pytest.fixture(autouse=True)
    def _instrumented_target(self, monkeypatch):
        monkeypatch.setattr("fuzzer_tool.core.elf.sancov_guard_status", lambda _t: "present")
        monkeypatch.setattr("fuzzer_tool.core.elf.detect_ctx_bits", lambda _t: 4)

    def _fuzzer(self, tmp_path, **kw):
        from fuzzer_tool.adapters.filesystem import save_to_corpus

        corpus, crashes = tmp_path / "c", tmp_path / "k"
        (corpus / "seeds").mkdir(parents=True, exist_ok=True)
        crashes.mkdir(parents=True, exist_ok=True)
        save_to_corpus(self.SEED, corpus, set(), None)
        return Fuzzer(
            target="targets/test_target",
            corpus_dir=str(corpus),
            crashes_dir=str(crashes),
            max_len=4096,
            **kw,
        )

    def test_c11_survives_resume(self, tmp_path):
        f = self._fuzzer(tmp_path)
        f.seed_meta[self.SEED]["c11"] = LOCALS
        f._save_state()

        resumed = self._fuzzer(tmp_path, resume=True)
        assert resumed.seed_meta[self.SEED]["c11"] == LOCALS

    def test_unmeasured_seed_resumes_as_unmeasured(self, tmp_path):
        """Adversarial: a state without c11 (older run, --c11 off) resumes as 0."""
        f = self._fuzzer(tmp_path)
        f._save_state()

        resumed = self._fuzzer(tmp_path, resume=True)
        assert resumed.seed_meta[self.SEED].get("c11", 0) == 0
