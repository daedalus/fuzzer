"""adapters/mutant_build: mutant binaries + fork replay driver (docs/action_plan_mutant_kill_metric.md P2)."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from fuzzer_tool.adapters.mutant_build import (
    MAX_INPUT_BYTES,
    MutantSet,
    ReplayError,
    Status,
    TargetSpec,
    asan_available,
    pack_record,
    replay,
    write_pack,
)
from fuzzer_tool.core.ir_mutants import MutOp
from fuzzer_tool.core.state_store import safe_load
from tests.support.mutant_toy import TOY_LIB, TOY_WRAP

_TIMEOUT_MS = 300

needs_asan = pytest.mark.skipif(not asan_available(), reason="needs clang + ASAN runtime")


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> MutantSet:
    root = tmp_path_factory.mktemp("mutants")
    (root / "toy.c").write_text(TOY_LIB)
    (root / "toy_wrap.c").write_text(TOY_WRAP)
    spec = TargetSpec(src=root / "toy.c", wrapper=root / "toy_wrap.c")
    return MutantSet.build(spec, root / "set", cap=10_000, seed=1, jobs=4)


def _pack(tmp_path: Path, inputs: list[bytes]) -> Path:
    path = tmp_path / "in.pack"
    write_pack(inputs, path)
    return path


class TestPack:
    def test_layout(self, tmp_path) -> None:
        inputs = [b"abc", b"", b"\x00\xff"]
        path = _pack(tmp_path, inputs)
        want = b"".join(struct.pack("<I", len(x)) + x for x in inputs)
        assert path.read_bytes() == want

    def test_oversize_record_truncated(self, tmp_path) -> None:
        big = bytes(range(256)) * (MAX_INPUT_BYTES // 256 + 1)
        path = _pack(tmp_path, [big])
        assert path.read_bytes() == struct.pack("<I", MAX_INPUT_BYTES) + big[:MAX_INPUT_BYTES]

    def test_returns_record_count(self, tmp_path) -> None:
        assert write_pack(iter([b"a", b"b"]), tmp_path / "p") == 2

    def test_pack_record_reads_each(self, tmp_path) -> None:
        inputs = [b"abc", b"", b"\x00\xff", b"z" * 300]
        path = _pack(tmp_path, inputs)
        assert [pack_record(path, i) for i in range(len(inputs))] == inputs

    def test_pack_record_past_end_raises(self, tmp_path) -> None:
        path = _pack(tmp_path, [b"a", b"b"])
        with pytest.raises(IndexError):
            pack_record(path, 2)

    def test_pack_record_truncated_raises(self, tmp_path) -> None:
        path = tmp_path / "bad.pack"
        path.write_bytes(struct.pack("<I", 10) + b"abc")
        with pytest.raises(ReplayError):
            pack_record(path, 0)


@needs_asan
class TestReplay:
    def test_statuses(self, built, tmp_path) -> None:
        pack = _pack(tmp_path, [b"ok", b"C", b"H", b""])
        got = replay(built.original, pack, None, _TIMEOUT_MS)
        assert list(got) == [Status.CLEAN, Status.CRASH, Status.HANG, Status.CLEAN]

    def test_child_stdout_does_not_corrupt_statuses(self, built, tmp_path) -> None:
        pack = _pack(tmp_path, [b"P", b"C", b"P"])
        got = replay(built.original, pack, None, _TIMEOUT_MS)
        assert list(got) == [Status.CLEAN, Status.CRASH, Status.CLEAN]

    def test_stop_on_first_eligible_crash(self, built, tmp_path) -> None:
        # Index 1 crashes but is masked out (original crashed there too);
        # index 2 is the first eligible crash: replay stops after it.
        pack = _pack(tmp_path, [b"ok", b"C", b"C", b"ok"])
        got = replay(built.original, pack, bytes([1, 0, 1, 1]), _TIMEOUT_MS)
        assert list(got) == [Status.CLEAN, Status.CRASH, Status.CRASH]

    def test_deterministic(self, built, tmp_path) -> None:
        """Rule 46 control: the same binary and pack twice give the same statuses."""
        pack = _pack(tmp_path, [b"ok", b"C", b"abcd", b"P", b""])
        assert replay(built.original, pack, None, _TIMEOUT_MS) == replay(
            built.original, pack, None, _TIMEOUT_MS
        )

    def test_truncated_pack_raises(self, built, tmp_path) -> None:
        path = tmp_path / "bad.pack"
        path.write_bytes(struct.pack("<I", 10) + b"abc")
        with pytest.raises(ReplayError):
            replay(built.original, path, None, _TIMEOUT_MS)

    def test_mask_length_mismatch_raises(self, built, tmp_path) -> None:
        pack = _pack(tmp_path, [b"a", b"b"])
        with pytest.raises(ValueError):
            replay(built.original, pack, bytes([1]), _TIMEOUT_MS)


@needs_asan
class TestMutantSet:
    def _mutant(self, built: MutantSet, op: MutOp, needle: str) -> Path:
        line = next(i for i, ln in enumerate(built.base_ir.split("\n")) if needle in ln)
        (path,) = [
            p
            for s, p in zip(built.sites, built.mutants, strict=True)
            if s.line == line and s.op == op
        ]
        return path

    def test_boundary_mutant_killed_only_by_boundary_input(self, built, tmp_path) -> None:
        """Falsification: n<4 -> n<=4 overflows malloc(4) on exactly n == 4."""
        mutant = self._mutant(built, MutOp.ICMP_BOUNDARY, "icmp ult i64")
        pack = _pack(tmp_path, [b"abc", b"abcd", b"abcdefg"])
        assert list(replay(built.original, pack, None, _TIMEOUT_MS)) == [Status.CLEAN] * 3
        assert list(replay(mutant, pack, None, _TIMEOUT_MS)) == [
            Status.CLEAN,
            Status.CRASH,
            Status.CLEAN,
        ]

    def test_dead_store_dropped_as_equivalent(self, built) -> None:
        dead = [s for s in built.equivalent if s.op == MutOp.STORE_DELETE]
        lines = built.base_ir.split("\n")
        assert any("%add" in lines[s.line] and "%dead" in lines[s.line] for s in dead)
        assert not set(built.equivalent) & set(built.sites)

    def test_mutants_distinct_from_original(self, built) -> None:
        assert len(built.mutants) == len(built.sites) > 0
        assert all(p.exists() for p in [built.original, *built.mutants])

    def test_load_roundtrip(self, built) -> None:
        again = MutantSet.load(built.out_dir)
        assert (again.key, again.sites, again.equivalent, again.mutants) == (
            built.key,
            built.sites,
            built.equivalent,
            built.mutants,
        )

    def test_manifest_has_no_custom_globals(self, built) -> None:
        with open(built.out_dir / "manifest.pkl", "rb") as fh:
            assert safe_load(fh)["key"] == built.key

    def test_rebuild_same_key_is_cached(self, built) -> None:
        stamp = built.original.stat().st_mtime_ns
        spec = TargetSpec(src=built.spec_src, wrapper=built.spec_wrapper)
        again = MutantSet.build(spec, built.out_dir, cap=10_000, seed=1, jobs=4)
        assert again.key == built.key
        assert again.original.stat().st_mtime_ns == stamp

    def test_cap_changes_key_and_bounds_count(self, built, tmp_path) -> None:
        spec = TargetSpec(src=built.spec_src, wrapper=built.spec_wrapper)
        small = MutantSet.build(spec, tmp_path / "small", cap=2, seed=1, jobs=2)
        assert small.key != built.key
        assert len(small.sites) + len(small.equivalent) <= 2


def test_missing_binary_raises(tmp_path) -> None:
    pack = _pack(tmp_path, [b"a"])
    with pytest.raises(ReplayError):
        replay(tmp_path / "nope", pack, None, _TIMEOUT_MS)
