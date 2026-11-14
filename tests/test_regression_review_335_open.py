"""Regressions for PR #335 review items first left open."""

import gzip
import io
import os
import pickle
import struct
import tarfile
import tracemalloc
import zipfile

import pytest

from fuzzer_tool.adapters import filesystem as fs
from fuzzer_tool.adapters import seed_tar, seed_zip
from fuzzer_tool.adapters.seed_zip import ZipMode
from fuzzer_tool.core import dwarf
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.state_store import StateStore, UnsafeStateError, safe_load

_BASE = "crash_x"
_OLD = b"old"
_NEW = b"new"


# ── _publish_crash ────────────────────────────────────────────────────


def _artifacts(content=_NEW):
    return [(ext, content + ext.encode(), 0o644) for ext in (".bin", ".json", ".txt")]


def _fail_rename_to(monkeypatch, final_name):
    """Make the tmp -> *final_name* rename raise; every other os.replace runs."""
    real = os.replace

    def replace(src, dst):
        if str(src).endswith(".tmp") and os.path.basename(dst) == final_name:
            raise OSError(28, "No space left on device")
        return real(src, dst)

    monkeypatch.setattr(fs.os, "replace", replace)


class TestPublishCrashAtomic:
    """A failed rename left the earlier renames published: half a crash, old reproducer gone."""

    def test_regression_failure_restores_old_set(self, tmp_path, monkeypatch):
        for ext, content, _ in _artifacts(_OLD):
            (tmp_path / f"{_BASE}{ext}").write_bytes(content)
        before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
        _fail_rename_to(monkeypatch, f"{_BASE}.json")

        with pytest.raises(OSError):
            fs._publish_crash(tmp_path, _BASE, _artifacts())

        assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before

    def test_regression_failure_leaves_nothing_new(self, tmp_path, monkeypatch):
        """Adversarial: no prior crash; the last rename fails."""
        _fail_rename_to(monkeypatch, f"{_BASE}.txt")

        with pytest.raises(OSError):
            fs._publish_crash(tmp_path, _BASE, _artifacts())

        assert list(tmp_path.iterdir()) == []

    def test_success_replaces_and_cleans(self, tmp_path):
        """Falsification: a clean publish replaces every artifact, no side files."""
        (tmp_path / f"{_BASE}.bin").write_bytes(_OLD)

        fs._publish_crash(tmp_path, _BASE, _artifacts())

        expected = {f"{_BASE}{ext}": c for ext, c, _ in _artifacts()}
        assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == expected


# ── DWARF inflate ─────────────────────────────────────────────────────

_SHF_COMPRESSED = 0x800
_E_SHOFF = 40


def _elf(declared: int, packed: bytes) -> bytes:
    """ELF64 with .shstrtab and one zlib-compressed .debug_info of body *packed*."""
    names = b"\x00.shstrtab\x00.debug_info\x00"
    # Elf64_Chdr: ch_type, ch_reserved, ch_size, ch_addralign
    body = struct.pack("<IIQQ", 1, 0, declared, 1) + packed
    shoff = 64 + len(names) + len(body)
    hdr = bytearray(b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 56)
    struct.pack_into("<Q", hdr, _E_SHOFF, shoff)
    struct.pack_into("<HHH", hdr, 58, 64, 3, 1)  # e_shentsize, e_shnum, e_shstrndx

    def shdr(name, flags, off, size):
        return struct.pack("<IIQQQQIIQQ", name, 1, flags, 0, off, size, 0, 0, 1, 0)

    table = (
        bytes(64)
        + shdr(1, 0, 64, len(names))
        + shdr(11, _SHF_COMPRESSED, 64 + len(names), len(body))
    )
    return bytes(hdr) + names + body + table


class TestInflateRatio:
    """A ~64 KiB zlib bomb could declare (and get) up to 1 GiB of section."""

    def test_regression_bomb_stops_at_floor(self):
        import zlib

        declared = 4 * dwarf._FREE_INFLATE
        packed = zlib.compress(bytes(declared), 9)
        assert len(packed) * dwarf._MAX_RATIO < dwarf._FREE_INFLATE

        section = dwarf._elf_sections(_elf(declared, packed))[b".debug_info"]

        assert len(section) == dwarf._FREE_INFLATE

    def test_incompressible_above_floor_inflates(self):
        """Adversarial: a large honest section with a real-world ratio is kept whole."""
        import zlib

        payload = RandPool(seed=0).randbytes(dwarf._FREE_INFLATE + 4096)

        section = dwarf._elf_sections(_elf(len(payload), zlib.compress(payload, 1)))

        assert section[b".debug_info"] == payload

    def test_small_section_any_ratio(self):
        """Falsification: below the floor, a highly regular section inflates fully."""
        import zlib

        payload = bytes(1 << 20)

        section = dwarf._elf_sections(_elf(len(payload), zlib.compress(payload, 9)))

        assert section[b".debug_info"] == payload


# ── seed archive member cap ───────────────────────────────────────────

_CAP = 1024
_SMALL = b"small-seed" * 8
_BIG = bytes(4096 * _CAP)  # 4 MiB: well above the salvage overhead


@pytest.fixture
def capped(monkeypatch):
    monkeypatch.setattr(seed_zip, "MAX_MEMBER_BYTES", _CAP)
    seed_zip._PEEK.clear()
    yield
    seed_zip._PEEK.clear()


@pytest.fixture
def reads(monkeypatch):
    """Names passed to ``ZipFile.read``."""
    seen = []
    real = zipfile.ZipFile.read

    def spy(self, name, pwd=None):
        seen.append(name.filename if isinstance(name, zipfile.ZipInfo) else name)
        return real(self, name, pwd)

    monkeypatch.setattr(zipfile.ZipFile, "read", spy)
    return seen


def _foreign_zip(corpus, members):
    with zipfile.ZipFile(corpus / seed_zip.ZIP_NAME, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)


class TestZipMemberCap:
    """``seeds.zip`` inflated every member whole: one huge member exhausted memory."""

    def test_regression_live_skips_oversized(self, tmp_path, capped, reads):
        _foreign_zip(tmp_path, {"big.bin": _BIG, "a.bin": _SMALL})

        assert [d for d, _p in seed_zip.open_readonly(tmp_path).live()] == [_SMALL]
        assert "big.bin" not in reads

    def test_regression_compact_keeps_unread(self, tmp_path, capped, reads):
        """compact never inflates it whole, and never drops it (corpus is never deleted)."""
        _foreign_zip(tmp_path, {"big.bin": _BIG, "a.bin": _SMALL, "dir/": b""})

        seed_zip.compact(tmp_path)

        assert "big.bin" not in reads
        with zipfile.ZipFile(tmp_path / seed_zip.ZIP_NAME) as zf:
            assert zf.read("big.bin") == _BIG

    def test_regression_salvage_bounded(self, tmp_path, capped):
        path = tmp_path / seed_zip.ZIP_NAME
        _foreign_zip(tmp_path, {"big.bin": _BIG, "a.bin": _SMALL})
        raw = path.read_bytes()
        path.write_bytes(raw[: raw.index(b"PK\x01\x02")])  # torn central directory

        tracemalloc.start()
        try:
            seed_zip._salvage(path)
            _cur, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        assert peak < len(_BIG)
        with zipfile.ZipFile(path) as zf:
            assert zf.namelist() == ["a.bin"]
        assert (tmp_path / f"{seed_zip.ZIP_NAME}.corrupt.0").read_bytes() == raw[
            : raw.index(b"PK\x01\x02")
        ]

    def test_member_at_cap_is_read(self, tmp_path, capped):
        """Falsification: exactly MAX_MEMBER_BYTES is a seed."""
        edge = b"e" * _CAP
        _foreign_zip(tmp_path, {"edge.bin": edge})

        assert [d for d, _p in seed_zip.open_readonly(tmp_path).live()] == [edge]

    def test_writer_adopts_small_keeps_big(self, tmp_path, capped):
        """Adversarial: the writer adopts foreign seeds; the oversized one stays as it was."""
        _foreign_zip(tmp_path, {"big.bin": _BIG, "a.bin": _SMALL})

        store = seed_zip.configure(tmp_path, ZipMode.ON)
        try:
            assert [d for d, _p in store.live()] == [_SMALL]
        finally:
            seed_zip.configure(tmp_path, ZipMode.OFF)
        with zipfile.ZipFile(tmp_path / seed_zip.ZIP_NAME) as zf:
            assert zf.read("big.bin") == _BIG


def _tar(corpus, members):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    (corpus / seed_tar.TAR_NAME).write_bytes(gzip.compress(buf.getvalue()))


class TestTarMemberCap:
    """``seeds.tar.gz`` read every member whole (408 KB tar -> 400 MB)."""

    def test_regression_live_skips_oversized(self, tmp_path, capped):
        _tar(tmp_path, {"seeds/00/big": _BIG, "seeds/00/a": _SMALL})

        assert [d for d, _p in seed_tar.live(tmp_path)] == [_SMALL]

    def test_member_at_cap_is_read(self, tmp_path, capped):
        """Falsification: exactly MAX_MEMBER_BYTES is a seed."""
        edge = b"e" * _CAP
        _tar(tmp_path, {"seeds/00/edge": edge})

        assert [d for d, _p in seed_tar.live(tmp_path)] == [edge]


# ── state unpickler allocation ────────────────────────────────────────

_MID = 64 << 20


def _sized(ctor: str, n: int) -> bytes:
    """GLOBAL builtins.<ctor>, MARK, LONG n, TUPLE, REDUCE, STOP."""
    return b"\x80\x04cbuiltins\n%s\n(L%dL\ntR." % (ctor.encode(), n)


class TestSizedConstructor:
    """``bytearray(n)`` / ``bytes(n)`` from a tampered pickle allocated n zero bytes."""

    @pytest.mark.parametrize("ctor", ["bytearray", "bytes"])
    def test_regression_mid_size_rejected(self, ctor):
        tracemalloc.start()
        try:
            with pytest.raises(UnsafeStateError):
                safe_load(io.BytesIO(_sized(ctor, _MID)))
            _cur, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        assert peak < _MID

    def test_regression_state_file_ignored(self, tmp_path):
        (tmp_path / "state.pkl.gz").write_bytes(gzip.compress(_sized("bytearray", _MID)))

        assert StateStore(tmp_path).load() == {}

    @pytest.mark.parametrize("protocol", [3, 4, 5])
    def test_content_round_trips(self, protocol):
        """Falsification: a bytearray still loads (3 and 4 call ``bytearray(bytes)``)."""
        value = {"a": bytearray(b"\x00\xffab")}

        assert safe_load(io.BytesIO(pickle.dumps(value, protocol=protocol))) == value

    def test_negative_size_rejected(self):
        """Adversarial: a negative count is a size too."""
        with pytest.raises(UnsafeStateError):
            safe_load(io.BytesIO(_sized("bytearray", -1)))
