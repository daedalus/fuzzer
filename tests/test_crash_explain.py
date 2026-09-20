"""Crash explanation, static half: baseline pick and field map annotation.

For a novel crash the fuzzer now records which input the crash was mutated
from (the baseline) and which named fields of the crashing input differ from
it. The causal search (which changed field actually triggers the crash) is a
later step; nothing here executes the target.
"""

from __future__ import annotations

import json
import logging
import struct
import types
import zlib
from pathlib import Path

import pytest

from fuzzer_tool.adapters.filesystem import hash_data
from fuzzer_tool.core.crash_metadata import MAX_TXT_ROWS, CrashMetadata
from fuzzer_tool.services import corpus_manager as cm_service
from fuzzer_tool.services import crash_explain as ce
from fuzzer_tool.services.corpus_manager import CorpusManager
from fuzzer_tool.services.crash_explain import (
    MAX_EXPLAIN_BYTES,
    Baseline,
    BaselineSource,
    explain_fields,
    pick_baseline,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
U32_MAX = 0xFFFFFFFF
ASAN_STDERR = (
    "==1234==ERROR: AddressSanitizer: heap-buffer-overflow on address 0xdeadbeef\n"
    "    #0 0x4011a0 in parse_header /src/target.c:42\n"
    "    #1 0x401300 in main /src/target.c:99\n"
)
IHDR_WIDTH_OFF = 16


def _png_chunk(ctype: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body)) + ctype + body + struct.pack(">I", zlib.crc32(ctype + body))


def _png(width: int = 16, idat: bytes = b"\x78\x9c\x03\x00\x00\x00\x00\x01") -> bytes:
    ihdr = struct.pack(">IIBBBBB", width, 8, 8, 2, 0, 0, 0)
    return (
        PNG_MAGIC + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", idat) + _png_chunk(b"IEND", b"")
    )


def _patch(data: bytes, offset: int, new: bytes) -> bytes:
    return data[:offset] + new + data[offset + len(new) :]


def _rows(result) -> dict:
    return {r["name"]: r for r in result.rows}


def _seed_on_disk(corpus_dir: Path, data: bytes) -> str:
    h = hash_data(data)
    dest = corpus_dir / "seeds" / h[:2] / f"id_{h}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return h


class TestPickBaseline:
    def _pick(self, crash: bytes, **kw) -> Baseline:
        args = {
            "parent": None,
            "parent_hash": "",
            "crash_hashes": set(),
            "corpus_dir": None,
            "corpus": [],
            "nearest_label": "",
        }
        args.update(kw)
        return pick_baseline(crash, **args)

    def test_in_memory_parent_wins(self):
        parent = _png(width=16)
        base = self._pick(_png(width=99), parent=parent, parent_hash=hash_data(parent))

        assert base.source is BaselineSource.PARENT
        assert base.data == parent
        assert base.hash == hash_data(parent)

    def test_parent_is_rehydrated_from_disk_by_hash(self, tmp_path):
        parent = _png(width=16)
        h = _seed_on_disk(tmp_path, parent)

        base = self._pick(_png(width=99), parent_hash=h, corpus_dir=tmp_path)

        assert base.source is BaselineSource.DISK
        assert base.data == parent

    def test_nearest_is_the_last_resort(self):
        corpus = [b"zero", _png(width=16)]
        base = self._pick(_png(width=99), corpus=corpus, nearest_label="seed_1")

        assert base.source is BaselineSource.NEAREST
        assert base.data == corpus[1]

    def test_nothing_available_is_none(self):
        base = self._pick(_png())
        assert base == Baseline(None, BaselineSource.NONE, "")

    # -- falsification: a baseline that cannot explain anything is refused --

    def test_parent_identical_to_crash_is_refused(self):
        crash = _png()
        base = self._pick(crash, parent=crash, parent_hash=hash_data(crash))
        assert base.source is BaselineSource.NONE

    def test_parent_known_to_crash_is_refused(self):
        parent = _png(width=16)
        h = hash_data(parent)
        base = self._pick(_png(width=99), parent=parent, parent_hash=h, crash_hashes={h})
        assert base.source is BaselineSource.NONE

    def test_refused_parent_falls_through_to_nearest(self):
        crash = _png(width=99)
        corpus = [_png(width=16)]
        base = self._pick(crash, parent=crash, corpus=corpus, nearest_label="seed_0")
        assert base.source is BaselineSource.NEAREST

    def test_nearest_identical_to_crash_is_refused(self):
        crash = _png()
        base = self._pick(crash, corpus=[crash], nearest_label="seed_0")
        assert base.source is BaselineSource.NONE

    # -- adversarial: the hash and the label come from files on disk --------

    @pytest.mark.parametrize(
        "bad", ["../../etc/passwd", "zz" * 8, "abc", "", "0123456789abcdef0", "id_0123456789abcd"]
    )
    def test_malformed_hash_never_touches_disk(self, tmp_path, monkeypatch, bad):
        def boom(*a, **kw):
            raise AssertionError("rehydrate_by_hash reached with a malformed hash")

        monkeypatch.setattr(ce, "rehydrate_by_hash", boom)
        base = self._pick(_png(), parent_hash=bad, corpus_dir=tmp_path)
        assert base.source is BaselineSource.NONE

    @pytest.mark.parametrize(
        "label", ["seed_99", "seed_-1", "seed_x", "", "seed_", "99", "seed_0_1"]
    )
    def test_malformed_nearest_label_is_ignored(self, label):
        base = self._pick(_png(width=99), corpus=[_png(width=16)], nearest_label=label)
        assert base.source is BaselineSource.NONE

    def test_valid_hash_absent_from_disk_is_none(self, tmp_path):
        base = self._pick(_png(), parent_hash="0123456789abcdef", corpus_dir=tmp_path)
        assert base.source is BaselineSource.NONE


class TestExplainFields:
    def test_changed_field_is_named_with_old_and_new_value(self):
        base = _png(width=16)
        crash = _patch(base, IHDR_WIDTH_OFF, struct.pack(">I", U32_MAX))

        result = explain_fields(crash, base)
        row = _rows(result)["IHDR[0].width"]

        assert result.fmt == "png"
        assert row["changed"] is True
        assert row["value"] == "0xffffffff"
        assert row["baseline"] == "0x00000010"
        assert (row["offset"], row["width"], row["kind"]) == (IHDR_WIDTH_OFF, 4, "value")

    def test_only_the_edited_field_is_changed(self):
        base = _png(width=16)
        crash = _patch(base, IHDR_WIDTH_OFF, struct.pack(">I", U32_MAX))

        changed = [r["name"] for r in explain_fields(crash, base).rows if r["changed"]]
        assert changed == ["IHDR[0].width"]

    def test_length_change_flags_the_field_without_a_paired_old_value(self):
        base = _png()
        at = 8 + 25 + 8 + 2  # two bytes into IDAT data
        crash = base[:at] + b"\xde\xad\xbe" + base[at:]

        rows = _rows(explain_fields(crash, base))

        assert rows["IDAT[1].data"]["changed"] is True
        assert rows["IDAT[1].data"]["baseline"] is None

    def test_fields_shifted_by_a_longer_chunk_are_not_flagged(self):
        base = _png(idat=b"\x01\x02\x03\x04")
        crash = _png(idat=b"\x01\x02\x03\x04\x05\x06\x07")

        rows = _rows(explain_fields(crash, base))

        assert rows["IDAT[1].length"]["changed"] is True
        assert rows["IDAT[1].data"]["changed"] is True
        # IEND moved 3 bytes later but is byte-identical.
        assert rows["IEND[2].length"]["changed"] is False
        assert rows["IEND[2].crc"]["changed"] is False

    def test_deletion_gets_its_own_row_with_the_lost_bytes(self):
        base = _png(idat=b"\x01\x02\x03\x04\x05\x06")
        at = 8 + 25 + 8 + 2
        crash = base[:at] + base[at + 2 :]

        rows = _rows(explain_fields(crash, base))

        assert rows["deleted"]["width"] == 0
        assert rows["deleted"]["offset"] == at
        assert rows["deleted"]["baseline"] == "0304"
        assert rows["IDAT[1].data"]["changed"] is True

    def test_unknown_format_reports_changed_runs(self):
        result = explain_fields(b"AAAAXYAA", b"AAAAAAAA")

        assert result.fmt == ""
        assert len(result.rows) == 1
        row = result.rows[0]
        assert (row["offset"], row["width"], row["kind"]) == (4, 2, "unknown")
        assert row["value"] == "5859"
        assert row["baseline"] == "4141"
        assert row["changed"] is True

    def test_no_baseline_leaves_change_unknown(self):
        result = explain_fields(_png(), None)

        assert result.fmt == "png"
        assert result.rows
        assert all(r["changed"] is None for r in result.rows)

    def test_unknown_format_without_baseline_has_no_rows(self):
        assert explain_fields(b"just bytes", None).rows == []

    # -- falsification -----------------------------------------------------

    def test_identical_input_flags_nothing(self):
        data = _png()
        rows = explain_fields(data, data).rows

        assert rows
        assert not any(r["changed"] for r in rows)
        assert "deleted" not in {r["name"] for r in rows}

    def test_flags_track_content_not_position(self):
        # Same edit in a different place flags a different field.
        base = _png(width=16)
        height_off = IHDR_WIDTH_OFF + 4
        crash = _patch(base, height_off, struct.pack(">I", U32_MAX))

        changed = [r["name"] for r in explain_fields(crash, base).rows if r["changed"]]
        assert changed == ["IHDR[0].height"]

    # -- adversarial --------------------------------------------------------

    def test_truncated_crash_is_explained(self):
        base = _png()
        crash = base[:-6]

        result = explain_fields(crash, base)

        assert any(r["name"] == "deleted" for r in result.rows)
        assert result.fmt == "png"

    def test_oversize_input_is_skipped(self):
        big = b"\x00" * (MAX_EXPLAIN_BYTES + 1)
        assert explain_fields(big, big[:-1] + b"\x01") == ce.ExplainFields("", [])

    def test_empty_crash_and_baseline(self):
        assert explain_fields(b"", b"").rows == []
        assert explain_fields(b"", b"abc").rows[0]["name"] == "deleted"
        assert explain_fields(b"abc", b"").rows[0]["changed"] is True

    def test_inconsistent_alignment_leaves_change_unknown(self, monkeypatch):
        # An edit script that does not account for the crash's bytes must
        # not be read as "nothing changed".
        monkeypatch.setattr(ce, "levenshtein_align", lambda a, b: [("match", 0, b"")])

        result = explain_fields(_png(width=99), _png(width=16))

        assert result.rows
        assert all(r["changed"] is None for r in result.rows)

    def test_rows_are_json_serialisable(self):
        base = _png()
        crash = _patch(base, IHDR_WIDTH_OFF, b"\xff\xff\xff\xff")
        json.dumps(explain_fields(crash, base).rows)


class TestSidecar:
    def _meta(self, base: bytes, crash: bytes) -> CrashMetadata:
        meta = CrashMetadata(parent_seed_hash=hash_data(base))
        ce.explain_static(
            meta,
            crash,
            parent=base,
            parent_hash=hash_data(base),
            crash_hashes=set(),
            corpus_dir=None,
            corpus=[],
            nearest_label="",
        )
        return meta

    def test_metadata_carries_baseline_and_fields(self):
        base = _png()
        crash = _patch(base, IHDR_WIDTH_OFF, struct.pack(">I", U32_MAX))

        meta = self._meta(base, crash)

        assert meta.baseline_source == "parent"
        assert meta.baseline_hash == hash_data(base)
        assert meta.field_format == "png"
        assert meta.fields

    def test_txt_lists_changed_fields_and_counts_the_rest(self):
        base = _png()
        crash = _patch(base, IHDR_WIDTH_OFF, struct.pack(">I", U32_MAX))

        txt = self._meta(base, crash).format_sidecar()

        assert "=== fields (png; baseline: parent" in txt
        assert "IHDR[0].width @0x10 +4 value 0xffffffff (was 0x00000010)" in txt
        assert "IHDR[0].height" not in txt
        assert "unchanged fields not shown" in txt

    def test_json_has_baseline_and_fields(self):
        base = _png()
        crash = _patch(base, IHDR_WIDTH_OFF, struct.pack(">I", U32_MAX))

        d = self._meta(base, crash).to_dict()

        assert d["baseline"] == {"source": "parent", "hash": hash_data(base)}
        assert d["field_format"] == "png"
        assert any(r["name"] == "IHDR[0].width" and r["changed"] for r in d["fields"])

    def test_no_fields_no_section(self):
        meta = CrashMetadata()
        assert "=== fields" not in meta.format_sidecar()
        assert meta.to_dict()["fields"] == []

    def test_without_baseline_first_fields_are_listed(self):
        meta = CrashMetadata()
        ce.explain_static(
            meta,
            _png(),
            parent=None,
            parent_hash="",
            crash_hashes=set(),
            corpus_dir=None,
            corpus=[],
            nearest_label="",
        )
        txt = meta.format_sidecar()

        assert meta.baseline_source == "none"
        assert "baseline: none" in txt
        assert "IHDR[0].width" in txt

    def test_txt_caps_changed_rows(self):
        meta = CrashMetadata()
        meta.baseline_source = "parent"
        meta.fields = [
            {
                "offset": i * 2,
                "width": 1,
                "name": f"diff@{i}",
                "kind": "unknown",
                "value": "01",
                "baseline": "00",
                "changed": True,
            }
            for i in range(200)
        ]

        txt = meta.format_sidecar()

        assert txt.count("diff@") == MAX_TXT_ROWS
        assert f"+{200 - MAX_TXT_ROWS} more changed fields" in txt


class _Fuzzer:
    """Minimal fuzzer surface read by CorpusManager.save_crash."""

    def __init__(self, tmp_path: Path, parent: bytes | None):
        self.corpus_dir = tmp_path / "corpus"
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        self.crashes_dir = tmp_path / "crashes"
        self.crashes_dir.mkdir(parents=True, exist_ok=True)
        self.corpus = [b"seed-a" * 10, parent] if parent else [b"seed-a" * 10]
        self.seed_meta: dict = {}
        self.exec_count = 100
        self.target = "/bin/true"
        self._last_ops_used: list = []
        self._stats = types.SimpleNamespace(format_elapsed=lambda: "1s")
        self._last_regs = None
        self.ptrace_cov = None
        self._last_fault_addr = None
        self.crash_hashes: set[str] = set()
        self.crash_sigs: dict[str, int] = {}
        self.crash_frames: dict = {}
        self.crash_min_sizes: dict = {}
        self.save_smaller = False
        self.crash_blocklist: set = set()
        self.crash_allowlist: set = set()
        self.seen_hashes: set[str] = set()
        self.irreplaceable_hashes: set[str] = set()
        self.bloom = None
        if parent is not None:
            self._last_parent_seed = parent


@pytest.fixture(autouse=True)
def _no_gdb(monkeypatch):
    monkeypatch.setattr(cm_service, "_gdb_crash_replay", lambda f, data, rc: "")


class TestWiring:
    def test_novel_crash_sidecars_carry_the_explanation(self, tmp_path):
        parent = _png(width=16)
        crash = _patch(parent, IHDR_WIDTH_OFF, struct.pack(">I", U32_MAX))
        f = _Fuzzer(tmp_path, parent)

        name = CorpusManager(f).save_crash(crash, -11, ASAN_STDERR)

        d = json.loads((f.crashes_dir / f"{name}.json").read_text())
        txt = (f.crashes_dir / f"{name}.txt").read_text()
        assert d["baseline"]["source"] == "parent"
        assert d["field_format"] == "png"
        assert [r["name"] for r in d["fields"] if r["changed"]] == ["IHDR[0].width"]
        assert "IHDR[0].width @0x10 +4" in txt

    def test_parent_hash_matches_the_recorded_parent_seed(self, tmp_path):
        parent = _png(width=16)
        crash = _patch(parent, IHDR_WIDTH_OFF, struct.pack(">I", U32_MAX))
        f = _Fuzzer(tmp_path, parent)

        name = CorpusManager(f).save_crash(crash, -11, ASAN_STDERR)

        d = json.loads((f.crashes_dir / f"{name}.json").read_text())
        assert d["baseline"]["hash"] == d["parent_seed_hash"] == hash_data(parent)

    def test_repeat_signature_is_not_explained(self, tmp_path, monkeypatch):
        parent = _png(width=16)
        f = _Fuzzer(tmp_path, parent)
        mgr = CorpusManager(f)
        mgr.save_crash(_patch(parent, IHDR_WIDTH_OFF, b"\xff\xff\xff\xff"), -11, ASAN_STDERR)

        calls = []
        monkeypatch.setattr(cm_service, "explain_static", lambda *a, **kw: calls.append(1))
        again = mgr.save_crash(
            _patch(parent, IHDR_WIDTH_OFF, b"\xff\xff\xff\xfe"), -11, ASAN_STDERR
        )

        assert again is False
        assert calls == []

    def test_no_parent_falls_back_to_nearest(self, tmp_path):
        near = _png(width=16)
        f = _Fuzzer(tmp_path, near)
        del f._last_parent_seed
        crash = _patch(near, IHDR_WIDTH_OFF, struct.pack(">I", U32_MAX))

        name = CorpusManager(f).save_crash(crash, -11, ASAN_STDERR)

        d = json.loads((f.crashes_dir / f"{name}.json").read_text())
        assert d["baseline"]["source"] == "nearest"
        assert [r["name"] for r in d["fields"] if r["changed"]] == ["IHDR[0].width"]

    def test_unformatted_input_gets_changed_runs(self, tmp_path):
        parent = b"AAAAAAAAAAAAAAAA"
        crash = b"AAAAAAAAXXAAAAAA"
        f = _Fuzzer(tmp_path, parent)

        name = CorpusManager(f).save_crash(crash, -11, ASAN_STDERR)

        d = json.loads((f.crashes_dir / f"{name}.json").read_text())
        assert d["field_format"] == ""
        assert [(r["offset"], r["width"], r["baseline"]) for r in d["fields"]] == [(8, 2, "4141")]

    def test_explain_failure_never_loses_the_crash(self, tmp_path, monkeypatch, caplog):
        parent = _png(width=16)
        f = _Fuzzer(tmp_path, parent)

        def boom(*a, **kw):
            raise ValueError("walker bug")

        monkeypatch.setattr(cm_service, "explain_static", boom)
        with caplog.at_level(logging.WARNING):
            name = CorpusManager(f).save_crash(
                _patch(parent, IHDR_WIDTH_OFF, b"\xff\xff\xff\xff"), -11, ASAN_STDERR
            )

        assert name
        assert list(f.crashes_dir.glob("*.bin"))
        assert "walker bug" in caplog.text

    def test_crashing_seed_is_still_saved(self, tmp_path):
        parent = _png(width=16)
        f = _Fuzzer(tmp_path, parent)
        crash = _patch(parent, IHDR_WIDTH_OFF, b"\xff\xff\xff\xff")

        CorpusManager(f).save_crash(crash, -11, ASAN_STDERR)

        h = hash_data(crash)
        assert (f.corpus_dir / "seeds" / "crashing" / h[:2] / f"id_{h}").is_file()
