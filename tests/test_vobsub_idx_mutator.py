"""Tests for the VobSub index mutator (`vobsub_idx_mutate`)."""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

from fuzzer_tool.core.mutations.vobsub_idx import (
    FILEPOS_EDGES,
    ID_LINES,
    mutate_vobsub_idx,
    sniff_vobsub_idx,
)
from fuzzer_tool.core.operator_categories import OPERATOR_CATEGORIES
from fuzzer_tool.core.operator_registry import format_gate_matches

from .support.scripted_rng import ScriptedRng

FFMPEG = shutil.which("ffmpeg")

IDX = (
    b"# VobSub index file, v7 (20:34:44)\n"
    b"\n"
    b"id: en, index: 0\n"
    b"# alt: English\n"
    b"\n"
    b"id: de, index: 1\n"
    b"# alt: Deutsch\n"
    b"\n"
    b"timestamp: 0:00:01:000, filepos: 000000800\n"
    b"timestamp: 0:00:04:047, filepos: 000001234\n"
    b"timestamp: 0:00:07:101, filepos: 000001730\n"
)
NO_TS_IDX = b"# VobSub index file, v7\n\nid: en, index: 0\n"


def _ids(data: bytes) -> set[bytes]:
    return {m.group(1) for m in ID_LINES.finditer(data)}


def _filepos(data: bytes) -> list[bytes]:
    return [m.group(1) for m in re.finditer(rb"filepos: (\w+)", data)]


class TestSniff:
    def test_header_and_ids(self):
        assert sniff_vobsub_idx(IDX)

    def test_no_header_declines(self):
        assert not sniff_vobsub_idx(b"id: en, index: 0\n" + bytes(60))

    def test_no_id_line_declines(self):
        assert not sniff_vobsub_idx(b"# VobSub index file\n")


class TestMutate:
    def test_inject_ids(self):
        # op 0 (inject): k = 1+draw(0..7) fresh ids, each drawn from the pool.
        out = mutate_vobsub_idx(IDX, rng=ScriptedRng(randints=[0, 0, 0]))
        assert len(_ids(out)) == 3  # one fresh tag beside en/de
        assert _ids(IDX) < _ids(out)
        assert out.startswith(b"# VobSub index file")  # grammar kept

    def test_rewrite_id(self):
        # op 1 (rewrite): one tag re-drawn from the pool.
        out = mutate_vobsub_idx(IDX, rng=ScriptedRng(randints=[1, 0, 0]))
        assert len(_ids(out)) == 2
        assert _ids(out) != _ids(IDX)

    def test_filepos_edge(self):
        # op 2 (filepos), entry 2, edge 4 (2^32-1).
        out = mutate_vobsub_idx(IDX, rng=ScriptedRng(randints=[2, 2, 4]))
        assert _filepos(out)[2] == b"%09x" % FILEPOS_EDGES[4]

    def test_time_edge(self):
        # op 3 (time), entry 1, edge 2 (99:59:59:999).
        out = mutate_vobsub_idx(IDX, rng=ScriptedRng(randints=[3, 1, 2]))
        assert b"timestamp: 99:59:59:999" in out

    def test_id_compounding(self):
        # The CVE trigger is cumulative: repeated injects grow the distinct-id
        # table past the 32-slot vobsub->q[] array. One tag per call: op 0
        # (inject), k 0, pool draw 0.
        cur = IDX
        rng = ScriptedRng(randints=[0] * 600)
        while len(_ids(cur)) < 40:
            out = mutate_vobsub_idx(cur, rng=rng)
            assert out is not None
            cur = out
        assert len(_ids(cur)) >= 40

    # Falsification: a file the demuxer would not probe.
    def test_wrong_header_declines(self):
        assert mutate_vobsub_idx(b"# other\nid: en, index: 0\n", rng=ScriptedRng()) is None

    def test_empty_declines(self):
        assert mutate_vobsub_idx(b"# VobSub index file\n", rng=ScriptedRng()) is None

    # Adversarial: truncated indexes and marker-free noise never raise.
    def test_truncations_never_raise(self):
        rng = ScriptedRng(randints=[0] * 2000)
        for cut in range(len(IDX) + 1):
            out = mutate_vobsub_idx(IDX[:cut], rng=rng)
            assert out is None or isinstance(out, bytes)

    def test_no_timestamps_offer_two_ops(self):
        # Only inject/rewrite are feasible; op draw 3 is past the list end.
        out = mutate_vobsub_idx(NO_TS_IDX, rng=ScriptedRng(randints=[1, 0, 0]))
        assert out is not None and _ids(out) != _ids(NO_TS_IDX)


class TestRegistry:
    def test_gate(self):
        assert format_gate_matches("vobsub_idx_mutate", IDX)
        assert not format_gate_matches("vobsub_idx_mutate", b"id: en, index: 0\n" + bytes(60))

    def test_category(self):
        assert "vobsub_idx_mutate" in OPERATOR_CATEGORIES["format"]


@pytest.mark.skipif(FFMPEG is None, reason="needs ffmpeg")
class TestFfmpeg:
    def test_vobsub_demuxer_accepts(self, tmp_path):
        # FFmpeg's vobsub demuxer probes on the header line and parses ids;
        # the missing .sub sidecar is its expected graceful complaint.
        path = tmp_path / "test.idx"
        path.write_bytes(IDX)
        cmd = [FFMPEG, "-v", "warning", "-i", str(path), "-f", "null", "-"]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert "test.sub" in proc.stderr

    def test_injected_ids_reach_demuxer(self, tmp_path):
        # 40 distinct ids: past the array bound of the fixed version.
        cur = IDX
        rng = ScriptedRng(randints=[0] * 600)
        while len(_ids(cur)) < 40:
            cur = mutate_vobsub_idx(cur, rng=rng)
        path = tmp_path / "grown.idx"
        path.write_bytes(cur)
        cmd = [FFMPEG, "-v", "warning", "-i", str(path), "-f", "null", "-"]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert "grown.sub" in proc.stderr
        assert "memory corruption" not in proc.stderr
