"""Tests for the ARMovie/RPL mutator (`rpl_chunk_mutate`)."""

from __future__ import annotations

import sys

import pytest

from fuzzer_tool.core.mutations.rpl import (
    AUDIO_BITS,
    AUDIO_FORMATS,
    COUNT_EDGES,
    FPS_EDGES,
    INT_EDGES,
    ROW_EDGES,
    VIDEO_FORMATS,
    Op,
    mutate_rpl,
    parse_rpl,
)
from fuzzer_tool.core.operator_registry import format_gate_matches

from .support.scripted_rng import ScriptedRng

CATALOG_LINE = 17


def _rpl(rows: list[str], video: str = "0", gap: str = "") -> bytes:
    """Header (20 lines, 21 with video) + gap + index rows; catalog points at the rows."""
    lines = ["ARMovie", "t", "c", "a", video, "0", "0", "0", "1", "1", "1", "1", "8"]
    lines += ["0", str(len(rows) - 1), "0", "0", "@", "0", "0"]
    if video != "0":
        lines.append("0")

    head = "".join(f"{s}\n" for s in lines)
    base = len(head) - 1  # "@" is one byte; the offset's digits replace it
    off = base + len(str(base + len(str(base))))
    head = head.replace("@", str(off + len(gap)))
    assert len(head) == off
    return (head + gap + "".join(f"{r}\n" for r in rows)).encode()


def _catalog(data: bytes) -> int:
    return int(data.split(b"\n")[CATALOG_LINE])


SAMPLE = _rpl(["100 , 10 ; 20", "130 , 11 ; 21"])


class TestParse:
    def test_header_and_rows(self):
        hdr, gap, rows = parse_rpl(SAMPLE)
        assert len(hdr) == 20
        assert gap == b""
        assert rows == [b"100 , 10 ; 20\n", b"130 , 11 ; 21\n"]

    def test_video_header_has_21_lines(self):
        hdr, _, _ = parse_rpl(_rpl(["1 , 1 ; 1"], video="124"))
        assert len(hdr) == 21

    def test_not_rpl(self):
        assert parse_rpl(b"ARMovi\n" + bytes(40)) is None

    # Adversarial: a truncated header is declined, not mis-split.
    def test_short_header(self):
        assert parse_rpl(b"ARMovie\nt\nc\n") is None

    def test_catalog_points_at_rows(self):
        assert SAMPLE[_catalog(SAMPLE) :].startswith(b"100 ,")


class TestMutate:
    def test_video_format(self):
        out = mutate_rpl(SAMPLE, rng=ScriptedRng(randints=[Op.VIDEO_FMT], choice_idxs=[2]))
        assert out.split(b"\n")[4] == VIDEO_FORMATS[2]

    def test_fps(self):
        out = mutate_rpl(SAMPLE, rng=ScriptedRng(randints=[Op.FPS], choice_idxs=[0]))
        assert out.split(b"\n")[8] == FPS_EDGES[0]

    def test_audio_format(self):
        out = mutate_rpl(SAMPLE, rng=ScriptedRng(randints=[Op.AUDIO_FMT], choice_idxs=[1]))
        assert out.split(b"\n")[9] == AUDIO_FORMATS[1]

    def test_audio_bits(self):
        out = mutate_rpl(SAMPLE, rng=ScriptedRng(randints=[Op.AUDIO_BITS], choice_idxs=[3]))
        assert out.split(b"\n")[12] == AUDIO_BITS[3]

    def test_chunk_count(self):
        out = mutate_rpl(SAMPLE, rng=ScriptedRng(randints=[Op.CHUNK_COUNT], choice_idxs=[3]))
        assert out.split(b"\n")[14] == COUNT_EDGES[3]

    # Header length changes: the catalog offset is repaired to still hit the rows.
    def test_catalog_repaired_after_resize(self):
        out = mutate_rpl(SAMPLE, rng=ScriptedRng(randints=[Op.INT_FIELD, 5 - 4], choice_idxs=[4]))
        assert out.split(b"\n")[5] == INT_EDGES[4]
        assert len(out) != len(SAMPLE)
        assert out[_catalog(out) :].startswith(b"100 ,")

    # Falsification: editing the catalog line itself is not undone by the repair.
    def test_catalog_field_not_repaired(self):
        out = mutate_rpl(
            SAMPLE, rng=ScriptedRng(randints=[Op.INT_FIELD, CATALOG_LINE - 4], choice_idxs=[0])
        )
        assert out.split(b"\n")[CATALOG_LINE] == INT_EDGES[0]

    # The 9.0.2-only overflow: offset + video_size in the index loop.
    def test_row_field(self):
        out = mutate_rpl(SAMPLE, rng=ScriptedRng(randints=[Op.ROW_FIELD, 1, 0], choice_idxs=[2]))
        assert parse_rpl(out)[2][1] == ROW_EDGES[2] + b" , 11 ; 21\n"

    def test_row_dup(self):
        out = mutate_rpl(SAMPLE, rng=ScriptedRng(randints=[Op.DUP, 0]))
        assert parse_rpl(out)[2] == [b"100 , 10 ; 20\n"] * 2 + [b"130 , 11 ; 21\n"]

    # Falsification: row ops decline with no index rows.
    @pytest.mark.parametrize("op", [Op.ROW_FIELD, Op.DUP, Op.DROP, Op.SWAP])
    def test_row_ops_decline_without_rows(self, op):
        data = SAMPLE[: _catalog(SAMPLE)]
        assert mutate_rpl(data, rng=ScriptedRng(randints=[op, 0, 0])) is None

    def test_non_rpl_declined(self):
        assert mutate_rpl(b"\x00" * 64, rng=ScriptedRng()) is None

    def test_max_len_truncates(self):
        full = mutate_rpl(SAMPLE, rng=ScriptedRng(randints=[Op.FPS], choice_idxs=[0]))
        out = mutate_rpl(SAMPLE, max_len=30, rng=ScriptedRng(randints=[Op.FPS], choice_idxs=[0]))
        assert out == full[:30]
        assert out != SAMPLE[:30]

    # Rows are located by the catalog offset; bytes before it stay untouched.
    def test_regression_rpl_catalog_after_gap(self):
        data = _rpl(["100 , 10 ; 20"], gap="JUNK 1 , 2 ; 3\n")
        assert parse_rpl(data)[1:] == (b"JUNK 1 , 2 ; 3\n", [b"100 , 10 ; 20\n"])

        out = mutate_rpl(data, rng=ScriptedRng(randints=[Op.ROW_FIELD, 0, 1], choice_idxs=[1]))
        assert out[_catalog(out) :] == b"100 , -1 ; 20\n"
        assert b"JUNK 1 , 2 ; 3\n" in out

    # Gap survives a header resize: catalog shifts by the header delta only.
    def test_regression_rpl_gap_resize(self):
        data = _rpl(["100 , 10 ; 20"], gap="JUNK\n")
        out = mutate_rpl(data, rng=ScriptedRng(randints=[Op.INT_FIELD, 1], choice_idxs=[4]))
        assert out[_catalog(out) :] == b"100 , 10 ; 20\n"

    # Toggling video on adds the key frame list line; rows stay reachable.
    @pytest.mark.parametrize(("video", "idx", "n_hdr"), [("0", 2, 21), ("124", 0, 20)])
    def test_regression_rpl_video_toggle(self, video, idx, n_hdr):
        data = _rpl(["100 , 10 ; 20"], video=video)
        out = mutate_rpl(data, rng=ScriptedRng(randints=[Op.VIDEO_FMT], choice_idxs=[idx]))
        hdr, gap, rows = parse_rpl(out)
        assert (len(hdr), gap, rows) == (n_hdr, b"", [b"100 , 10 ; 20\n"])
        assert out[_catalog(out) :] == b"100 , 10 ; 20\n"

    # DUP/DROP keep "last chunk index" consistent with the row count.
    @pytest.mark.parametrize(("op", "n_rows"), [(Op.DUP, 3), (Op.DROP, 1)])
    def test_regression_rpl_count_follows_rows(self, op, n_rows):
        out = mutate_rpl(SAMPLE, rng=ScriptedRng(randints=[op, 0]))
        assert out.split(b"\n")[14] == str(n_rows - 1).encode()
        rows = parse_rpl(out)[2]
        assert len(rows) == n_rows
        assert out[_catalog(out) :] == b"".join(rows)

    # Falsification: a count that already disagreed with the rows is left alone.
    def test_regression_rpl_stale_count_kept(self):
        lines = SAMPLE.split(b"\n")
        lines[14] = b"7"
        out = mutate_rpl(b"\n".join(lines), rng=ScriptedRng(randints=[Op.DUP, 0]))
        assert out.split(b"\n")[14] == b"7"

    # Adversarial: a catalog offset past int()'s digit limit never raises.
    # The floor limit (640) lets a header still fit the 4 KiB sniff window.
    def test_regression_rpl_huge_catalog_digits(self):
        lines = SAMPLE.split(b"\n")
        lines[CATALOG_LINE] = b"9" * 1000
        data = b"\n".join(lines)

        saved = sys.get_int_max_str_digits()
        sys.set_int_max_str_digits(640)
        try:
            out = mutate_rpl(data, rng=ScriptedRng(randints=[Op.FPS], choice_idxs=[0]))
        finally:
            sys.set_int_max_str_digits(saved)
        assert out.split(b"\n")[CATALOG_LINE] == b"9" * 1000

    # Adversarial: every op on every truncation never raises.
    def test_truncations_never_raise(self):
        for cut in range(len(SAMPLE) + 1):
            for op in Op:
                rng = ScriptedRng(randints=[op, 0, 0, 0], choice_idxs=[0, 0])
                out = mutate_rpl(SAMPLE[:cut], rng=rng)
                assert out is None or out.startswith(b"ARMovie\n")


class TestRegistry:
    def test_sniffer(self):
        assert format_gate_matches("rpl_chunk_mutate", SAMPLE)
        assert not format_gate_matches("rpl_chunk_mutate", b"ARMovie")
