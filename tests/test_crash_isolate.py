"""Tests for services/crash_isolate.py -- fuzz-time FIC over the field map."""

from __future__ import annotations

import struct
import zlib
from types import SimpleNamespace
from unittest.mock import patch

from fuzzer_tool.cli import commands
from fuzzer_tool.core.crash_metadata import CrashMetadata
from fuzzer_tool.core.field_map import map_fields
from fuzzer_tool.services import crash_isolate as ci


def _png(width=1, height=1, depth=8, ctype=2) -> bytes:
    def chunk(tag: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", zlib.crc32(tag + body))

    ihdr = struct.pack(">IIBBBBB", width, height, depth, ctype, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IEND", b"")


def _crashes(data: bytes) -> bool:
    # synthetic bug: palette (3) with 16-bit depth
    return data[24] == 16 and data[25] == 3


def _replay(data: bytes) -> tuple[int, str]:
    return (-6, "") if _crashes(data) else (0, "")


class TestFieldsFromSpans:
    def test_png_ihdr_fields_selected(self):
        spans = map_fields(_png()).spans
        fields = ci.fields_from_spans(spans)
        names = {f.name for f in fields}
        assert {"IHDR[0].width", "IHDR[0].bit_depth", "IHDR[0].color_type"} <= names
        assert all(1 <= f.size <= 8 for f in fields)
        ordered = sorted(fields, key=lambda f: f.offset)
        assert all(a.end <= b.offset for a, b in zip(ordered, ordered[1:], strict=False))
        assert len(fields) <= ci.MAX_FIELDS

    def test_unrecognised_format_has_no_fields(self):
        assert ci.fields_from_spans(map_fields(b"just some text").spans) == []


class TestIsolateCrash:
    def test_png_pair_isolated(self):
        data = _png(depth=16, ctype=3)
        fields = ci.fields_from_spans(map_fields(data).spans)
        schema, status = ci.isolate_crash(data, fields, _replay, -6)
        assert status == "isolated"
        assert schema == {"IHDR[0].bit_depth": 16, "IHDR[0].color_type": 3}

    def test_probe_budget_respected(self):
        data = _png(depth=16, ctype=3)
        fields = ci.fields_from_spans(map_fields(data).spans)
        n = []
        ci.isolate_crash(data, fields, lambda d: n.append(1) or _replay(d), -6)
        assert 0 < len(n) <= ci.MAX_PROBES

    def test_same_failure_requires_matching_returncode(self):
        data = _png(depth=16, ctype=3)
        fields = ci.fields_from_spans(map_fields(data).spans)
        _, status = ci.isolate_crash(data, fields, _replay, -11)  # crash is -6
        assert status == "not_failing"

    def test_sanitizer_must_appear_in_stderr(self):
        data = _png(depth=16, ctype=3)
        fields = ci.fields_from_spans(map_fields(data).spans)
        rep = lambda d: (1, "==1==ERROR: AddressSanitizer: x") if _crashes(d) else (0, "")  # noqa: E731
        assert ci.isolate_crash(data, fields, rep, 1, sanitizer="AddressSanitizer")[1] == "isolated"
        assert (
            ci.isolate_crash(data, fields, rep, 1, sanitizer="MemorySanitizer")[1] == "not_failing"
        )

    def test_no_fields_and_oversize(self):
        assert ci.isolate_crash(b"x", [], _replay, -6) == ({}, "no_fields")
        big = _png() + b"\0" * ci.MAX_INPUT_BYTES
        fields = ci.fields_from_spans(map_fields(_png()).spans)
        assert ci.isolate_crash(big, fields, _replay, -6) == ({}, "too_large")


class TestForFuzzer:
    def _f(self, **kw):
        base = dict(
            _inprocess_runner=None,
            file_mode=False,
            target="/t",
            timeout=1.0,
            target_args=[],
            _tmp_dir="/tmp",
        )
        base.update(kw)
        return SimpleNamespace(**base)

    def test_fills_metadata_via_stdin_replay(self):
        meta = CrashMetadata()
        data = _png(depth=16, ctype=3)
        with patch(
            "fuzzer_tool.adapters.process.run_target_stdin",
            side_effect=lambda t, d, to, env=None: (*_replay(d), 1),
        ):
            ci.isolate_for_fuzzer(self._f(), meta, data, -6)
        assert meta.failure_schema == {"IHDR[0].bit_depth": 16, "IHDR[0].color_type": 3}
        assert meta.failure_schema_status == "isolated"
        d = meta.to_dict()
        assert d["failure_schema"] == meta.failure_schema
        assert "IHDR[0].bit_depth = 16 (0x10)" in meta.format_sidecar()

    def test_file_mode_uses_file_runner(self):
        meta = CrashMetadata()
        data = _png(depth=16, ctype=3)
        with patch(
            "fuzzer_tool.adapters.process.run_target_file",
            side_effect=lambda t, d, to, tmp, args, env=None: (*_replay(d), 1),
        ) as m:
            ci.isolate_for_fuzzer(self._f(file_mode=True), meta, data, -6)
        assert m.called and meta.failure_schema_status == "isolated"

    def test_inprocess_runner_skipped(self):
        meta = CrashMetadata()
        ci.isolate_for_fuzzer(self._f(_inprocess_runner=object()), meta, _png(), -6)
        assert meta.failure_schema_status == "unsupported_runner" and not meta.failure_schema

    def test_errors_never_propagate(self):
        meta = CrashMetadata()
        with patch("fuzzer_tool.adapters.process.run_target_stdin", side_effect=OSError("boom")):
            ci.isolate_for_fuzzer(self._f(), meta, _png(depth=16, ctype=3), -6)
        assert meta.failure_schema == {}


def test_flag_is_in_hail_mary_list():
    assert "isolate_crash_fields" in commands._HAIL_MARY_FLAGS
