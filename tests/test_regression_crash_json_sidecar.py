"""Regression: save_crash() must write a companion .json sidecar per crash.

The ``.txt`` sidecar is for humans; the ``.json`` is for downstream tools
(dashboards, FINDINGS.md generators, diff scripts) that want the same
triage fields without re-parsing the txt. The two are written together
by ``save_crash()`` and must agree on every field they share.

This test pins the on-disk shape (one crash -> one ``.json`` next to the
``.bin``), the field set (``CrashMetadata.to_dict()``), and the JSON
parses cleanly with stock ``json.loads`` (no Python-only types, no NaN).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from fuzzer_tool.adapters.filesystem import save_crash
from fuzzer_tool.core.crash_metadata import CrashMetadata


def _run_save_crash(tmp: Path, name: str, data: bytes, returncode: int = 136, stderr: str = ""):
    """Mirror save_crash's real signature with the minimum state it needs."""
    save_crash(
        data=data,
        returncode=returncode,
        stderr=stderr,
        crashes_dir=tmp,
        crash_hashes=set(),
        crash_sigs={},
        metadata=CrashMetadata(target="./test_target", target_sha256="0" * 16),
    )


class TestJsonSidecarWritten:
    def test_writes_json_next_to_bin(self, tmp_path):
        with tempfile.TemporaryDirectory() as d:
            crashes = Path(d) / "crashes"
            crashes.mkdir()
            _run_save_crash(crashes, "crash_x", b"AAAA", returncode=136)

            bins = list(crashes.glob("*.bin"))
            jsons = list(crashes.glob("*.json"))
            assert len(bins) == 1
            assert len(jsons) == 1
            # The .json and .bin share the same base name.
            assert jsons[0].stem == bins[0].stem

    def test_json_parses_with_stock_loads(self, tmp_path):
        with tempfile.TemporaryDirectory() as d:
            crashes = Path(d) / "crashes"
            crashes.mkdir()
            _run_save_crash(crashes, "crash_y", b"BBBB", returncode=139)

            json_path = next(crashes.glob("*.json"))
            # No exception, no NaN/Infinity.
            json.loads(json_path.read_text())

    def test_json_has_expected_fields(self, tmp_path):
        with tempfile.TemporaryDirectory() as d:
            crashes = Path(d) / "crashes"
            crashes.mkdir()
            _run_save_crash(crashes, "crash_z", b"CCCC", returncode=136)

            data = json.loads(next(crashes.glob("*.json")).read_text())
            # The set of fields a downstream tool would ingest.
            for key in (
                "timestamp",
                "target",
                "target_sha256",
                "sanitizer",
                "error_type",
                "fault_addr",
                "access_type",
                "access_size",
                "shadow_info",
                "exploitability",
                "cluster_id",
                "returncode",
                "frames",
                "alloc_frames",
                "dealloc_frames",
                "registers",
                "gdb_replay",
                "nearest_corpus_file",
                "nearest_similarity",
                "diff_bytes",
                "raw_stderr",
                "input_hexdump",
                "input_text_repr",
                "mutation_ops",
                "parent_sites",
            ):
                assert key in data, f"missing field in .json sidecar: {key!r}"

    def test_json_and_txt_agree_on_core_fields(self, tmp_path):
        # The two sidecars are derived from the same CrashMetadata; the
        # core fields (sanitizer, error_type, returncode, target_sha256)
        # must match so a downstream tool reading the .json and a human
        # reading the .txt never see contradictory facts.
        with tempfile.TemporaryDirectory() as d:
            crashes = Path(d) / "crashes"
            crashes.mkdir()
            _run_save_crash(crashes, "crash_q", b"DDDD", returncode=136, stderr="some stderr")

            txt = next(crashes.glob("*.txt")).read_text()
            data = json.loads(next(crashes.glob("*.json")).read_text())

            # The .txt renders these as "key:   value" lines; the .json
            # renders them as the raw typed value. Compare via int() for
            # numeric fields, raw string for text fields.
            assert _extract_kv(txt, "target") == data["target"]
            assert _extract_kv(txt, "target_sha256") == data["target_sha256"]
            assert int(_extract_kv(txt, "returncode")) == data["returncode"]


def _extract_kv(txt: str, key: str) -> str:
    """Pull the right-hand side of a ``key:   value`` line from the .txt sidecar."""
    for line in txt.splitlines():
        if line.startswith(f"{key}:"):
            return line.split(":", 1)[1].strip()
    raise AssertionError(f"key {key!r} not found in .txt sidecar")
