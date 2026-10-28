"""SARIF export of crash sidecars (core/sarif.py, services/sarif_export.py)."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from fuzzer_tool.adapters.filesystem import load_crash_sidecars
from fuzzer_tool.cli.commands import cmd_sarif
from fuzzer_tool.core.sanitizer import SanitizerReport, crash_frames, report_stacks
from fuzzer_tool.core.sarif import (
    MAX_RESULTS,
    SARIF_VERSION,
    SRCROOT,
    build_sarif,
)
from fuzzer_tool.services.sarif_export import export_sarif

SRC = "/work/proj"

# ASAN UAF: frame #0 is the runtime interceptor, #1 is project code.
UAF = (
    "==7==ERROR: AddressSanitizer: heap-use-after-free on address 0x602000000010\n"
    "READ of size 4 at 0x602000000010 thread T0\n"
    "    #0 0x4a1b2c in __interceptor_memcpy /llvm/compiler-rt/asan_interceptors.cpp:22:3\n"
    "    #1 0x4011d6 in parse_chunk /work/proj/src/png.c:120:9\n"
    "    #2 0x401300 in main /work/proj/src/main.c:7\n"
    "    #3 0x7f00 in __libc_start_main (/lib/libc.so.6+0x29d90)\n"
    "\n"
    "freed by thread T0 here:\n"
    "    #0 0x4a0000 in free /llvm/compiler-rt/asan_malloc.cpp:52:3\n"
    "    #1 0x401100 in drop_chunk /work/proj/src/png.c:80:5\n"
    "\n"
    "previously allocated by thread T0 here:\n"
    "    #0 0x4a0100 in malloc /llvm/compiler-rt/asan_malloc.cpp:69:3\n"
    "    #1 0x401050 in new_chunk /work/proj/src/png.c:40:12\n"
    "\n"
    "SUMMARY: AddressSanitizer: heap-use-after-free /work/proj/src/png.c:120:9 in parse_chunk\n"
)

UNSYMBOLIZED = (
    "==7==ERROR: AddressSanitizer: SEGV on unknown address 0x000000000000\n"
    "    #0 0x4011d6  (/bin/target+0x11d6)\n"
    "    #1 0x401300  (/bin/target+0x1300)\n"
)


def _one_frame(path: str) -> str:
    """ASAN report whose only frame sits at *path*:3:1."""
    return (
        "==7==ERROR: AddressSanitizer: SEGV on unknown address 0x0\n"
        f"    #0 0x401000 in f {path}:3:1\n"
    )


def _sidecar(stderr: str, cluster: str = "deadbeef", tier: str = "CRITICAL", **kw) -> dict:
    """Minimal crash .json sidecar, shaped like CrashMetadata.to_dict()."""
    rep = SanitizerReport.parse(stderr)
    d = {
        "sanitizer": rep.sanitizer if rep else "",
        "error_type": rep.error_type if rep else "",
        "fault_addr": rep.fault_addr if rep else "",
        "access_type": rep.access_type if rep else None,
        "access_size": rep.access_size if rep else None,
        "exploitability": tier,
        "cluster_id": cluster,
        "target": "/work/proj/build/png_read",
        "parent_seed_hash": "",
        "mutation_ops": [],
        "raw_stderr": stderr,
        "crash_input": f"crash_1_{cluster}_asan_uaf.bin",
    }
    d.update(kw)
    return d


def _result(sarif: dict, i: int = 0) -> dict:
    return sarif["runs"][0]["results"][i]


def _phys(loc: dict) -> dict:
    return loc["physicalLocation"]


class TestCrashFrames:
    def test_symbolized_frame_keeps_file_line_col(self):
        fr = crash_frames(UAF)
        assert fr[1].func == "parse_chunk"
        assert fr[1].path == "/work/proj/src/png.c"
        assert (fr[1].line, fr[1].col) == (120, 9)

    def test_missing_column_is_zero(self):
        assert crash_frames(UAF)[2].col == 0

    def test_unsymbolized_frame_has_module_no_path(self):
        fr = crash_frames(UNSYMBOLIZED)
        assert fr[0].path == ""
        assert fr[0].module == "target+0x11d6"

    def test_function_with_module_has_no_path(self):
        fr = crash_frames(UAF)[3]
        assert fr.func == "__libc_start_main"
        assert fr.path == ""

    def test_stops_at_next_stack(self):
        # Crash stack has 4 frames; the free/alloc stacks are separate.
        assert len(crash_frames(UAF)) == len(SanitizerReport.parse(UAF).frames)

    def test_regression_names_unchanged(self):
        # Adding locations must not leak into the dedup signature.
        assert SanitizerReport.parse(UAF).frames[1] == "parse_chunk"


class TestReportStacks:
    def test_crash_alloc_free(self):
        st = report_stacks(UAF)
        assert st["crash"][1].func == "parse_chunk"
        assert st["free"][1].func == "drop_chunk"
        assert st["alloc"][1].func == "new_chunk"

    def test_ubsan_without_stack_uses_site(self):
        st = report_stacks("/work/proj/src/a.c:12:5: runtime error: signed integer overflow\n")
        assert st["crash"][0].path == "/work/proj/src/a.c"
        assert (st["crash"][0].line, st["crash"][0].col) == (12, 5)

    def test_no_report_is_empty(self):
        assert report_stacks("plain output\n") == {}


class TestBuildSarif:
    def test_envelope(self):
        s = build_sarif([_sidecar(UAF)], SRC)
        assert s["version"] == SARIF_VERSION
        assert s["runs"][0]["tool"]["driver"]["name"] == "fuzzer-tool"
        assert s["runs"][0]["originalUriBaseIds"][SRCROOT]["uri"] == "file:///work/proj/"

    def test_rule_and_level(self):
        s = build_sarif([_sidecar(UAF)], SRC)
        r = _result(s)
        assert r["ruleId"] == "AddressSanitizer/heap-use-after-free"
        assert r["level"] == "error"
        rule = s["runs"][0]["tool"]["driver"]["rules"][r["ruleIndex"]]
        assert rule["id"] == r["ruleId"]
        assert float(rule["properties"]["security-severity"]) > 0

    def test_primary_location_skips_runtime_frame(self):
        loc = _phys(_result(build_sarif([_sidecar(UAF)], SRC))["locations"][0])
        assert loc["artifactLocation"] == {"uri": "src/png.c", "uriBaseId": SRCROOT}
        assert loc["region"] == {"startLine": 120, "startColumn": 9}

    def test_code_flow_runs_outermost_first(self):
        flow = _result(build_sarif([_sidecar(UAF)], SRC))["codeFlows"][0]
        names = [
            step["location"]["logicalLocations"][0]["name"]
            for step in flow["threadFlows"][0]["locations"]
        ]
        expected = [f.func for f in reversed(crash_frames(UAF))]
        assert names == expected

    def test_stacks_carry_alloc_and_free(self):
        stacks = _result(build_sarif([_sidecar(UAF)], SRC))["stacks"]
        msgs = [st["message"]["text"] for st in stacks]
        assert msgs == ["crash", "free", "alloc"]

    def test_fingerprint_is_cluster(self):
        r = _result(build_sarif([_sidecar(UAF, cluster="cafe0001")], SRC))
        assert r["partialFingerprints"]["crashCluster/v1"] == "cafe0001"

    def test_same_cluster_deduplicated(self):
        s = build_sarif([_sidecar(UAF), _sidecar(UAF)], SRC)
        assert len(s["runs"][0]["results"]) == 1

    def test_message_names_access(self):
        text = _result(build_sarif([_sidecar(UAF)], SRC))["message"]["text"]
        assert "READ of size 4" in text
        assert "parse_chunk" in text

    def test_no_src_root_keeps_absolute_uri(self):
        s = build_sarif([_sidecar(UAF)], "")
        loc = _phys(_result(s)["locations"][0])
        assert loc["artifactLocation"] == {"uri": "file:///work/proj/src/png.c"}
        assert "originalUriBaseIds" not in s["runs"][0]

    def test_severity_orders_results(self):
        low = _sidecar(UAF, cluster="00000001", tier="LOW")
        crit = _sidecar(UAF, cluster="00000002", tier="CRITICAL")
        s = build_sarif([low, crit], SRC)
        assert [r["level"] for r in s["runs"][0]["results"]] == ["error", "note"]


class TestFalsification:
    """Inputs the exporter must not anchor to a source line."""

    def test_unsymbolized_crash_falls_back_to_target(self):
        r = _result(build_sarif([_sidecar(UNSYMBOLIZED)], SRC))
        loc = _phys(r["locations"][0])
        assert loc["artifactLocation"]["uri"] == "build/png_read"
        assert "region" not in loc

    def test_runtime_only_stack_is_not_anchored_to_runtime(self):
        stderr = (
            "==7==ERROR: AddressSanitizer: SEGV on unknown address 0x0\n"
            "    #0 0x4a1b2c in __asan_memcpy /llvm/compiler-rt/asan.cpp:22:3\n"
        )
        loc = _phys(_result(build_sarif([_sidecar(stderr)], SRC))["locations"][0])
        assert "compiler-rt" not in loc["artifactLocation"]["uri"]


class TestAdversarial:
    def test_path_outside_root_is_absolute(self):
        stderr = _one_frame("/usr/include/x.h")
        loc = _phys(_result(build_sarif([_sidecar(stderr)], SRC))["locations"][0])
        assert loc["artifactLocation"] == {"uri": "file:///usr/include/x.h"}

    def test_dotdot_cannot_escape_root(self):
        stderr = _one_frame("/work/proj/../etc/passwd")
        loc = _phys(_result(build_sarif([_sidecar(stderr)], SRC))["locations"][0])
        assert loc["artifactLocation"] == {"uri": "file:///work/etc/passwd"}

    def test_sibling_prefix_is_not_under_root(self):
        stderr = _one_frame("/work/project2/a.c")
        loc = _phys(_result(build_sarif([_sidecar(stderr)], SRC))["locations"][0])
        assert "uriBaseId" not in loc["artifactLocation"]

    def test_results_capped(self):
        many = [_sidecar(UAF, cluster=f"{i:08x}") for i in range(MAX_RESULTS + 3)]
        assert len(build_sarif(many, SRC)["runs"][0]["results"]) == MAX_RESULTS

    def test_garbage_fields_do_not_raise(self):
        bad = {"raw_stderr": 7, "frames": "x", "exploitability": None, "cluster_id": None}
        s = build_sarif([bad], SRC)
        assert len(s["runs"][0]["results"]) == 1
        assert s["runs"][0]["results"][0]["ruleId"] == "crash/unknown"

    def test_line_zero_has_no_region(self):
        stderr = UAF.replace("png.c:120:9", "png.c:0")
        loc = _phys(_result(build_sarif([_sidecar(stderr)], SRC))["locations"][0])
        assert "region" not in loc


def _write_crash(d: Path, side: dict, base: str) -> None:
    (d / f"{base}.json").write_text(json.dumps(side))
    (d / f"{base}.bin").write_bytes(b"x")


class TestLoader:
    def test_skips_non_crash_and_corrupt_json(self, tmp_path):
        _write_crash(tmp_path, _sidecar(UAF), "crash_1_deadbeef_asan_uaf")
        (tmp_path / "crash_2_00000002_asan_x.json").write_text("{not json")
        (tmp_path / "crash_3_00000003_asan_x.json").write_text("[1, 2]")
        (tmp_path / "deadbeefdeadbeef_sanitizer_report.json").write_text("{}")
        side = load_crash_sidecars(tmp_path)
        assert [s["crash_input"] for s in side] == ["crash_1_deadbeef_asan_uaf.bin"]

    def test_missing_dir_is_empty(self, tmp_path):
        assert load_crash_sidecars(tmp_path / "nope") == []


class TestExport:
    def test_writes_valid_sarif(self, tmp_path):
        crashes = tmp_path / "crashes"
        crashes.mkdir()
        _write_crash(crashes, _sidecar(UAF), "crash_1_deadbeef_asan_uaf")
        out = tmp_path / "out" / "fuzz.sarif"
        assert export_sarif(crashes, out, SRC) == 1
        doc = json.loads(out.read_text())
        assert doc["runs"][0]["results"][0]["ruleId"] == "AddressSanitizer/heap-use-after-free"
        assert not list(out.parent.glob(".*tmp"))

    def test_empty_dir_writes_empty_run(self, tmp_path):
        out = tmp_path / "e.sarif"
        assert export_sarif(tmp_path, out, "") == 0
        assert json.loads(out.read_text())["runs"][0]["results"] == []

    def test_cli(self, tmp_path, capsys):
        _write_crash(tmp_path, _sidecar(UAF), "crash_1_deadbeef_asan_uaf")
        out = tmp_path / "c.sarif"
        args = Namespace(crashes_dir=str(tmp_path), output=str(out), src_root=SRC)
        assert cmd_sarif(args) == 0
        assert out.exists()
        assert "1 result" in capsys.readouterr().out

    def test_cli_missing_dir(self, tmp_path):
        args = Namespace(crashes_dir=str(tmp_path / "x"), output=str(tmp_path / "o"), src_root="")
        assert cmd_sarif(args) == 1


class TestEndToEnd:
    def test_save_crash_then_export(self, tmp_path):
        from fuzzer_tool.adapters.filesystem import save_crash

        crashes = tmp_path / "crashes"
        crashes.mkdir()
        assert save_crash(b"\x89PNG", -6, UAF, crashes, set(), {})

        out = tmp_path / "e2e.sarif"
        assert export_sarif(crashes, out, SRC) == 1
        r = json.loads(out.read_text())["runs"][0]["results"][0]
        assert r["ruleId"] == "AddressSanitizer/heap-use-after-free"
        assert (crashes / r["properties"]["crash_input"]).is_file()
        assert _phys(r["locations"][0])["region"]["startLine"] == 120


class TestParserWiring:
    def test_sarif_subcommand(self, tmp_path, monkeypatch):
        from fuzzer_tool.cli.commands import main

        _write_crash(tmp_path, _sidecar(UAF), "crash_1_deadbeef_asan_uaf")
        out = tmp_path / "w.sarif"
        argv = ["fuzzer-tool", "sarif", str(tmp_path), "-o", str(out), "--src-root", SRC]
        monkeypatch.setattr("sys.argv", argv)
        assert main() == 0
        assert json.loads(out.read_text())["runs"][0]["results"]

    def test_fuzz_flags_registered(self, monkeypatch, capsys):
        import pytest

        from fuzzer_tool.cli.commands import main

        monkeypatch.setattr("sys.argv", ["fuzzer-tool", "fuzz", "--help"])
        with pytest.raises(SystemExit):
            main()
        text = capsys.readouterr().out
        assert "--sarif " in text
        assert "--sarif-src-root" in text
