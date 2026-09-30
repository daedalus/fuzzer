"""Failure-inducing combination isolation against a real crashing target.

Ground truth comes from reading ``targets/proto_target.c``: each crash needs
an exact set of header bytes and nothing else. Every case below builds the
target with ASAN, replays a crashing input through the real binary, and
checks that ``isolate_fields_failure`` returns exactly that set, no more
(irrelevant fields excluded) and no less.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from fuzzer_tool.core import field_spec
from fuzzer_tool.core.failure_inducing import ISOLATED
from fuzzer_tool.services.root_cause import root_cause

TARGET_SRC = Path(__file__).resolve().parent.parent / "targets" / "proto_target.c"

# One layout per crash family (fields may not overlap): a u16 at 8 for the
# READ crash, a u32 at 8 for the STATUS crash. ``b`` and the tail are never
# needed by the WRITE / CLOSE crashes, so they must drop out of those results.
SPEC_WORD = "magic@0:4,ver@4:1,cmd@5:1,a@6:1,b@7:1,word@8:2le,tail@10:4"
SPEC_CS = "magic@0:4,ver@4:1,cmd@5:1,a@6:1,b@7:1,cs@8:4le,tail@12:2"
OPEN, CLOS = 0x4F50454E, 0x434C4F53  # "OPEN", "CLOS" (big-endian u32)

CASES = {
    # name: (crash input, spec, expected {field: value})
    "null_deref": (
        b"OPENVRLE" + bytes([0xAD, 0xDE]) + b"pad!",
        SPEC_WORD,
        {
            "magic": OPEN,
            "ver": ord("V"),
            "cmd": ord("R"),
            "a": ord("L"),
            "b": ord("E"),
            "word": 0xDEAD,
        },
    ),
    "heap_overflow": (
        b"OPENVWXqzzzzzzzzzz",
        SPEC_WORD,
        {"magic": OPEN, "ver": ord("V"), "cmd": ord("W"), "a": ord("X")},
    ),
    "stack_overflow": (
        b"OPENVSUM" + bytes([0xBE, 0xBA, 0xFE, 0xCA]) + b"tail",
        SPEC_CS,
        {
            "magic": OPEN,
            "ver": ord("V"),
            "cmd": ord("S"),
            "a": ord("U"),
            "b": ord("M"),
            "cs": 0xCAFEBABE,
        },
    ),
    "abort": (
        b"CLOSEDxxxxxxxxxxxx",
        SPEC_WORD,
        {"magic": CLOS, "ver": ord("E"), "cmd": ord("D")},
    ),
}


@pytest.fixture(scope="module")
def proto_asan(tmp_path_factory) -> str:
    gcc = shutil.which("gcc")
    if gcc is None or not TARGET_SRC.is_file():
        pytest.skip("gcc or targets/proto_target.c not available")
    out = tmp_path_factory.mktemp("proto") / "proto_asan"
    r = subprocess.run(
        [gcc, "-O0", "-g", "-fsanitize=address", "-o", str(out), str(TARGET_SRC)],
        capture_output=True,
    )
    if r.returncode != 0:
        pytest.skip("cannot build proto_target with ASAN here")
    return str(out)


def _crashes(binary: str, data: bytes) -> bool:
    r = subprocess.run([binary], input=data, capture_output=True, timeout=10)
    return r.returncode != 0


@pytest.mark.parametrize("name", sorted(CASES))
def test_isolates_exact_crash_fields(proto_asan, name):
    data, spec, expected = CASES[name]
    assert _crashes(proto_asan, data), "sanity: the input must crash the real target"
    fields = field_spec.parse_spec(spec)

    schema = field_spec.isolate_fields_failure(
        data, lambda b: _crashes(proto_asan, b), fields, verify_samples=8
    )

    assert schema is not None and schema.status == ISOLATED
    assert {fields[i].name: v for i, v in schema.params.items()} == expected
    assert schema.verified is True
    assert schema.probes < 60  # ~k probes, nowhere near exhaustive domain product


def test_root_cause_service_reports_fields(proto_asan, tmp_path):
    """End to end through ``root_cause(isolate_fields=...)`` with the ASAN signature oracle."""
    data, spec, expected = CASES["heap_overflow"]
    crash = tmp_path / "crash.bin"
    crash.write_bytes(data)
    base = tmp_path / "base.bin"
    base.write_bytes(b"OPENVRLE" + bytes([0, 0]) + b"pad!")

    result = root_cause(proto_asan, str(crash), baseline_file=str(base), isolate_fields=spec)

    assert result is not None
    assert result["custom_field_schema"] == expected
    assert "Fields responsible" in result["report"]


def test_non_crashing_input_is_not_isolated(proto_asan):
    fields = field_spec.parse_spec(SPEC_WORD)
    ok = b"OPENVRLE" + bytes([0, 0]) + b"pad!"
    schema = field_spec.isolate_fields_failure(ok, lambda b: _crashes(proto_asan, b), fields)
    assert schema is not None and schema.status == "not_failing" and schema.params == {}
