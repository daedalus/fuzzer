"""Regression: ``_read_v5_table`` trusted the binary's entry count.

A crafted .debug_line with zero-width entries (no formats, or only
DW_FORM_flag_present / DW_FORM_implicit_const) and a huge ULEB count
appended ``{}`` forever: hang + unbounded memory in the fuzzer itself.

Hang-prone calls run in a subprocess with a wall-clock timeout and an
address-space cap so a regression fails instead of wedging the suite.
"""

import resource
import subprocess
import sys
from pathlib import Path

import pytest

import fuzzer_tool
from fuzzer_tool.core.dwarf import _read_v5_table

_SRC_ROOT = str(Path(fuzzer_tool.__file__).resolve().parents[1])
_CHILD_TIMEOUT_S = 10
_CHILD_MEM_BYTES = 1 << 30
_HUGE_COUNT = 1 << 40

# DWARF 5 constants, restated from the spec (independent of dwarf.py).
_LNCT_PATH = 1
_LNCT_DIR_INDEX = 2
_FORM_STRING = 0x08
_FORM_UDATA = 0x0F
_FORM_FLAG_PRESENT = 0x19
_FORM_DATA16 = 0x1E
_FORM_IMPLICIT_CONST = 0x21


def _uleb(value: int) -> bytes:
    """Encode *value* as ULEB128."""
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
            continue
        out.append(byte)
        return bytes(out)


def _table(formats: list[tuple[int, int]], count: int, body: bytes) -> bytes:
    """[format_count][(ct, form)…][count][body]."""
    head = _uleb(len(formats))
    for ct, form in formats:
        head += _uleb(ct) + _uleb(form)
    return head + _uleb(count) + body


def _cap_memory() -> None:
    resource.setrlimit(resource.RLIMIT_AS, (_CHILD_MEM_BYTES, _CHILD_MEM_BYTES))


def _run_guarded(data: bytes) -> tuple[int, int]:
    """Parse *data* in a capped child → (entry count, end offset)."""
    code = (
        "import sys\n"
        "from fuzzer_tool.core.dwarf import _read_v5_table\n"
        f"entries, off = _read_v5_table({data!r}, 0, {{}})\n"
        "print(len(entries), off)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=_CHILD_TIMEOUT_S,
        preexec_fn=_cap_memory,
        env={"PYTHONPATH": _SRC_ROOT},
    )
    assert proc.returncode == 0, proc.stderr[-500:]
    n, off = proc.stdout.split()
    return int(n), int(off)


@pytest.mark.timeout(30)
def test_regression_dwarf_v5_table_unbounded() -> None:
    """format_count=0 + huge count must terminate with no entries."""
    data = _table([], _HUGE_COUNT, b"")

    n, off = _run_guarded(data)

    assert n == 0
    assert off == len(data)


@pytest.mark.timeout(30)
@pytest.mark.parametrize("form", [_FORM_FLAG_PRESENT, _FORM_IMPLICIT_CONST])
def test_zero_width_forms_bounded(form: int) -> None:
    """Adversarial: forms that consume no bytes cannot drive the loop."""
    data = _table([(_LNCT_PATH, form)], _HUGE_COUNT, b"")

    n, off = _run_guarded(data)

    assert n == 0
    assert off == len(data)


@pytest.mark.timeout(30)
def test_overrunning_form_bounded() -> None:
    """Adversarial: data16 slices past the end silently; must not loop."""
    data = _table([(_LNCT_PATH, _FORM_DATA16)], _HUGE_COUNT, b"\x01\x02\x03")

    n, _off = _run_guarded(data)

    assert n == 0


def test_count_exceeds_remaining_bytes() -> None:
    """Adversarial: count > remaining bytes yields only the real entries."""
    body = bytes([0, 1, 2])
    data = _table([(_LNCT_DIR_INDEX, _FORM_UDATA)], 1000, body)

    entries, off = _read_v5_table(data, 0, {})

    assert entries == [{_LNCT_DIR_INDEX: 0}, {_LNCT_DIR_INDEX: 1}, {_LNCT_DIR_INDEX: 2}]
    assert off == len(data)


def test_explicit_end_stops_table() -> None:
    """Entries past the caller's *end* (header end) are not read."""
    body = bytes([7, 8, 9])
    data = _table([(_LNCT_DIR_INDEX, _FORM_UDATA)], len(body), body)
    end = len(data) - 1

    entries, off = _read_v5_table(data, 0, {}, end)

    assert entries == [{_LNCT_DIR_INDEX: 7}, {_LNCT_DIR_INDEX: 8}]
    assert off == end


def test_truncated_format_list_raises() -> None:
    """Adversarial: a format list cut mid-pair is reported, not looped."""
    data = _uleb(3) + _uleb(_LNCT_PATH)

    with pytest.raises(ValueError):
        _read_v5_table(data, 0, {})


def test_well_formed_table_parses() -> None:
    """Falsification: a valid table still yields the exact entries."""
    formats = [(_LNCT_PATH, _FORM_STRING), (_LNCT_DIR_INDEX, _FORM_UDATA)]
    body = b"a.c\x00" + _uleb(0) + b"b.c\x00" + _uleb(1)
    trailer = b"\xff\xff"
    data = _table(formats, 2, body) + trailer

    entries, off = _read_v5_table(data, 0, {})

    assert entries == [
        {_LNCT_PATH: "a.c", _LNCT_DIR_INDEX: 0},
        {_LNCT_PATH: "b.c", _LNCT_DIR_INDEX: 1},
    ]
    assert off == len(data) - len(trailer)
