"""glibc-version-aware crash severity multiplier (Phrack #73 "Poor Little Chunks").

Covers the two halves of the feature:

* ``core.elf.glibc_build_version`` -- reading the target's build-time glibc from
  its ``.gnu.version_r`` records. Exercised against a hand-built synthetic ELF
  (hermetic) and, when available, cross-checked against ``readelf`` on a real
  binary (an oracle independent of the code under test, per Hard Rule 46/§Testing).
* ``core.sanitizer.glibc_severity`` -- the tier-upgrade logic.
"""

import re
import shutil
import struct
import subprocess

import pytest

from fuzzer_tool.core.elf import (
    _glibc_from_verneed,
    _parse_glibc_token,
    _read_cstr,
    glibc_build_version,
)
from fuzzer_tool.core.sanitizer import (
    GLIBC_HEAP_REGRESSION,
    glibc_regressed,
    glibc_severity,
    heap_metadata_corruption,
    parse_glibc_version,
)

_SHT_GNU_VERNEED = 0x6FFFFFFE


def _build_elf_with_verneed(versions: list[str]) -> bytes:
    """A minimal ELF64-LE exposing *versions* as .gnu.version_r tokens.

    Three section headers: null, a string table holding the version strings,
    and a SHT_GNU_verneed record (one Verneed, one Vernaux per version) linked
    to that string table. This is the structure ``_glibc_from_verneed`` walks.
    """
    strtab = bytearray(b"\x00")
    name_offsets = []
    for v in versions:
        name_offsets.append(len(strtab))
        strtab += v.encode() + b"\x00"

    verneed = bytearray()
    # Elf64_Verneed: version, cnt, file, aux(=16), next(=0 last).
    verneed += struct.pack("<HHIII", 1, len(versions), 0, 16, 0)
    for i, noff in enumerate(name_offsets):
        last = i == len(name_offsets) - 1
        # Elf64_Vernaux: hash, flags, other, name, next.
        verneed += struct.pack("<IHHII", 0, 0, 0, noff, 0 if last else 16)

    ehdr_size, shdr_size, shnum = 64, 64, 3
    shoff = ehdr_size
    strtab_off = shoff + shnum * shdr_size
    verneed_off = strtab_off + len(strtab)
    verneed_off += (-verneed_off) % 4  # 4-align
    total = verneed_off + len(verneed)

    buf = bytearray(total)
    struct.pack_into(
        "<16sHHIQQQIHHHHHH", buf, 0,
        b"\x7fELF\x02\x01\x01" + b"\x00" * 9,  # e_ident (class 64, LE)
        2, 0x3E, 1, 0, 0, shoff, 0, 64, 56, 0, 64, shnum, 0,
    )  # fmt: skip
    # shdr fields: name(I) type(I) flags(Q) addr(Q) offset(Q) size(Q) link(I)
    #              info(I) addralign(Q) entsize(Q).
    # section 1: the string table (.dynstr-style)
    struct.pack_into(
        "<IIQQQQIIQQ", buf, shoff + shdr_size,
        0, 3, 0, 0, strtab_off, len(strtab), 0, 0, 0, 0,
    )  # fmt: skip
    # section 2: verneed, linked (sh_link=1) to the string table above
    struct.pack_into(
        "<IIQQQQIIQQ", buf, shoff + 2 * shdr_size,
        0, _SHT_GNU_VERNEED, 0, 0, verneed_off, len(verneed), 1, len(versions), 0, 0,
    )  # fmt: skip
    buf[strtab_off : strtab_off + len(strtab)] = strtab
    buf[verneed_off : verneed_off + len(verneed)] = verneed
    return bytes(buf)


# ── token / string parsing ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "token,expected",
    [
        (b"GLIBC_2.41", (2, 41)),
        (b"GLIBC_2.43", (2, 43)),
        (b"GLIBC_2.2.5", (2, 2)),  # only major.minor taken
        (b"GLIBC_PRIVATE", None),
        (b"GLIBC_2", None),  # no minor
        (b"FOO_1.0", None),
        (b"", None),
    ],
)
def test_parse_glibc_token(token, expected):
    assert _parse_glibc_token(token) == expected


def test_read_cstr_is_bounded():
    buf = b"\x00abc\x00def"
    assert _read_cstr(buf, 1, len(buf)) == b"abc"
    assert _read_cstr(buf, 5, len(buf)) == b"def"
    # start at/after end clamps to empty rather than reading past.
    assert _read_cstr(buf, len(buf), len(buf)) == b""
    assert _read_cstr(buf, 100, len(buf)) == b""


# ── verneed walk over a synthetic ELF ─────────────────────────────────────


def test_glibc_from_verneed_takes_the_max():
    elf = _build_elf_with_verneed(["GLIBC_2.30", "GLIBC_2.41", "GLIBC_2.17"])
    # Max of {2.30, 2.41, 2.17} = 2.41, derived by hand, not from the reader.
    assert _glibc_from_verneed(elf) == (2, 41)


def test_glibc_build_version_from_file(tmp_path):
    elf = _build_elf_with_verneed(["GLIBC_2.43", "GLIBC_2.4"])
    p = tmp_path / "fake.elf"  # tmp_path is auto-cleaned (Hard Rule 21)
    p.write_bytes(elf)
    assert glibc_build_version(str(p)) == "2.43"


def test_glibc_from_verneed_ignores_private_and_nonglibc():
    elf = _build_elf_with_verneed(["GLIBC_PRIVATE", "GCC_3.0", "GLIBC_2.34"])
    assert _glibc_from_verneed(elf) == (2, 34)


def test_no_glibc_tokens_yields_none():
    elf = _build_elf_with_verneed(["GCC_3.0", "GLIBC_PRIVATE"])
    assert _glibc_from_verneed(elf) is None


# ── adversarial: hostile / degenerate binaries never raise ────────────────


@pytest.mark.parametrize(
    "blob",
    [b"", b"\x7fELF", b"not an elf at all", b"\x7fELF\x02\x01\x01" + b"\x00" * 200],
)
def test_malformed_binary_returns_empty(blob):
    assert _glibc_from_verneed(blob) is None


def test_nonexistent_target_returns_empty(tmp_path):
    assert glibc_build_version(str(tmp_path / "does-not-exist")) == ""


def test_statically_linked_has_no_verneed(tmp_path):
    # No verneed section at all (a static binary) -> empty, multiplier no-ops.
    p = tmp_path / "static.elf"
    p.write_bytes(_build_elf_with_verneed([])[:64])  # header only, truncated
    assert glibc_build_version(str(p)) == ""


# ── cross-check against readelf on a real binary (independent oracle) ──────


@pytest.mark.skipif(not shutil.which("readelf"), reason="readelf not installed")
@pytest.mark.skipif(not shutil.which("ls"), reason="/bin/ls not found")
def test_matches_readelf_on_real_binary():
    target = shutil.which("ls")
    out = subprocess.run(
        ["readelf", "-V", target], capture_output=True, text=True, check=True
    ).stdout
    tokens = {tuple(map(int, m.split(".")[:2])) for m in re.findall(r"GLIBC_(\d+\.\d+)", out)}
    if not tokens:
        pytest.skip("target has no GLIBC version-needed records")
    expected = max(tokens)
    assert glibc_build_version(target) == f"{expected[0]}.{expected[1]}"


# ── version comparison ────────────────────────────────────────────────────


def test_regression_threshold_is_2_41():
    # Pins the constant to the Phrack #73 versions, not whatever the code says.
    assert GLIBC_HEAP_REGRESSION == (2, 41)


@pytest.mark.parametrize(
    "version,regressed",
    [
        ("2.41", True),
        ("2.43", True),
        ("3.0", True),
        ("2.40", False),
        ("2.38", False),
        ("2.31", False),
        ("", False),
        ("garbage", False),
        ("2", False),
    ],
)
def test_glibc_regressed(version, regressed):
    assert glibc_regressed(version) is regressed


@pytest.mark.parametrize(
    "version,expected",
    [("2.41", (2, 41)), ("2.2.5", (2, 2)), ("", None), ("x.y", None), ("2", None)],
)
def test_parse_glibc_version(version, expected):
    assert parse_glibc_version(version) == expected


# ── heap-metadata-corruption classification ───────────────────────────────


@pytest.mark.parametrize(
    "san,err,stderr,expected",
    [
        ("AddressSanitizer", "heap-buffer-overflow", "", True),
        ("AddressSanitizer", "heap-use-after-free", "", True),
        ("AddressSanitizer", "double-free", "", True),
        ("AddressSanitizer", "stack-buffer-overflow", "", False),  # stack, not heap
        ("AddressSanitizer", "global-buffer-overflow", "", False),
        ("", "SEGV", "free(): invalid pointer\n", True),  # glibc allocator abort
        ("", "SEGV", "corrupted size vs. prev_size\n", True),
        ("", "SEGV", "Segmentation fault\n", False),  # plain SEGV, not allocator
    ],
)
def test_heap_metadata_corruption(san, err, stderr, expected):
    assert heap_metadata_corruption(san, err, stderr) is expected


# ── the multiplier itself ─────────────────────────────────────────────────


def test_bumps_heap_crash_on_regressed_glibc():
    # HIGH heap-UAF on glibc 2.41 -> one tier up to CRITICAL.
    assert (
        glibc_severity("HIGH", "AddressSanitizer", "heap-use-after-free", "", "2.41") == "CRITICAL"
    )


def test_bumps_medium_to_high():
    assert (
        glibc_severity("MEDIUM", "AddressSanitizer", "heap-use-after-scope", "", "2.43") == "HIGH"
    )


def test_critical_is_the_ceiling():
    assert glibc_severity("CRITICAL", "AddressSanitizer", "double-free", "", "2.41") == "CRITICAL"


def test_signal_heap_abort_bumped():
    stderr = "free(): invalid pointer\n"
    assert glibc_severity("HIGH", "", "SEGV", stderr, "2.41") == "CRITICAL"


# ── falsification: the multiplier must NOT fire outside its contract ───────


def test_pre_regression_glibc_never_changes_tier():
    # Same heap crash, glibc 2.38 -> untouched (this host's glibc; inert case).
    assert glibc_severity("HIGH", "AddressSanitizer", "heap-use-after-free", "", "2.38") == "HIGH"


def test_unknown_glibc_never_changes_tier():
    assert glibc_severity("MEDIUM", "AddressSanitizer", "double-free", "", "") == "MEDIUM"


def test_non_heap_crash_never_bumped_even_on_regressed_glibc():
    # A stack overflow is unaffected by the heap-hardening regression.
    assert glibc_severity("HIGH", "AddressSanitizer", "stack-buffer-overflow", "", "2.43") == "HIGH"


def test_unknown_tier_stays_unknown():
    # A crash with no tier must not be invented into one by the multiplier.
    assert (
        glibc_severity("UNKNOWN", "AddressSanitizer", "heap-buffer-overflow", "", "2.41")
        == "UNKNOWN"
    )


# ── integration: the multiplier is applied in the crash-triage write path ──

_UAF_STDERR = (
    "==123==ERROR: AddressSanitizer: heap-use-after-free on address 0x602000000010\n"
    "READ of size 4 at 0x602000000010 thread T0\n"
    "    #0 0x4011d6 in use_it /src/a.c:5\n"
)


def _uaf_meta(glibc: str):
    from fuzzer_tool.adapters.filesystem import _fill_report
    from fuzzer_tool.core.crash_metadata import CrashMetadata
    from fuzzer_tool.core.sanitizer import SanitizerReport

    report = SanitizerReport.parse(_UAF_STDERR)
    meta = CrashMetadata()
    meta.raw_stderr = _UAF_STDERR
    meta.glibc_version = glibc
    _fill_report(meta, report, 0)
    return report, meta


def test_regression_fill_report_bumps_on_regressed_glibc():
    # A READ heap-UAF parses as HIGH; on glibc 2.41 _fill_report must raise it.
    report, meta = _uaf_meta("2.41")
    assert report.exploitability == "HIGH"  # base tier, pre-multiplier
    assert meta.exploitability == "CRITICAL"


def test_regression_fill_report_control_pre_regression_glibc():
    # Control (Hard Rule 46): identical crash, glibc 2.38 -> tier unchanged.
    report, meta = _uaf_meta("2.38")
    assert report.exploitability == "HIGH"
    assert meta.exploitability == "HIGH"


def test_regression_fill_report_no_glibc_is_noop():
    # Default (feature off): glibc_version empty -> base tier preserved.
    _report, meta = _uaf_meta("")
    assert meta.exploitability == "HIGH"
