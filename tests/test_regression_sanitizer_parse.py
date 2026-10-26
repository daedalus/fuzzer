"""Regression: sanitizer report parsing (header anchor, crash stack, stack hash, UBSAN)."""

from fuzzer_tool.adapters.filesystem import classify_crash
from fuzzer_tool.core.sanitizer import SanitizerReport

# Real ASAN layout: crash stack, then the allocation stack after a blank line.
_HBO = (
    "=================================================================\n"
    "==42==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x602000000014"
    " at pc 0x55d0004011d6 bp 0x7ffc sp 0x7ffc\n"
    "READ of size 4 at 0x602000000014 thread T0\n"
    "    #0 0x55d0004011d6 in parse_chunk /src/a.c:10:5\n"
    "    #1 0x55d000401250 in main /src/a.c:20:3\n"
    "\n"
    "0x602000000014 is located 0 bytes after 4-byte region\n"
    "allocated by thread T0 here:\n"
    "    #0 0x55d000499aaa in malloc\n"
    "    #1 0x55d000401240 in main /src/a.c:18:9\n"
    "\n"
    "SUMMARY: AddressSanitizer: heap-buffer-overflow /src/a.c:10:5 in parse_chunk\n"
)


def _stack(pcs: list[int], names: list[str]) -> str:
    """ASAN report whose crash stack is *pcs* (one frame per entry)."""
    lines = ["==1==ERROR: AddressSanitizer: heap-use-after-free on address 0x1\n"]
    for i, (pc, name) in enumerate(zip(pcs, names, strict=True)):
        lines.append(f"    #{i} {pc:#x} in {name}\n")
    return "".join(lines)


def test_regression_echoed_sanitizer_text_not_report() -> None:
    # Adversarial: the target echoes the input into an error message.
    assert SanitizerReport.parse("bad token 'AddressSanitizer: heap-buffer-overflow'\n") is None
    assert SanitizerReport.parse("log: x UndefinedBehaviorSanitizer: undefined\n") is None


def test_real_report_still_parses() -> None:
    # Falsification: a genuine header is a report.
    report = SanitizerReport.parse(_HBO)

    assert report is not None
    assert report.sanitizer == "AddressSanitizer"
    assert report.error_type == "heap-buffer-overflow"


def test_regression_error_type_from_header_line() -> None:
    # An echoed line with another error type must not override the header.
    stderr = (
        "log: AddressSanitizer: heap-buffer-overflow\n"
        "==1==ERROR: AddressSanitizer: heap-use-after-free on address 0x1\n"
    )
    report = SanitizerReport.parse(stderr)

    assert report is not None
    assert report.error_type == "heap-use-after-free"


def test_regression_frames_from_crash_stack_only() -> None:
    report = SanitizerReport.parse(_HBO)

    assert report is not None
    assert report.frames == ["parse_chunk", "main"]
    assert report.alloc_frames == ["malloc", "main"]


def test_regression_unsymbolized_frames_use_module_offset() -> None:
    stderr = (
        "==1==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x1\n"
        "    #0 0x55d0004011d6  (/opt/t/target_asan+0x11d6)\n"
        "    #1 0x7f0000002d1c  (/lib/x86_64-linux-gnu/libc.so.6+0x2d1c)\n"
    )
    other = stderr.replace("+0x11d6", "+0x22e8").replace("0x55d0004011d6", "0x55d0004022e8")
    a, b = SanitizerReport.parse(stderr), SanitizerReport.parse(other)

    assert a is not None and b is not None
    assert a.frames == ["target_asan+0x11d6", "libc.so.6+0x2d1c"]
    assert a.signature != b.signature


def test_regression_stack_hash_is_order_sensitive() -> None:
    a = SanitizerReport.parse(_stack([0x401111, 0x402222], ["f", "g"]))
    b = SanitizerReport.parse(_stack([0x402222, 0x401111], ["g", "f"]))

    assert a is not None and b is not None
    assert a.stack_hash() != b.stack_hash()


def test_regression_stack_hash_recursion_does_not_cancel() -> None:
    # XOR folding made [A, B, B] hash like [A, C, C].
    a = SanitizerReport.parse(_stack([0x401111, 0x402222, 0x402222], ["f", "g", "g"]))
    b = SanitizerReport.parse(_stack([0x401111, 0x403333, 0x403333], ["f", "h", "h"]))

    assert a is not None and b is not None
    assert a.stack_hash() != b.stack_hash()


def test_stack_hash_stable_across_aslr() -> None:
    # Falsification: the same stack at another load base hashes the same.
    base = 0x7F0000000000
    a = SanitizerReport.parse(_stack([0x55D000401111, 0x55D000402222], ["f", "g"]))
    b = SanitizerReport.parse(_stack([base + 0x1111, base + 0x2222], ["f", "g"]))

    assert a is not None and b is not None
    assert a.stack_hash() == b.stack_hash()


def test_regression_stack_hash_ignores_alloc_stack() -> None:
    other = _HBO.replace("0x55d000499aaa in malloc", "0x55d000499bbb in calloc")
    a, b = SanitizerReport.parse(_HBO), SanitizerReport.parse(other)

    assert a is not None and b is not None
    assert a.stack_hash() == b.stack_hash()


def test_regression_asan_segv_fault_addr() -> None:
    stderr = (
        "==1==ERROR: AddressSanitizer: SEGV on unknown address 0x000000000010"
        " (pc 0x401000 bp 0x0 sp 0x0 T0)\n"
    )
    report = SanitizerReport.parse(stderr)

    assert report is not None
    assert report.error_type == "SEGV"
    assert report.fault_addr == "0x000000000010"


def test_regression_lsan_error_type() -> None:
    stderr = (
        "==1==ERROR: LeakSanitizer: detected memory leaks\n\n"
        "Direct leak of 24 byte(s) in 1 object(s) allocated from:\n"
        "    #0 0x401000 in malloc\n\n"
        "SUMMARY: AddressSanitizer: 24 byte(s) leaked in 1 allocation(s).\n"
    )
    report = SanitizerReport.parse(stderr)

    assert report is not None
    assert report.sanitizer == "LeakSanitizer"
    assert report.error_type == "detected memory leaks"


_UBSAN_OVF = (
    "/src/a.c:3:5: runtime error: signed integer overflow: 2147483647 + 1"
    " cannot be represented in type 'int'\n"
    "SUMMARY: UndefinedBehaviorSanitizer: undefined-behavior /src/a.c:3:5\n"
)


def test_regression_ubsan_runtime_error_type() -> None:
    report = SanitizerReport.parse(_UBSAN_OVF)

    assert report is not None
    assert report.sanitizer == "UndefinedBehaviorSanitizer"
    assert report.error_type == "signed-integer-overflow"
    assert report.exploitability == "MEDIUM"


def test_regression_ubsan_location_splits_signature() -> None:
    # No stack (UBSAN default): two sites must not share one bucket.
    other = _UBSAN_OVF.replace("a.c:3:5", "b.c:9:1")
    a, b = SanitizerReport.parse(_UBSAN_OVF), SanitizerReport.parse(other)

    assert a is not None and b is not None
    assert a.signature != b.signature


def test_ubsan_runtime_error_without_summary() -> None:
    report = SanitizerReport.parse("/src/a.c:7:2: runtime error: division by zero\n")

    assert report is not None
    assert report.error_type == "integer-divide-by-zero"


def test_ubsan_runtime_error_echo_not_report() -> None:
    # Adversarial: a mid-line "runtime error:" is not a UBSAN header.
    assert SanitizerReport.parse("got: x.c:1:1: runtime error: division by zero\n") is None


# ── Dedup: fuzzy signature matching ─────────────────────────────────────


def _known(stderr: str) -> dict[str, int]:
    report = SanitizerReport.parse(stderr)
    assert report is not None
    return {report.signature: 1}


def test_regression_fuzzy_match_keeps_error_type() -> None:
    # A UAF in the same frames as a known overflow is another bug.
    frames = ["parse_chunk", "read_block", "decode", "main"]
    hbo = _stack([0x401000 + i for i in range(4)], frames).replace(
        "heap-use-after-free", "heap-buffer-overflow"
    )
    uaf = _stack([0x401000 + i for i in range(4)], frames)

    v = classify_crash(b"u", 1, uaf, set(), _known(hbo))

    assert v.novel


def test_regression_fuzzy_match_keeps_unsymbolized_offsets() -> None:
    # Offsets are the only identity of an unsymbolized frame.
    def unsym(off: int) -> str:
        return (
            "==1==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x1\n"
            f"    #0 0x55d000400000  (/opt/t/target+{off:#x})\n"
        )

    v = classify_crash(b"b", 1, unsym(0x22E8), set(), _known(unsym(0x11D6)))

    assert v.novel


def test_fuzzy_match_folds_inlined_frame() -> None:
    # Falsification: same bug, one extra inlined frame, still folds.
    base = ["parse_chunk", "read_block", "decode", "main"]
    inlined = ["parse_chunk", "parse_chunk_inl", "read_block", "decode", "main"]
    known = _stack([0x401000 + i for i in range(4)], base)
    new = _stack([0x401000 + i for i in range(5)], inlined)

    v = classify_crash(b"i", 1, new, set(), _known(known))

    assert not v.novel
    assert v.matched_signature == next(iter(_known(known)))
