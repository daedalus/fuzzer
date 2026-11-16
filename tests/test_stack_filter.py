"""--stack-ignore regexes and the rebuild-stable function hash (OneFuzz port P2).

User regexes extend runtime-frame stripping so project wrappers
(``av_free``) stop being the signature. ``func_hash`` hashes function names
only, so a ``--crash-blocklist`` survives a rebuild that moves every PC.
"""

from __future__ import annotations

import argparse
import json

import pytest

from fuzzer_tool.adapters.filesystem import classify_crash
from fuzzer_tool.cli.commands import _load_hash_list, _stack_ignore_file
from fuzzer_tool.core.sanitizer import (
    FUNC_HASH_TAG,
    SanitizerReport,
    configure_stack_ignore,
)
from fuzzer_tool.services.corpus_manager import CorpusManager
from tests.test_regression_crash_triage_cost import _Fuzzer

_UAF = "==1==ERROR: AddressSanitizer: heap-use-after-free on address 0x602000000010\n"
_FRAMES = ["free", "av_free", "av_freep", "decode_frame", "avcodec_send_packet", "fuzz_shm_run"]


@pytest.fixture(autouse=True)
def _builtin_filter():
    """Every test starts and ends on the built-in predicate (Hard Rule 21)."""
    configure_stack_ignore([])
    yield
    configure_stack_ignore([])


def _asan(frames: list[str], base: int = 0x401000) -> str:
    body = "".join(f"    #{i} 0x{base + i * 0x10:x} in {f}\n" for i, f in enumerate(frames))
    return _UAF + body


def _report(frames: list[str], base: int = 0x401000) -> SanitizerReport:
    r = SanitizerReport.parse(_asan(frames, base))
    assert r is not None
    return r


def _sig(*frames: str) -> str:
    return "AddressSanitizer:heap-use-after-free" + "".join(f"@{f}" for f in frames)


def test_no_patterns_keeps_builtin_signature() -> None:
    # Falsification: without a file, `free` is stripped and wrappers remain.
    assert _report(_FRAMES).signature == _sig("av_free", "av_freep", "decode_frame")


def test_patterns_strip_wrappers() -> None:
    configure_stack_ignore(["^av_free"])

    assert _report(_FRAMES).signature == _sig("decode_frame", "avcodec_send_packet")


def test_reset_restores_builtin() -> None:
    configure_stack_ignore(["^av_free"])
    configure_stack_ignore([])

    assert _report(_FRAMES).signature == _sig("av_free", "av_freep", "decode_frame")


def test_match_all_keeps_innermost() -> None:
    # Adversarial: ".*" must not empty the signature or the hash.
    configure_stack_ignore([".*"])
    r = _report(_FRAMES)

    assert r.signature == _sig("free")
    assert r.func_hash()


def test_func_hash_rebuild_stable() -> None:
    # A rebuild moves every PC: stack_hash changes, func_hash does not.
    a, b = _report(_FRAMES, 0x401000), _report(_FRAMES, 0x401004)

    assert a.stack_hash() != b.stack_hash()
    assert a.func_hash() == b.func_hash()


def test_func_hash_control_differs_on_function() -> None:
    # Control (Hard Rule 46): a different bug site must change the hash.
    other = ["free", "av_free", "av_freep", "decode_header", "avcodec_send_packet"]

    assert _report(_FRAMES).func_hash() != _report(other).func_hash()


def test_func_hash_name_boundary() -> None:
    # Adversarial: ["ab", "c"] and ["a", "bc"] concatenate equally.
    assert _report(["ab", "c"]).func_hash() != _report(["a", "bc"]).func_hash()


def test_func_hash_cut_at_harness() -> None:
    # Frames below the harness are the fuzzer's (ctypes, libffi): ignored.
    deeper = _FRAMES + ["ffi_call", "_ctypes_callproc"]

    assert _report(_FRAMES).func_hash() == _report(deeper).func_hash()


def test_func_hash_skips_unsymbolized() -> None:
    # "module+0x…" offsets move per build; they are not function names.
    def text(off: int) -> str:
        return _asan(["decode_frame"]) + f"    #1 0x402000  (/bin/t+0x{off:x})\n"

    a, b = SanitizerReport.parse(text(0x11D6)), SanitizerReport.parse(text(0x22E8))
    assert a is not None and b is not None

    assert a.func_hash() == b.func_hash()


def test_func_hash_empty_stack() -> None:
    # Edge: UBSAN's default has no stack.
    r = SanitizerReport.parse("x.c:3:5: runtime error: signed integer overflow\n")
    assert r is not None

    assert r.func_hash() == ""


def test_func_hash_cache_reset_on_configure() -> None:
    # Cache safety: rebinding the filter must not serve the old hash.
    before = _report(_FRAMES).func_hash()
    configure_stack_ignore(["^av_free"])

    assert _report(_FRAMES).func_hash() != before


def test_func_hash_cache_bounded(monkeypatch) -> None:
    # Hard Rule 54: distinct stacks never grow the memo past its cap.
    from fuzzer_tool.core import sanitizer

    monkeypatch.setattr(sanitizer, "_FUNC_HASH_CACHE_MAX", 4)
    for i in range(10):
        _report([f"site_{i}"]).func_hash()

    assert len(sanitizer._FUNC_HASH_CACHE) <= 4


def test_blocklist_matches_func_hash() -> None:
    want = _report(_FRAMES).func_hash()

    v = classify_crash(b"x", 1, _asan(_FRAMES, 0x501000), set(), {}, crash_blocklist={want})

    assert v.blocked and not v.novel


def test_allowlist_overrides_func_hash() -> None:
    want = _report(_FRAMES).func_hash()

    v = classify_crash(
        b"x", 1, _asan(_FRAMES), set(), {}, crash_blocklist={want}, crash_allowlist={want}
    )

    assert v.novel and not v.blocked


def test_func_tag_list_silent(tmp_path, capsys) -> None:
    p = tmp_path / "block.txt"
    p.write_text(f"{FUNC_HASH_TAG}\nabcd\n")

    assert _load_hash_list(str(p)) == {"abcd"}
    assert capsys.readouterr().out == ""


def test_ignore_file_parsed(tmp_path) -> None:
    p = tmp_path / "ignore.txt"
    p.write_text("# wrappers\n^av_free\n\n  ^xmalloc$  \n")

    assert _stack_ignore_file(str(p)) == ["^av_free", "^xmalloc$"]


def test_ignore_file_bad_regex_names_line(tmp_path) -> None:
    # Adversarial: an invalid regex fails at parse time with its line number.
    p = tmp_path / "ignore.txt"
    p.write_text("^ok\n(unclosed\n")

    with pytest.raises(argparse.ArgumentTypeError, match=":2:"):
        _stack_ignore_file(str(p))


def test_ignore_file_missing(tmp_path) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        _stack_ignore_file(str(tmp_path / "absent.txt"))


def test_sidecar_records_func_hash(tmp_path, monkeypatch) -> None:
    f = _Fuzzer(tmp_path)
    cm = CorpusManager(f)
    monkeypatch.setattr(cm, "_novel_crash_meta", lambda *a, **k: None)
    assert cm.save_crash(b"in", 1, _asan(_FRAMES))

    want = _report(_FRAMES).func_hash()
    (side,) = f.crashes_dir.glob("*.json")
    (txt,) = f.crashes_dir.glob("*.txt")
    assert want and json.loads(side.read_text())["func_hash"] == want
    assert f"func_hash:     {want}" in txt.read_text()


class _Bound(Exception):
    """Stops cmd_fuzz right after the stack filter is bound."""


def test_cli_wires_stack_ignore(tmp_path, monkeypatch) -> None:
    # Wiring: --stack-ignore FILE → parser → cmd_fuzz → configure_stack_ignore.
    import sys

    from fuzzer_tool.cli import commands
    from fuzzer_tool.core import sanitizer

    p = tmp_path / "ignore.txt"
    p.write_text("^av_free\n")
    seen: list[list[str]] = []

    def _spy(patterns: list[str]) -> None:
        seen.append(patterns)
        raise _Bound

    monkeypatch.setattr(sanitizer, "configure_stack_ignore", _spy)
    monkeypatch.setattr(sys, "argv", ["fuzzer-tool", "fuzz", "/bin/true", "--stack-ignore", str(p)])

    with pytest.raises(_Bound):
        commands.main()
    assert seen == [["^av_free"]]
