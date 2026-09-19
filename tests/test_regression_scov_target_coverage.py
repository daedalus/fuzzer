"""Every target in the simple-target passes must receive ``$extra_cflags``.

``--clang-scov`` is plumbed by passing ``-fsanitize-coverage=trace-pc-guard``
as ``build_simple_targets``' fifth argument, which the function forwards to
each ``build_target``/``build_so_target`` call as ``$extra_cflags``.  A call
that puts something else in that slot -- an include path, say -- silently
drops the instrumentation for that one target, and nothing downstream
notices: the shim is ``-include``'d either way, so ``verify_afl`` still finds
its symbols and the fuzzer still prints "AFL instrumentation: detected".

Measured when this was written: ``fuzzgoat_read`` passed ``-I$VENDOR/fuzzgoat``
in that slot in both passes, so it was built without compiler-inserted
coverage in *every* mode.  ``readelf -S`` found no ``__sancov_guards``
section, and a 250-input run recorded 3 edges per execution -- the harness's
own ``__afl_map_edge`` calls -- against 73 once the flags reached it.
``verify_sancov`` covers the .so path only, and by section rather than by
call site, so it never reported the executable.
"""

import os
import re
from pathlib import Path

import pytest

SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tools",
    "build_targets.sh",
)

# build_target <src> <out> <libs> <flags> <cc> <extra_cflags>
_CALL = re.compile(r'\b(build_target|build_so_target)\s+("[^"]*"\s*){5}"([^"]*)"')


def _function_body(text, name):
    start = text.index(f"\n{name}() {{")
    end = text.index("\n}\n", start)
    return text[start:end]


@pytest.mark.parametrize("func", ["build_simple_targets", "build_simple_so_targets"])
def test_every_call_forwards_extra_cflags(func):
    body = _function_body(Path(SCRIPT).read_text(), func)
    offenders = [
        f"{m.group(1)} -> {m.group(3)!r}"
        for m in _CALL.finditer(body)
        if "$extra_cflags" not in m.group(3)
    ]
    assert not offenders, (
        f"{func} drops $extra_cflags for: {offenders}. Under --clang-scov that is "
        "-fsanitize-coverage=trace-pc-guard, so those targets build with no "
        "compiler-inserted edge coverage and record only manual __afl_map_edge calls."
    )


def test_fuzzgoat_object_is_compiled_with_the_same_flags():
    """The library object needs the flags too, or the parser itself is dark."""
    body = _function_body(Path(SCRIPT).read_text(), "build_simple_so_targets")
    calls = re.findall(r"compile_fuzzgoat_object\s+(.*)", body)
    assert calls, "compile_fuzzgoat_object call not found"
    for args in calls:
        assert "$extra_cflags" in args, (
            f"compile_fuzzgoat_object called with {args!r}: fuzzgoat.c compiles without "
            "instrumentation, so no edge in the JSON parser is ever recorded"
        )
