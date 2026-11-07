"""Target names in tools/build_targets.sh say what they carry.

The ASAN pass wrote ``<t>_noasan`` and the no-sanitizer pass ``<t>_nosan``,
while the .so builders already wrote ``_asan.so`` / ``_noasan.so``: an
ASAN binary named "noasan". The vendored .so targets (secp256k1, brainflayer,
sqlite) still wrote ``_nosan.so``. The builders are extracted and sourced with
``build_target`` stubbed to print the output path.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

BUILD_SCRIPT = Path(__file__).resolve().parent.parent / "tools" / "build_targets.sh"
BUILDERS = ("build_fgrep_targets", "build_simple_targets")


def _extract(name: str) -> str:
    text = BUILD_SCRIPT.read_text()
    start = text.index(f"{name}() {{")
    return text[start : text.index("\n}\n", start) + 3]


def _outputs(builder: str, suffix: str) -> list[str]:
    stubs = (
        'build_target() { echo "OUT $2"; }\nwarn() { :; }\ncompile_grep_objects() { return 1; }\n'
        "TARGETS=/b\nHAS_FUZZGOAT=0\n"
    )
    script = stubs + "".join(_extract(b) for b in BUILDERS) + f'\n{builder} "{suffix}" "" "L"\n'
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    return [ln.split(" ", 1)[1] for ln in r.stdout.splitlines() if ln.startswith("OUT ")]


@pytest.mark.parametrize("builder", BUILDERS)
def test_regression_asan_pass_names_asan(builder):
    outs = _outputs(builder, "_asan")
    assert outs and all(o.endswith("_asan") for o in outs), outs


@pytest.mark.parametrize("builder", BUILDERS)
def test_no_sanitizer_pass_names_noasan(builder):
    outs = _outputs(builder, "_nosan")
    assert outs and all(o.endswith("_noasan") for o in outs), outs


_VENDORED = [
    "secp256k1_read",
    "sqlite_read",
    "brainflayer_read",
    "ec_pubkey_diff",
    "hash_simd_diff",
]


def test_regression_vendored_so_noasan():
    """No-sanitizer vendored .so targets are _noasan.so, never _nosan.so."""
    targets_src = BUILD_SCRIPT.parent.parent / "targets"
    stubs = 'build_so_target() { echo "OUT $2"; }\nwarn() { :; }\nok() { :; }\n'
    stubs += "".join(f"{fn}() {{ return 0; }}\n" for fn in _COMPILERS)
    stubs += f'TARGETS=/b\nTARGETS_SRC="{targets_src}"\n'
    script = (
        stubs
        + _extract("build_standalone_so_targets")
        + '\nbuild_standalone_so_targets "_nosan" "" "L"\n'
    )
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr

    outs = [ln.split(" ", 1)[1] for ln in r.stdout.splitlines() if ln.startswith("OUT ")]
    named = [o for o in outs if any(f"/{t}_" in o for t in _VENDORED)]
    assert named and all(o.endswith("_noasan.so") for o in named), outs


_COMPILERS = (
    "compile_lz4_objects",
    "compile_secp256k1_objects",
    "compile_sqlite_objects",
    "compile_brainflayer_ec_objects",
    "compile_brainflayer_hash_objects",
    "compile_brainflayer_bloom_objects",
)
