"""Regression: disassembly constants were capped in PYTHONHASHSEED order.

`extract_constants_pure` returned `list(set(...))[:256]`. Bytes hash per
interpreter, so a target with more than 256 constants got a different subset
in every profile build: rebuilding fuzzgoat's profile cache changed which
constants seeded the dictionary, and two `--seed 1 --clock virtual` runs
loading different caches split by exec 15.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import requires_clang

SRC = str(Path(__file__).resolve().parents[1] / "src")

# More distinct compare immediates than the cap, so truncation decides.
N_CONSTANTS = 400
CAP = 256
BASE, STRIDE = 0x13572468, 7919

_PROBE = """
import json, sys
from fuzzer_tool.core.elf import extract_constants_pure
print(json.dumps([c.hex() for c in extract_constants_pure(sys.argv[1])]))
"""


@pytest.fixture(scope="module")
def many_constants(tmp_path_factory):
    """A binary with N_CONSTANTS distinct 32-bit register-compare immediates.

    -O1: at -O0 clang compares against a stack slot, which the decoder skips.
    """
    d = tmp_path_factory.mktemp("consts")
    arms = "\n".join(
        f"    if (x == {BASE + i * STRIDE:#x}u) return {i % 100};" for i in range(N_CONSTANTS)
    )
    src = d / "t.c"
    src.write_text(
        f"int f(unsigned x) {{\n{arms}\n    return 0;\n}}\nint main(int c, char **v) {{ return f(c); }}\n"
    )
    exe = d / "t"
    subprocess.run(["clang", "-O1", "-fno-jump-tables", "-o", str(exe), str(src)], check=True)
    return str(exe)


def _extract(target: str, hash_seed: str) -> list[str]:
    env = {**os.environ, "PYTHONHASHSEED": hash_seed, "PYTHONPATH": SRC}
    out = subprocess.run(
        [sys.executable, "-c", _PROBE, target], env=env, capture_output=True, text=True, check=True
    )
    return json.loads(out.stdout)


@requires_clang
def test_regression_constants_ignore_hash_seed(many_constants):
    """Same binary, two hash seeds: same capped list, same order."""
    first = _extract(many_constants, "1")

    assert len(first) == CAP
    assert first == _extract(many_constants, "2")


@requires_clang
def test_falsify_control_same_seed_agrees(many_constants):
    """Control (Rule 46): one hash seed twice must agree, or the probe is broken."""
    assert _extract(many_constants, "7") == _extract(many_constants, "7")


def _objdump_order(target: str) -> list[int]:
    """Source indices of the compare immediates, in code-address order (objdump)."""
    asm = subprocess.run(["objdump", "-d", target], capture_output=True, text=True, check=True)
    index = {BASE + i * STRIDE: i for i in range(N_CONSTANTS)}
    seen: dict[int, None] = {}
    for m in re.finditer(r"\bcmp\s+\$0x([0-9a-f]+),", asm.stdout):
        i = index.get(int(m.group(1), 16))
        if i is not None:
            seen.setdefault(i, None)
    return list(seen)


@requires_clang
@pytest.mark.skipif(not shutil.which("objdump"), reason="objdump not installed")
def test_adversarial_cap_keeps_first_seen(many_constants):
    """The cap keeps constants in code order, as an independent disassembler sees it."""
    index = {(BASE + i * STRIDE).to_bytes(4, "little").hex(): i for i in range(N_CONSTANTS)}
    got = [index[h] for h in _extract(many_constants, "3") if h in index]
    expected = _objdump_order(many_constants)

    assert got, "no source constant survived the cap"
    assert got == expected[: len(got)]
