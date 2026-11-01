"""``--custom-mutator``: libFuzzer-style LLVMFuzzerCustomMutator/CrossOver plugins.

Each test builds a real shared object with the system C compiler, so the
whole path is exercised: host forwarder for ``LLVMFuzzerMutate``, dlopen of a
plugin with an undefined symbol, ctypes calls in both directions.
"""

from __future__ import annotations

import random
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from fuzzer_tool.core.mutations import custom_mutator as cm
from fuzzer_tool.core.mutator_interface import MutationContext

CC = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
pytestmark = pytest.mark.skipif(CC is None, reason="no C compiler")

# Keeps a 4-byte magic and asks the host to mutate everything after it.
_MAGIC_PLUGIN = r"""
#include <stddef.h>
#include <stdint.h>
#include <string.h>
size_t LLVMFuzzerMutate(uint8_t *, size_t, size_t);
size_t LLVMFuzzerCustomMutator(uint8_t *d, size_t n, size_t max, unsigned seed) {
    (void)seed;
    if (n < 4) { memcpy(d, "MAGC", 4 < max ? 4 : max); return 4 < max ? 4 : max; }
    size_t m = LLVMFuzzerMutate(d + 4, n - 4, max - 4);
    memcpy(d, "MAGC", 4);
    return m + 4;
}
"""

# Returns the seed it was given as the first byte, then a fixed tail.
_SEED_PLUGIN = r"""
#include <stddef.h>
#include <stdint.h>
size_t LLVMFuzzerCustomMutator(uint8_t *d, size_t n, size_t max, unsigned seed) {
    (void)n; (void)max;
    d[0] = (uint8_t)seed; d[1] = (uint8_t)(seed >> 8);
    return 2;
}
"""

_CROSS_PLUGIN = r"""
#include <stddef.h>
#include <stdint.h>
#include <string.h>
size_t LLVMFuzzerCustomCrossOver(const uint8_t *a, size_t na, const uint8_t *b,
                                 size_t nb, uint8_t *out, size_t max, unsigned seed) {
    (void)seed;
    size_t h = na / 2, t = nb / 2;
    if (h + (nb - t) > max) return 0;
    memcpy(out, a, h);
    memcpy(out + h, b + t, nb - t);
    return h + (nb - t);
}
"""

_BAD_SIZE_PLUGIN = r"""
#include <stddef.h>
#include <stdint.h>
size_t LLVMFuzzerCustomMutator(uint8_t *d, size_t n, size_t max, unsigned seed) {
    (void)d; (void)n; (void)seed;
    return max + 1;
}
"""

_ZERO_PLUGIN = r"""
#include <stddef.h>
#include <stdint.h>
size_t LLVMFuzzerCustomMutator(uint8_t *d, size_t n, size_t max, unsigned seed) {
    (void)d; (void)n; (void)max; (void)seed;
    return 0;
}
"""

_NO_SYMBOL_PLUGIN = "int unrelated(void) { return 1; }\n"


def _build(tmp_path, name: str, src: str) -> str:
    c = tmp_path / f"{name}.c"
    so = tmp_path / f"{name}.so"
    c.write_text(src)
    proc = subprocess.run(
        [CC, "-shared", "-fPIC", "-o", str(so), str(c)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return str(so)


@pytest.fixture
def rng():
    return random.Random(1234)


@pytest.fixture(autouse=True)
def _clean_registry():
    cm.unregister_custom_mutators()
    yield
    cm.unregister_custom_mutators()


class TestHavoc:
    def test_bounds_and_change(self, rng):
        for n in (0, 1, 2, 5, 40, 300):
            for cap in (1, 8, 64, 512):
                data = bytes(rng.randrange(256) for _ in range(n))
                for _ in range(50):
                    out = cm.havoc(data, cap, rng)
                    assert len(out) <= cap

    def test_changes_most_inputs(self, rng):
        data = bytes(range(32))
        changed = sum(cm.havoc(data, 64, rng) != data for _ in range(200))
        assert changed > 180

    def test_empty_input_grows(self, rng):
        assert any(cm.havoc(b"", 16, rng) for _ in range(10))

    def test_zero_cap_gives_empty(self, rng):
        assert cm.havoc(b"abc", 0, rng) == b""


class TestPluginCalls:
    def test_mutator_keeps_magic_and_calls_back_into_havoc(self, tmp_path, rng):
        plugin = cm.CustomMutatorPlugin(_build(tmp_path, "magic", _MAGIC_PLUGIN))
        assert plugin.has_mutator and not plugin.has_crossover
        data = b"MAGC" + bytes(range(40))
        outs = [plugin.mutate(data, 128, rng) for _ in range(200)]
        assert all(o is not None and o.startswith(b"MAGC") and len(o) <= 128 for o in outs)
        # The tail only changes if LLVMFuzzerMutate reached the Python havoc.
        assert sum(o != data for o in outs) > 150

    def test_seed_comes_from_the_rng(self, tmp_path):
        plugin = cm.CustomMutatorPlugin(_build(tmp_path, "seed", _SEED_PLUGIN))
        a = [plugin.mutate(b"xx", 16, random.Random(7)) for _ in range(3)]
        b = [plugin.mutate(b"xx", 16, random.Random(7)) for _ in range(3)]
        assert a == b
        assert plugin.mutate(b"xx", 16, random.Random(8)) != a[0]

    def test_uncapped_uses_default_capacity(self, tmp_path, rng):
        plugin = cm.CustomMutatorPlugin(_build(tmp_path, "magic", _MAGIC_PLUGIN))
        out = plugin.mutate(b"MAGC" + b"A" * 10, 0, rng)
        assert out is not None and len(out) <= cm.DEFAULT_CAP

    def test_oversized_return_is_a_decline(self, tmp_path, rng):
        plugin = cm.CustomMutatorPlugin(_build(tmp_path, "bad", _BAD_SIZE_PLUGIN))
        assert plugin.mutate(b"abc", 16, rng) is None

    def test_zero_return_is_a_decline(self, tmp_path, rng):
        plugin = cm.CustomMutatorPlugin(_build(tmp_path, "zero", _ZERO_PLUGIN))
        assert plugin.mutate(b"abc", 16, rng) is None

    def test_crossover(self, tmp_path, rng):
        plugin = cm.CustomMutatorPlugin(_build(tmp_path, "cross", _CROSS_PLUGIN))
        assert plugin.has_crossover and not plugin.has_mutator
        assert plugin.crossover(b"AAAABBBB", b"CCCCDDDD", 64, rng) == b"AAAADDDD"
        # Embedded NULs survive the c_char_p hop.
        assert plugin.crossover(b"A\0A\0", b"\0\0C\0", 64, rng) == b"A\0C\0"
        assert plugin.crossover(b"AAAABBBB", b"CCCCDDDD", 3, rng) is None

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            cm.CustomMutatorPlugin(tmp_path / "nope.so")

    def test_no_entry_points(self, tmp_path):
        with pytest.raises(RuntimeError, match="exports neither"):
            cm.CustomMutatorPlugin(_build(tmp_path, "none", _NO_SYMBOL_PLUGIN))


class TestRegistration:
    def test_registers_only_exported_entry_points(self, tmp_path):
        from fuzzer_tool.core.operator_registry import REGISTRY

        muts = cm.register_custom_mutator(_build(tmp_path, "magic", _MAGIC_PLUGIN))
        assert [m.name for m in muts] == ["custom_mutator"]
        assert "custom_mutator" in REGISTRY.names()
        assert "custom_crossover" not in REGISTRY.names()

    def test_second_registration_rebinds(self, tmp_path, rng):
        first = cm.register_custom_mutator(_build(tmp_path, "seed", _SEED_PLUGIN))[0]
        second = cm.register_custom_mutator(_build(tmp_path, "zero", _ZERO_PLUGIN))[0]
        assert first is second
        assert first.mutate(b"abc", rng, 16) is None  # now the zero-return plugin

    def test_category_snapshot_includes_operator(self, tmp_path):
        from fuzzer_tool.core import operator_categories

        cm.register_custom_mutator(_build(tmp_path, "magic", _MAGIC_PLUGIN))
        assert operator_categories.category_of("custom_mutator") == "adaptive"

    def test_unregister_removes_both(self, tmp_path):
        from fuzzer_tool.core.operator_registry import REGISTRY

        cm.register_custom_mutator(_build(tmp_path, "magic", _MAGIC_PLUGIN))
        cm.register_custom_mutator(_build(tmp_path, "cross", _CROSS_PLUGIN))
        cm.unregister_custom_mutators()
        assert not {"custom_mutator", "custom_crossover"} & set(REGISTRY.names())

    def test_through_the_registry_adapter(self, tmp_path):
        """The path the fuzz loop takes: registry adapter -> mutate()."""
        from fuzzer_tool.core.operator_registry import _mutator_adapter

        mutator = cm.register_custom_mutator(_build(tmp_path, "magic", _MAGIC_PLUGIN))[0]
        f = SimpleNamespace(_rng=random.Random(5), max_len=96, corpus=[], dictionary=[])
        handler = _mutator_adapter(mutator, SimpleNamespace(f=f))
        data = b"MAGC" + b"hello world, hello world"
        results = [handler(bytearray(data), 0, data) for _ in range(50)]
        got = [r for r in results if r is not None]
        assert got and all(bytes(r).startswith(b"MAGC") and len(r) <= 96 for r in got)

    def test_crossover_op_needs_a_second_seed(self, tmp_path, rng):
        op = cm.register_custom_mutator(_build(tmp_path, "cross", _CROSS_PLUGIN))[0]
        assert op.name == "custom_crossover"
        solo = MutationContext(corpus=[b"AAAABBBB"])
        assert not op.is_available(solo, b"AAAABBBB")
        assert op.mutate(b"AAAABBBB", rng, 64, context=solo) is None
        pair = MutationContext(corpus=[b"AAAABBBB", b"CCCCDDDD"])
        assert op.is_available(pair, b"AAAABBBB")
        outs = {op.mutate(b"AAAABBBB", rng, 64, context=pair) for _ in range(20)}
        assert outs == {None, b"AAAADDDD"}  # None when it drew itself as the partner


class TestCli:
    def test_flag_is_declared_and_forwarded(self):
        from pathlib import Path

        from fuzzer_tool.cli import commands

        src = Path(commands.__file__).read_text()
        assert '"--custom-mutator"' in src
        assert 'custom_mutator=getattr(args, "custom_mutator", None)' in src
