"""No randomness may bypass RandPool through OS entropy (Hard Rule 16).

``random.Random()``, ``np.random.default_rng()``, the module-level
``random`` functions and ``os.urandom`` all seed from the OS, so a call
site using them as a fallback made ``--seed`` runs irreproducible. Every
fallback now draws from ``get_default_rand_pool()``.

Out of scope: ``uuid.uuid4`` in ``core/cmplog.py`` names per-process
FIFOs; concurrent runs with one ``--seed`` need distinct names, and the
name drives no fuzzing decision.
"""

import ast
import os
import pathlib
import random
import zlib

import numpy as np
import pytest

from fuzzer_tool.core.rand_pool import RandPool, reset_default_rand_pool
from tests.test_regression_no_global_random import _make_fuzzer

SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "fuzzer_tool"
RANDOM_MODULES = {"random", "_random"}
OS_ENTROPY_ATTRS = {"urandom", "SystemRandom", "getrandom"}
POOL_SEED = 1234


@pytest.fixture(autouse=True)
def _restore_default_pool():
    """Leave the shared singleton as other tests expect."""
    yield
    reset_default_rand_pool()


# ── Static falsification: no OS-entropy source anywhere in src ───────────


def _is_module_fallback(node: ast.AST) -> bool:
    """``rng or random`` / ``rng if rng else random``: module as an rng."""
    if isinstance(node, ast.BoolOp):
        vals = node.values
    elif isinstance(node, ast.IfExp):
        vals = [node.body, node.orelse]
    else:
        return False
    return any(isinstance(v, ast.Name) and v.id in RANDOM_MODULES for v in vals)


def _is_unseeded_ctor(node: ast.AST) -> bool:
    """``random.Random()`` / ``np.random.default_rng()`` with no seed."""
    if not isinstance(node, ast.Call) or node.args or node.keywords:
        return False
    name = ast.unparse(node.func)
    return name.endswith(("random.Random", "default_rng")) or name == "Random"


def _is_os_entropy(node: ast.AST) -> bool:
    if isinstance(node, ast.Attribute):
        return node.attr in OS_ENTROPY_ATTRS
    return isinstance(node, ast.Name) and node.id == "secrets"


def _bypasses(path: pathlib.Path, root: pathlib.Path = SRC) -> list[str]:
    tree = ast.parse(path.read_text())
    hits = []
    for node in ast.walk(tree):
        if _is_module_fallback(node) or _is_unseeded_ctor(node) or _is_os_entropy(node):
            hits.append(f"{path.relative_to(root)}:{node.lineno}: {ast.unparse(node)}")
    return hits


def test_regression_no_os_entropy_in_src():
    hits = [h for p in sorted(SRC.rglob("*.py")) for h in _bypasses(p)]
    assert hits == []


def test_detector_flags_each_pattern(tmp_path):
    """Adversarial: the scanner must catch every form it claims to."""
    src = (
        "import os, random, secrets\nimport numpy as np\n"
        "a = rng or random\nb = rng if rng else random\nc = random.Random()\n"
        "d = np.random.default_rng()\ne = os.urandom(4)\nf = secrets.token_bytes(4)\n"
        "g = random.Random(0)\nh = np.random.default_rng(7)\n"
    )
    path = tmp_path / "probe.py"
    path.write_text(src)
    lines = sorted({int(h.split(":")[1]) for h in _bypasses(path, tmp_path)})
    assert lines == [3, 4, 5, 6, 7, 8]  # imports and seeded g/h (lines 9-10) pass


# ── Behavioral: fallbacks follow the default pool, not global random ─────


def _deflate_blob() -> bytes:
    return zlib.compress(bytes(range(256)) * 4)


def _run(call, pool_seed: int, global_seed: int):
    """Outputs of *call* after seeding the default pool and global random."""
    reset_default_rand_pool(pool_seed)
    random.seed(global_seed)
    return [call() for _ in range(8)]


def _field(mod):
    def call():
        buf = bytearray(16)
        mod._corrupt_field(buf, 4, 4)
        return bytes(buf)

    return call


def _fallback_calls():
    from fuzzer_tool.core import covering_array, feistel, gradient_cmp, magic_lock, token_shuffle
    from fuzzer_tool.core.mutations import (
        bmp,
        custom_mutator,
        deflate_struct,
        gzip,
        jpeg,
        tlv_mutate,
    )
    from fuzzer_tool.core.mutations import zlib as zlib_mut
    from fuzzer_tool.core.schedulers.seed_entropy_shapley import shapley_entropy

    data = bytes(range(64))
    text = b"alpha,beta;gamma delta=epsilon\nzeta"
    counts = np.arange(4 * 256, dtype=np.float64).reshape(4, 256) % 7
    return {
        "feistel": lambda: feistel.feistel_scramble(data),
        "tlv_mutate": lambda: tlv_mutate.tlv_mutate(data),
        "token_shuffle": lambda: token_shuffle.token_shuffle(text),
        "gradient_cmp": lambda: gradient_cmp.gradient_cmp(b"ABCD0000", [(b"ABCE", b"ABCD")]),
        "deflate_struct": lambda: deflate_struct.mutate_deflate_structure(_deflate_blob()),
        "magic_lock": lambda: magic_lock.format_lock_havoc(b"\x89PNG\r\n\x1a\n" + data, 256),
        "covering_array": lambda: covering_array.generate([[0, 1, 2]] * 4),
        "custom_havoc": lambda: custom_mutator.havoc(data, 128),
        "zlib_field": _field(zlib_mut),
        "gzip_field": _field(gzip),
        "bmp_field": _field(bmp),
        "jpeg_value": lambda: jpeg._corrupt_value(0x1234),
        "shapley": lambda: shapley_entropy(counts, counts.sum(axis=1)).__getitem__(0).tolist(),
    }


@pytest.mark.parametrize("name", sorted(_fallback_calls()))
def test_regression_fallback_ignores_global_random(name):
    call = _fallback_calls()[name]
    assert _run(call, POOL_SEED, 1) == _run(call, POOL_SEED, 2)


@pytest.mark.parametrize("name", sorted(_fallback_calls()))
def test_regression_fallback_follows_default_pool(name):
    """Falsification: a different pool seed must change the output."""
    call = _fallback_calls()[name]
    assert _run(call, 1, 0) != _run(call, 2, 0)


# ── format_seed_generator: RandPool crashed on randrange(a, b) ───────────


def test_regression_xor_byte_accepts_randpool():
    from fuzzer_tool.core.format_seed_generator import _op_xor_byte

    buf = bytearray(4)
    _op_xor_byte(buf, 0, 1, RandPool(seed=5))
    assert buf[0] == RandPool(seed=5).randint(1, 255)


def test_regression_xor_byte_never_zero():
    """Adversarial: the xor mask stays in [1, 255] for the stdlib rng too."""
    from fuzzer_tool.core.format_seed_generator import _op_xor_byte

    rng = random.Random(0)
    for _ in range(512):
        buf = bytearray(1)
        _op_xor_byte(buf, 0, 1, rng)
        assert 1 <= buf[0] <= 255


# ── Stall reseed without a campaign seed ─────────────────────────────────


def test_regression_unseeded_stall_seed_skips_urandom(monkeypatch):
    def boom(_n):
        raise AssertionError("os.urandom reached")

    f = _make_fuzzer(seed=None)
    monkeypatch.setattr(os, "urandom", boom)
    f._stall_reseed_count = 1
    first = f._derive_stall_seed()
    assert f._derive_stall_seed() == first

    f._stall_reseed_count = 2
    assert f._derive_stall_seed() != first


# ── Monte-Carlo null: seed=None passed straight to default_rng ───────────


def _js_pooled():
    return [({0: 5, 1: 3, 2: 2}, 5, 5), ({3: 4, 4: 6}, 4, 6)]


def test_regression_js_null_unseeded_follows_pool():
    from fuzzer_tool.core.schedulers.op_monte_carlo import MonteCarloScheduler

    run = MonteCarloScheduler._null_js_samples_numpy
    reset_default_rand_pool(POOL_SEED)
    first = run(_js_pooled(), 16)
    reset_default_rand_pool(POOL_SEED)
    assert run(_js_pooled(), 16) == first

    reset_default_rand_pool(POOL_SEED + 1)
    assert run(_js_pooled(), 16) != first


def test_regression_js_null_explicit_seed_unchanged():
    """Falsification: an explicit seed keeps its own stream, pool or not."""
    from fuzzer_tool.core.schedulers.op_monte_carlo import MonteCarloScheduler

    run = MonteCarloScheduler._null_js_samples_numpy
    reset_default_rand_pool(1)
    a = run(_js_pooled(), 16, seed=99)
    reset_default_rand_pool(2)
    assert run(_js_pooled(), 16, seed=99) == a
