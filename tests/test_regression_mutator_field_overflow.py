"""A format mutator must not raise on a header field at its type's limit.

Every one of these mutators reads a field out of the input, adds or
subtracts a delta, and packs it back at the same width. The input is
attacker-controlled by construction -- it is usually the mutator's *own*
previous output -- so the field arrives holding whatever the last mutation
left, including the maximum the type can hold. `struct.pack` then raises
rather than wrapping.

d84ca15 fixed one instance by masking (`(crc + delta) & 0xFFFF`). This
sweep found four more in cfhd and three more in shorten, including two
where a guard bounded the *field* and not the *sum*:

    if op == 0 and header.sample_rate < 0xFFFF:      # 0xFFFE + 8 still overflows
    if op == 0 and header.version > 0:               # 0 - 1 masked, not guarded

Measured before the fix, per 4000 mutations of one input:
cfhd 139 (3.5%) of its own generated seed, shorten 241 (6%) of a
max-valued header.

Driven rather than read statically: the arithmetic is spread over
`elif op == N` branches and only a real draw sequence reaches all of them.
The mutators are discovered from the package, so a format added later is
covered without editing this file.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil

import pytest

import fuzzer_tool.core.mutations as mutations_pkg
from fuzzer_tool.core.rand_pool import RandPool

_ITERATIONS = 3000


def _mutators():
    """Yield (module, class) for every mutator with a generator and mutate()."""
    for info in pkgutil.iter_modules(mutations_pkg.__path__):
        mod = importlib.import_module(f"{mutations_pkg.__name__}.{info.name}")
        for name, cls in vars(mod).items():
            if not inspect.isclass(cls) or cls.__module__ != mod.__name__:
                continue
            if not name.endswith("Mutator") or not hasattr(cls, "mutate"):
                continue
            if not any(n.startswith("_generate_random") for n in dir(cls)):
                continue
            try:
                cls(seed=1)
            except TypeError:
                continue  # takes a required rng: a different interface
            yield info.name, cls


_MUTATORS = list(_mutators())


def test_discovery_found_the_mutators():
    """Guard: an empty list would make the sweep below vacuous."""
    assert len(_MUTATORS) >= 15, f"only found {len(_MUTATORS)} mutators"


@pytest.mark.parametrize(
    "entry", _MUTATORS, ids=lambda e: f"{e[0]}.{e[1].__name__}" if isinstance(e, tuple) else str(e)
)
def test_mutating_its_own_generated_seed_never_raises(entry):
    """The generator writes deliberately extreme field values -- cfhd puts
    0xFFFFFFFF in slice_count -- so its output is the natural adversary for
    the header arithmetic downstream of it."""
    mod_name, cls = entry
    inst = cls(seed=1)
    gen = next(n for n in dir(cls) if n.startswith("_generate_random"))
    seed = getattr(inst, gen)(max_len=65536, rng=RandPool(seed=1))

    rng = RandPool(seed=7)
    for i in range(_ITERATIONS):
        try:
            inst.mutate(bytes(seed), max_len=65536, rng=rng)
        except Exception as exc:  # noqa: BLE001 - the assertion is "no exception"
            pytest.fail(f"{mod_name}.{cls.__name__} raised on iteration {i}: {exc!r}")


@pytest.mark.parametrize(
    "entry", _MUTATORS, ids=lambda e: f"{e[0]}.{e[1].__name__}" if isinstance(e, tuple) else str(e)
)
def test_mutating_an_all_ones_buffer_never_raises(entry):
    """Every fixed-width field at its maximum at once.

    A cheap way to reach the boundary for whichever offsets a given format
    happens to read, without hand-crafting a header per format. Most
    mutators will not parse it and take their unparseable branch, which is
    fine -- the ones that do parse it are exactly the ones whose deltas are
    about to leave the type.
    """
    mod_name, cls = entry
    inst = cls(seed=1)

    rng = RandPool(seed=13)
    for buf in (b"\xff" * 96, b"\x00" * 96):
        for i in range(_ITERATIONS // 4):
            try:
                inst.mutate(buf, max_len=65536, rng=rng)
            except Exception as exc:  # noqa: BLE001
                pytest.fail(
                    f"{mod_name}.{cls.__name__} raised on {buf[:1]!r}*96 iteration {i}: {exc!r}"
                )


def test_shorten_and_cfhd_boundary_headers_specifically():
    """The two the sweep above caught, pinned with their exact headers so a
    later change to the generators cannot quietly stop exercising them."""
    import struct

    from fuzzer_tool.core.mutations.cfhd import CfhdMutator
    from fuzzer_tool.core.mutations.shorten import ShnMutator

    shn = bytearray(64)
    struct.pack_into("<H", shn, 0, 0xFFFE)  # sample_rate: the `< 0xFFFF` guard
    struct.pack_into("B", shn, 2, 8)
    struct.pack_into("B", shn, 3, 0xFF)  # channels
    struct.pack_into("<H", shn, 6, 0xFFFF)  # frame_crc
    struct.pack_into("B", shn, 8, 4)

    rng = RandPool(seed=11)
    for frame_samples in (0x0000, 0xFFFF):  # both directions of the +/- delta
        struct.pack_into("<H", shn, 4, frame_samples)
        m = ShnMutator(seed=1)
        for _ in range(4000):
            m.mutate(bytes(shn), max_len=65536, rng=rng)

    cf = CfhdMutator(seed=1)
    seed = cf._generate_random_cfhd(max_len=65536, rng=RandPool(seed=1))
    for _ in range(4000):
        cf.mutate(bytes(seed), max_len=65536, rng=rng)
