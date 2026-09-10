"""A format parser must reject a short buffer, not raise on it.

Every ``parse_*`` in ``core/mutations/`` is called on attacker-controlled
bytes -- that is the whole point -- and every caller treats a falsy return
as "not this format". So the contract is total: any input, no exception.

Two broke it, both from b9baf25:

``rasc.parse_rasc`` looped on ``pos + 8 <= len(data)`` while the body reads
``offset`` at ``pos + 5``, which needs nine bytes. An 8-byte buffer raised
``struct.error``. ``av1_rtp.parse_av1_obus`` guarded ``pos + 3`` and then
read ``data[pos + 3]`` on the two-byte-size path, so a 3-byte buffer whose
last byte has 0x80 set raised ``IndexError``.

Swept rather than case-by-case: the interesting inputs are short and cheap,
and a parser added later is covered without editing this file. The byte
patterns are chosen to reach the branches a zero fill never does -- 0xFF
and 0x82 set the continuation and flag bits that lead into the multi-byte
length paths.

Signalling failure by exception is allowed where that is the declared
contract -- ``deflate_struct.parse_deflate`` raises ``DeflateError`` and
says so -- but only with an error the parser's own module defines. That
distinction is the whole point: ``DeflateError`` is a deliberate verdict,
while ``struct.error`` and ``IndexError`` are an implementation detail
escaping, and the second is what both bugs above looked like. Allowing
``ValueError`` subclasses instead would have let ``struct.error`` through,
since it is one.
"""

from __future__ import annotations

import inspect
import pkgutil
from importlib import import_module

import pytest

import fuzzer_tool.core.mutations as mutations_pkg

# 0x00 and 0xFF are the all-clear / all-set extremes; 0x82 and 0x02 set the
# low flag bits that select variable-length header forms; 0x7F is the value
# just below every "continuation follows" test.
_PATTERNS = (b"\x00", b"\xff", b"\x82", b"\x02", b"\x7f", b"A")
_MAX_LEN = 96


def _parsers():
    for info in pkgutil.iter_modules(mutations_pkg.__path__):
        mod = import_module(f"{mutations_pkg.__name__}.{info.name}")
        for name, fn in vars(mod).items():
            if not name.startswith("parse_") or not inspect.isfunction(fn):
                continue
            if fn.__module__ != mod.__name__:
                continue  # re-export; tested where it is defined
            required = [
                p
                for p in inspect.signature(fn).parameters.values()
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty
            ]
            if len(required) != 1:
                continue
            # parse_dict_line takes AFL dictionary *text*; a bytes sweep
            # says nothing about it.
            if required[0].annotation not in (bytes, "bytes"):
                continue
            own_errors = tuple(
                v
                for v in vars(mod).values()
                if isinstance(v, type)
                and issubclass(v, BaseException)
                and v.__module__ == mod.__name__
            )
            yield info.name, name, fn, own_errors


_PARSERS = list(_parsers())


def test_discovery_found_the_parsers():
    """Guard: an empty list would make the sweep below vacuous."""
    assert len(_PARSERS) >= 30, f"only found {len(_PARSERS)} parsers"


@pytest.mark.parametrize(
    "entry", _PARSERS, ids=lambda e: f"{e[0]}.{e[1]}" if isinstance(e, tuple) else str(e)
)
def test_parser_never_raises_on_a_truncated_buffer(entry):
    mod_name, fn_name, fn, own_errors = entry
    for pattern in _PATTERNS:
        for n in range(_MAX_LEN):
            buf = pattern * n
            try:
                fn(buf)
            except own_errors:
                pass  # the module's own verdict, not a leak
            except Exception as exc:  # noqa: BLE001 - the assertion is "no exception"
                pytest.fail(
                    f"{mod_name}.{fn_name} raised {type(exc).__name__} on {pattern!r} * {n}: {exc}"
                )


def test_rasc_parses_back_what_its_own_generator_writes():
    """The record layout had drifted between the two halves of one module.

    ``_generate_random_rasc`` writes seq_num at offset 9; the parser read it
    at 8, so a generated 0x10000000 came back as 255 -- the top byte of the
    preceding ``offset`` field. The stride was short by the same byte, which
    would have desynchronised every record after the first.
    """
    from fuzzer_tool.core.mutations.rasc import RascMutator, parse_rasc
    from fuzzer_tool.core.rand_pool import RandPool

    generated = RascMutator(seed=1)._generate_random_rasc(max_len=65536, rng=RandPool(seed=1))
    chunks = parse_rasc(generated)

    assert chunks is not None, "the generator must produce something its own parser accepts"
    assert chunks[0].chunk_type == 0x00
    assert chunks[0].size == 0xFFFFFFFF
    assert chunks[0].offset == 0xFFFFFFFF
    assert chunks[0].seq_num == 0x10000000, (
        "seq_num must come from offset 9, not overlap the field before it"
    )
