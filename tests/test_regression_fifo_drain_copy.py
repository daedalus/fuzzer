"""_collect_tokens_fifo slices only the lines it keeps.

It joined the carried partial line to the whole drain, sliced to the last
newline, then sliced again to CMPLOG_MAX_LINES_PER_READ lines: three
copies of drains up to ~2 MB, of which it kept ~half (2.7% of FFmpeg
fuzz-loop wall time). The cap is now found by counting newlines per chunk
before one join. Lines parsed and the carried partial must be unchanged.
"""

import random
import tracemalloc

import pytest

from fuzzer_tool.core import cmplog
from fuzzer_tool.core.cmplog import CmplogCollector, _first_lines


def _ref_fifo(state, data, cap):
    """Oracle: the pre-change reassembly, verbatim modulo self -> state."""
    if not data:
        return []
    data = state["partial"] + data
    if not data.endswith(b"\n"):
        last_nl = data.rfind(b"\n")
        if last_nl == -1:
            state["partial"] = data
            return []
        state["partial"] = data[last_nl + 1 :]
        data = data[: last_nl + 1]
    else:
        state["partial"] = b""
    data = _first_lines(data, cap)
    return data.decode("latin-1").splitlines()


class _Fifo:
    def __init__(self, chunks):
        self._chunks = iter(chunks)

    def drain(self):
        return next(self._chunks)


def _chunks(seed, n=40):
    """Random line stream cut at random points: mid-line, on a newline,
    empty drains, drains with no newline at all."""
    rnd = random.Random(seed)
    stream = b"".join(
        b"%d %s\n" % (i, rnd.randbytes(rnd.randint(0, 12)).hex().encode()) for i in range(3000)
    )
    cuts = sorted(rnd.sample(range(1, len(stream)), n - 1))
    parts = [stream[a:b] for a, b in zip([0, *cuts], [*cuts, len(stream)], strict=True)]
    parts[3:3] = [b"", b"no-newline-", b"still-none-"]
    return parts


def _run_new(chunks, cap, monkeypatch):
    monkeypatch.setattr(cmplog, "CMPLOG_MAX_LINES_PER_READ", cap)
    c = CmplogCollector()
    c._fifo = _Fifo(chunks)
    seen = []
    c._parse_lines = lambda lines: seen.append(lines) or []
    out = []
    for _ in chunks:
        before = len(seen)
        c._collect_tokens_fifo()
        out.append((seen[-1] if len(seen) > before else [], c._fifo_partial))
    return out


def _run_ref(chunks, cap):
    state = {"partial": b""}
    return [(_ref_fifo(state, ch, cap), state["partial"]) for ch in chunks]


def test_control_oracle_matches_itself():
    """Rule 46: the oracle agrees with a second run of itself."""
    ch = _chunks(1)
    assert _run_ref(ch, 50) == _run_ref(ch, 50)


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("cap", [1, 7, 50, 10_000])
@pytest.mark.parametrize("count_chunk", [1, 7, 4096])
def test_same_lines_and_partial(monkeypatch, seed, cap, count_chunk):
    """Falsification: per drain, identical parsed lines and carried tail,
    with the cap below, near and above the lines per drain, and the
    newline-count chunk splitting lines anywhere."""
    monkeypatch.setattr(cmplog, "LINES_END_CHUNK", count_chunk)
    ch = _chunks(seed)
    assert _run_new(ch, cap, monkeypatch) == _run_ref(ch, cap)


@pytest.mark.parametrize(
    "chunks",
    [
        [b"\n", b"\n\n", b"a\n"],
        [b"abc", b"def", b"\n"],
        [b"x" * 5, b"\nlast"],
        [b"one\ntwo\nthree\n"],
    ],
)
def test_degenerate_streams(monkeypatch, chunks):
    """Adversarial: blank lines, a line split over three drains, a drain
    starting with the newline that closes the carried line."""
    assert _run_new(chunks, 2, monkeypatch) == _run_ref(chunks, 2)


def test_dropped_tail_not_copied(monkeypatch):
    """Falsification: bytes past the cap are never copied (the old path
    joined the carried partial to the whole drain, then sliced it)."""
    monkeypatch.setattr(cmplog, "CMPLOG_MAX_LINES_PER_READ", 10)
    big = b"".join(b"line %d\n" % i for i in range(20_000))
    c = CmplogCollector()
    c._fifo = _Fifo([b"head-", big])
    c._parse_lines = lambda lines: []
    c._collect_tokens_fifo()  # carries b"head-"

    tracemalloc.start()
    try:
        c._collect_tokens_fifo()
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < len(big) // 4
