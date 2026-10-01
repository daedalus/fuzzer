"""Tests for core/reducer.py — C-Reduce-style pass manager."""

from fuzzer_tool.core.reducer import ChunkPass, Oracle, Phase, Reducer, Verdict


def _bool_test(fn):
    """Lift a bool predicate into a Verdict test."""
    return lambda d: Verdict.PASS if fn(d) else Verdict.FAIL


def _needs(*tokens):
    """Interesting iff every token is present."""
    return _bool_test(lambda d: all(t in d for t in tokens))


def _old_ddmin(data, fn):
    """Pre-port minimize_bytes loop: restart at largest chunk on success."""
    calls = 0
    best = bytearray(data)
    while len(best) > 1:
        improved = False
        sizes = sorted({len(best) // 2 >> k for k in range(64)} - {0} | {1}, reverse=True)
        for size in sizes:
            off = 0
            while off + size <= len(best):
                cand = best[:off] + best[off + size :]
                if cand:
                    calls += 1
                    if fn(bytes(cand)):
                        best, improved = cand, True
                        break
                off += size
            if improved:
                break
        if not improved:
            break
    return bytes(best), calls


class _ReplacePass:
    """Toy pass: one candidate, ``data`` with every ``byte`` removed."""

    def __init__(self, byte, log=None):
        self._byte = byte
        self._log = log

    def new(self, data):
        return False

    def transform(self, data, done):
        if self._log is not None:
            self._log.append(self._byte)
        if done or self._byte not in data:
            return None
        return data.replace(self._byte, b""), True

    def advance(self, data, done):
        return True


class _ConstPass:
    """Adversarial pass: proposes ``cand`` ``times`` times."""

    def __init__(self, cand, times):
        self._cand = cand
        self._times = times

    def new(self, data):
        return 0

    def transform(self, data, i):
        if i >= self._times:
            return None
        return self._cand, i

    def advance(self, data, i):
        return i + 1


class TestChunkPass:
    def test_reaches_one_minimal(self):
        data = b"..X....Y.." * 4
        out = Reducer(Oracle(_needs(b"X", b"Y")), [(Phase.MAIN, ChunkPass())]).run(data)
        assert out == b"XY"

    def test_falsification_result_is_one_minimal(self):
        """Independent check: the result passes, every 1-byte deletion fails."""
        pred = lambda d: d.count(b"A") >= 3 and b"B" in d  # noqa: E731
        data = bytes(range(32)) + b"AxAyBzA" + bytes(range(64, 96))
        out = Reducer(Oracle(_bool_test(pred)), [(Phase.MAIN, ChunkPass())]).run(data)

        assert pred(out)
        assert not any(pred(out[:i] + out[i + 1 :]) for i in range(len(out)))

    def test_fewer_tests_than_restart_ddmin(self):
        """No rewind on success: strictly fewer oracle runs than the old loop."""
        needed = 8
        pred = lambda d: d.count(b"Q") >= needed  # noqa: E731
        data = b"".join(b"." * (61 + 7 * i) + b"Q" for i in range(needed)) + b"." * 40

        old_out, old_calls = _old_ddmin(data, pred)
        oracle = Oracle(_bool_test(pred))
        new_out = Reducer(oracle, [(Phase.MAIN, ChunkPass())]).run(data)

        assert new_out == old_out == b"Q" * needed
        assert oracle.runs < old_calls

    def test_never_proposes_empty(self):
        seen = []
        oracle = Oracle(lambda d: seen.append(d) or Verdict.PASS)
        out = Reducer(oracle, [(Phase.MAIN, ChunkPass())]).run(b"abcd")
        assert len(out) == 1
        assert b"" not in seen

    def test_empty_input_untouched(self):
        assert ChunkPass().transform(b"", ChunkPass().new(b"")) is None


class TestFixpoint:
    def test_main_repeats_until_no_progress(self):
        """'aXb': removing a is illegal while b exists; B unlocks A next sweep."""

        def pred(d):
            return b"X" in d and not (b"b" in d and b"a" not in d)

        passes = [(Phase.MAIN, _ReplacePass(b"a")), (Phase.MAIN, _ReplacePass(b"b"))]
        out = Reducer(Oracle(_bool_test(pred)), passes).run(b"aXb")
        assert out == b"X"

    def test_phase_order(self):
        log = []
        passes = [
            (Phase.LAST, _ReplacePass(b"c", log)),
            (Phase.MAIN, _ReplacePass(b"b", log)),
            (Phase.FIRST, _ReplacePass(b"a", log)),
        ]
        Reducer(Oracle(_needs(b"X")), passes).run(b"abcX")
        grouped = [b"a"] * log.count(b"a") + [b"b"] * log.count(b"b") + [b"c"] * log.count(b"c")
        assert log == grouped
        assert log.count(b"a") and log.count(b"b") and log.count(b"c")

    def test_max_steps_caps_accepted_reductions(self):
        reducer = Reducer(Oracle(_needs(b"X")), [(Phase.MAIN, ChunkPass())], max_steps=2)
        reducer.run(b"X" + b"." * 50)
        assert reducer.accepted == 2


class TestOracle:
    def test_cache_runs_each_candidate_once(self):
        runs = []
        oracle = Oracle(lambda d: runs.append(d) or Verdict.FAIL)
        Reducer(oracle, [(Phase.MAIN, _ConstPass(b"ab", 5))]).run(b"abcd")
        assert runs == [b"ab"]
        assert oracle.hits == 4

    def test_cache_bounded(self):
        oracle = Oracle(lambda d: Verdict.FAIL, capacity=4)
        for i in range(32):
            oracle(bytes([i]))
        assert oracle.cached <= 4

    def test_also_interesting_reported_not_accepted(self):
        """ALSO = different bug: hand it to on_also, keep reducing the original."""
        also = []

        def test(d):
            if b"X" in d:
                return Verdict.PASS
            return Verdict.ALSO if b"Z" in d else Verdict.FAIL

        oracle = Oracle(test, on_also=also.append)
        out = Reducer(oracle, [(Phase.MAIN, ChunkPass())]).run(b"XXXXZZZZ")
        assert out == b"X"
        assert also
        assert all(b"X" not in d and b"Z" in d for d in also)


class TestAdversarial:
    def test_non_shrinking_candidate_rejected_without_run(self):
        runs = []
        oracle = Oracle(lambda d: runs.append(d) or Verdict.PASS)
        out = Reducer(oracle, [(Phase.MAIN, _ConstPass(b"abcdef", 3))]).run(b"abc")
        assert out == b"abc"
        assert runs == []

    def test_always_true_oracle_terminates(self):
        out = Reducer(Oracle(lambda d: Verdict.PASS), [(Phase.MAIN, ChunkPass())]).run(bytes(4096))
        assert len(out) == 1

    def test_always_false_oracle_keeps_input(self):
        data = bytes(range(200))
        out = Reducer(Oracle(lambda d: Verdict.FAIL), [(Phase.MAIN, ChunkPass())]).run(data)
        assert out == data
