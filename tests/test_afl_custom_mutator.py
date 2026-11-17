"""``--afl-custom-mutator``: AFL++ ``afl_custom_*`` plugins (C ``.so`` or Python ``.py``).

Builds a real C plugin with the system compiler and writes a Python plugin
with the same semantics, so both backends run every hook end to end.
"""

from __future__ import annotations

import random
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from fuzzer_tool.adapters.filesystem import save_to_corpus
from fuzzer_tool.core.mutations import custom_mutator as cm
from fuzzer_tool.core.mutator_interface import MutationContext
from fuzzer_tool.services.corpus_manager import CorpusManager
from fuzzer_tool.services.fuzzer import _QUEUE_GET_MAX_SKIPS, Fuzzer
from fuzzer_tool.services.runner import TargetRunner
from tests.support.scripted_rng import ScriptedRng

CC = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
pytestmark = pytest.mark.skipif(CC is None, reason="no C compiler")

# fuzz: buf + "|" + add_buf, cut to max. post_process: "PP:" + buf, declines
# (0) on a leading 'X'. havoc: reverses in place. trim: drops the last byte
# while kept; the first rejected cut ends the session. queue_get: refuses
# inputs starting with 'S'. queue_new_entry: counts calls, keeps the new
# file's first byte. fuzz faults on a leading 0xFF.
_PLUGIN = r"""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef struct { uint8_t out[4096]; uint8_t tb[4096]; size_t tlen; int32_t steps, cur; } st;
static int g_news, g_new_byte = -1, g_had_orig, g_deinit;
int fz_test_news(void) { return g_news; }
int fz_test_new_byte(void) { return g_new_byte; }
int fz_test_had_orig(void) { return g_had_orig; }
void *afl_custom_init(void *afl, unsigned int seed) { (void)afl; (void)seed; return calloc(1, sizeof(st)); }
size_t afl_custom_fuzz(st *s, uint8_t *buf, size_t n, uint8_t **out, uint8_t *add, size_t addn, size_t max) {
    if (n && buf[0] == 0xFF) { *(volatile int *)0 = 1; }
    size_t m = 0;
    memcpy(s->out, buf, n); m = n;
    s->out[m++] = '|';
    memcpy(s->out + m, add, addn); m += addn;
    *out = s->out;
    return m < max ? m : max;
}
size_t afl_custom_post_process(st *s, uint8_t *buf, size_t n, uint8_t **out) {
    if (n && buf[0] == 'X') return 0;
    memcpy(s->out, "PP:", 3); memcpy(s->out + 3, buf, n);
    *out = s->out;
    return n + 3;
}
size_t afl_custom_havoc_mutation(st *s, uint8_t *buf, size_t n, uint8_t **out, size_t max) {
    (void)s; (void)max;
    for (size_t i = 0; i < n / 2; i++) { uint8_t t = buf[i]; buf[i] = buf[n - 1 - i]; buf[n - 1 - i] = t; }
    *out = buf;
    return n;
}
int32_t afl_custom_init_trim(st *s, uint8_t *buf, size_t n) {
    memcpy(s->tb, buf, n); s->tlen = n; s->cur = 0; s->steps = n > 1 ? (int32_t)n - 1 : 0;
    return s->steps;
}
size_t afl_custom_trim(st *s, uint8_t **out) { *out = s->tb; return s->tlen - 1; }
int32_t afl_custom_post_trim(st *s, uint8_t ok) {
    if (!ok) return s->steps;
    s->tlen--;
    return ++s->cur;
}
uint8_t afl_custom_queue_get(st *s, const uint8_t *fn) {
    (void)s;
    FILE *f = fopen((const char *)fn, "rb");
    int c = f ? fgetc(f) : EOF;
    if (f) fclose(f);
    return c != 'S';
}
uint8_t afl_custom_queue_new_entry(st *s, const uint8_t *fn, const uint8_t *orig) {
    (void)s;
    FILE *f = fopen((const char *)fn, "rb");
    g_new_byte = f ? fgetc(f) : -2;
    if (f) fclose(f);
    g_had_orig = orig != NULL;
    g_news++;
    return 0;
}
void afl_custom_deinit(st *s) { g_deinit++; free(s); }
"""

_PY_PLUGIN = """
state = {"news": 0, "new_byte": None, "orig": None, "tb": b"", "steps": 0, "cur": 0}

def init(seed):
    state["seed"] = seed

def fuzz(buf, add_buf, max_size):
    if buf[:1] == b"\\xff":
        raise ValueError("boom")
    return bytearray((bytes(buf) + b"|" + bytes(add_buf))[:max_size])

def post_process(buf):
    if buf[:1] == b"X":
        return bytearray()
    return bytearray(b"PP:" + bytes(buf))

def havoc_mutation(buf, max_size):
    return bytearray(bytes(buf)[::-1])

def init_trim(buf):
    state["tb"] = bytes(buf)
    state["cur"] = 0
    state["steps"] = len(buf) - 1 if len(buf) > 1 else 0
    return state["steps"]

def trim():
    return bytearray(state["tb"][:-1])

def post_trim(success):
    if not success:
        return state["steps"]
    state["tb"] = state["tb"][:-1]
    state["cur"] += 1
    return state["cur"]

def queue_get(filename):
    with open(filename, "rb") as fh:
        return fh.read(1) != b"S"

def queue_new_entry(filename_new, filename_orig):
    with open(filename_new, "rb") as fh:
        state["new_byte"] = fh.read(1)
    state["orig"] = filename_orig
    state["news"] += 1
    return False

def deinit():
    state["deinit"] = True
"""


class _IndexRng(ScriptedRng):
    """ScriptedRng whose randrange() also consumes the randints script."""

    def randrange(self, _n):
        return next(self._randints)


def _build(tmp_path, name: str, src: str) -> str:
    c = tmp_path / f"{name}.c"
    so = tmp_path / f"{name}.so"
    c.write_text(src)
    proc = subprocess.run(
        [CC, "-shared", "-fPIC", "-o", str(so), str(c)], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    return str(so)


def _py(tmp_path, src: str = _PY_PLUGIN) -> str:
    path = tmp_path / "afl_py_plugin.py"
    path.write_text(src)
    return str(path)


@pytest.fixture(params=["c", "py"])
def plugin(request, tmp_path):
    path = _build(tmp_path, "afl", _PLUGIN) if request.param == "c" else _py(tmp_path)
    p = cm.AflCustomMutator(path, seed=7)
    yield p
    p.deinit()


@pytest.fixture(autouse=True)
def _clean_registry():
    cm.unregister_custom_mutators()
    yield
    cm.unregister_custom_mutators()


class TestHooks:
    def test_fuzz_gets_add_buf_and_max_size(self, plugin):
        assert plugin.fuzz(b"abc", b"xy", 64) == b"abc|xy"
        assert plugin.fuzz(b"abcdef", b"zz", 4) == b"abcd"

    def test_post_process(self, plugin):
        assert plugin.post_process(b"data") == b"PP:data"

    def test_havoc_mutation(self, plugin):
        assert plugin.havoc_mutation(b"abcd", 64) == b"dcba"

    def test_queue_get_reads_the_file(self, plugin):
        assert plugin.queue_get(b"Seed") is False
        assert plugin.queue_get(b"Other") is True

    def test_trim_session(self, plugin):
        assert plugin.init_trim(b"abc") == 2
        assert plugin.trim() == b"ab"
        assert plugin.post_trim(True) == 1
        assert plugin.trim() == b"a"


class TestCQueueNewEntry:
    def test_file_holds_the_bytes(self, tmp_path):
        p = cm.AflCustomMutator(_build(tmp_path, "afl", _PLUGIN), seed=1)
        lib = p._backend._lib  # noqa: SLF001 - test accessors exported by the plugin
        p.queue_new_entry(b"Qdata", None)
        p.queue_new_entry(b"Rdata", b"Qdata")
        assert lib.fz_test_news() == 2
        assert lib.fz_test_new_byte() == ord("R")
        assert lib.fz_test_had_orig() == 1


class TestPyQueueNewEntry:
    def test_orig_is_none_for_a_root_seed(self, tmp_path):
        p = cm.AflCustomMutator(_py(tmp_path), seed=1)
        p.queue_new_entry(b"Qdata", None)
        state = p._backend._mod.state  # noqa: SLF001
        assert (state["news"], state["new_byte"], state["orig"]) == (1, b"Q", None)


class TestFalsification:
    def test_post_process_decline_is_none(self, plugin):
        """A 0-length post_process result means 'leave the input as is'."""
        assert plugin.post_process(b"Xdata") is None

    def test_missing_init_rejected(self, tmp_path):
        src = "#include <stddef.h>\nsize_t afl_custom_fuzz(void){return 0;}\n"
        with pytest.raises(RuntimeError, match="afl_custom_init"):
            cm.AflCustomMutator(_build(tmp_path, "noinit", src), seed=1)

    def test_init_only_rejected(self, tmp_path):
        src = "void *afl_custom_init(void *a, unsigned s){(void)a;(void)s;return (void*)1;}\n"
        with pytest.raises(RuntimeError, match="no afl_custom_"):
            cm.AflCustomMutator(_build(tmp_path, "initonly", src), seed=1)


class TestAdversarial:
    def test_c_fault_contained(self, tmp_path):
        p = cm.AflCustomMutator(_build(tmp_path, "afl", _PLUGIN), seed=1)
        assert p.fuzz(b"\xffboom", b"", 64) is None
        assert p.failures == {"SIGSEGV": 1}
        assert p.fuzz(b"ok", b"", 64) == b"ok|"

    def test_py_exception_contained(self, tmp_path):
        p = cm.AflCustomMutator(_py(tmp_path), seed=1)
        assert p.fuzz(b"\xffboom", b"", 64) is None
        assert p.failures == {"ValueError": 1}
        assert p.fuzz(b"ok", b"", 64) == b"ok|"

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            cm.AflCustomMutator(str(tmp_path / "nope.so"), seed=1)


class TestRegistration:
    def test_ops_for_exported_hooks(self, tmp_path):
        from fuzzer_tool.core.operator_registry import REGISTRY

        cm.register_afl_custom_mutator(_build(tmp_path, "afl", _PLUGIN), seed=1)
        assert {"afl_custom_fuzz", "afl_custom_havoc"} <= set(REGISTRY.names())
        cm.unregister_custom_mutators()
        assert not {"afl_custom_fuzz", "afl_custom_havoc"} & set(REGISTRY.names())

    def test_post_process_only_registers_no_ops(self, tmp_path):
        from fuzzer_tool.core.operator_registry import REGISTRY

        src = (
            "#include <stddef.h>\n#include <stdint.h>\n"
            "void *afl_custom_init(void *a, unsigned s){(void)a;(void)s;return (void*)1;}\n"
            "size_t afl_custom_post_process(void *d, uint8_t *b, size_t n, uint8_t **o)"
            "{(void)d; *o = b; return n;}\n"
        )
        cm.register_afl_custom_mutator(_build(tmp_path, "pp", src), seed=1)
        assert not {"afl_custom_fuzz", "afl_custom_havoc"} & set(REGISTRY.names())

    def test_fuzz_op_splices_a_corpus_entry(self, tmp_path):
        _, ops = cm.register_afl_custom_mutator(_build(tmp_path, "afl", _PLUGIN), seed=1)
        fuzz_op = next(o for o in ops if o.name == "afl_custom_fuzz")
        ctx = MutationContext(corpus=[b"aa", b"bb"])
        rng = _IndexRng(randints=[1])
        assert fuzz_op.mutate(b"aa", rng, 64, context=ctx) == b"aa|bb"
        # Alone in the corpus: no splice partner, empty add_buf.
        assert fuzz_op.mutate(b"aa", random.Random(0), 64, context=MutationContext()) == b"aa|"


class TestRunnerPostProcess:
    def test_target_sees_processed_bytes(self):
        seen: list[bytes] = []
        runner = object.__new__(TargetRunner)
        runner.run_target = lambda data: (seen.append(data), (0, ""))[1]
        runner.bind_post_process(lambda d: None if d.startswith(b"X") else b"PP:" + d)

        runner.run_target(b"abc")
        runner.run_target(b"Xabc")
        assert seen == [b"PP:abc", b"Xabc"]


class _QueueFuzzer:
    """Stand-in for Fuzzer: just the attributes _queue_gated reads."""

    def __init__(self, verdict):
        self._afl_custom = SimpleNamespace(queue_get=verdict)
        self._queue_skips = 0
        self.fuzzed: list[bytes] = []

    def _fuzz_ungated(self, data):
        self.fuzzed.append(data)
        return True


class TestQueueGet:
    def test_refused_seed_skipped(self):
        f = _QueueFuzzer(lambda d: not d.startswith(b"S"))
        assert Fuzzer._queue_gated(f, b"Skip") is False
        assert Fuzzer._queue_gated(f, b"Run") is True
        assert f.fuzzed == [b"Run"]

    def test_adversarial_refuse_all_falls_through(self):
        """A plugin refusing every seed must not stall the loop."""
        f = _QueueFuzzer(lambda d: False)
        results = [Fuzzer._queue_gated(f, b"S") for _ in range(_QUEUE_GET_MAX_SKIPS + 1)]
        assert results == [False] * _QUEUE_GET_MAX_SKIPS + [True]
        assert f.fuzzed == [b"S"]


class _Shm:
    def __init__(self):
        self.edges = {1}

    def get_edge_ids(self):
        return set(self.edges)

    def is_new_coverage_with_edges(self):
        return False, set(self.edges)


class _TrimFuzzer:
    def __init__(self, corpus_dir: Path, plugin):
        self.corpus_dir = corpus_dir
        self.corpus: list[bytes] = []
        self.seed_meta: dict[bytes, dict] = {}
        self.seen_hashes: set[str] = set()
        self.irreplaceable_hashes: set[str] = set()
        self.shm_cov = _Shm()
        self.ptrace_cov = None
        self._edge_tracker = self
        self._agg_cache_valid = True
        self._use_lineage = False
        self._afl_custom = plugin
        self.calls: list[bytes] = []
        self._runner = self

    def run_target(self, data):
        self.calls.append(data)
        self.shm_cov.edges = {1} if data.startswith(b"KEEP") else {2}
        return 0, ""

    def get_seed_edge_count(self, key):
        return 1

    def _is_crash(self, rc, stderr):
        return False

    def _confirm_new_coverage(self, data, shm, has_new, edge_ids):
        return has_new, edge_ids

    def _exec_distance(self):
        return None


class TestCustomTrim:
    def test_plugin_trim_replaces_default_walk(self, tmp_path, plugin):
        """KEEP + 16x (over the 16 B trim floor): the plugin drops one byte per
        step while the trace holds; KEE is rejected and post_trim ends it."""
        corpus_dir = tmp_path / "corpus"
        (corpus_dir / "seeds").mkdir(parents=True)
        f = _TrimFuzzer(corpus_dir, plugin)
        seed = b"KEEP" + b"x" * 16
        save_to_corpus(seed, corpus_dir, f.seen_hashes, None)
        f.corpus.append(seed)
        f.seed_meta[seed] = {"fuzz_count": 0, "lineage_depth": 0}

        CorpusManager(f).trim_new_coverage(seed, seed)

        assert f.calls == [seed[:n] for n in range(len(seed) - 1, 2, -1)]
        assert f.corpus == [b"KEEP"]


class TestCli:
    def test_flag_is_declared_and_forwarded(self):
        from fuzzer_tool.cli import commands

        src = Path(commands.__file__).read_text()
        assert '"--afl-custom-mutator"' in src
        assert 'afl_custom_mutator=getattr(args, "afl_custom_mutator", None)' in src
