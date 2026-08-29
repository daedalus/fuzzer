"""Regression: crash triage enrichment runs only for crashes we keep.

``CorpusManager.save_crash`` used to build the full crash sidecar — nearest
corpus search over every seed, target hashing, a GDB replay of the crashing
input — and only then call ``filesystem.save_crash``, which drops the crash on
the floor when its signature is already known. Crashes are rare only until the
first bug is found; afterwards every mutation of the crashing seed arrives
here, so the discarded work was paid per crashing execution.

These tests pin the ordering: classify first, enrich only when novel, and keep
the recorded counters byte-identical to the pre-split behaviour.
"""

import types
from pathlib import Path

import pytest

from fuzzer_tool.adapters import filesystem as fs_mod
from fuzzer_tool.adapters.filesystem import classify_crash, save_crash
from fuzzer_tool.core import crash_metadata as cm_mod
from fuzzer_tool.services import corpus_manager as cm_service
from fuzzer_tool.services.corpus_manager import CorpusManager

SIG = "AddressSanitizer:heap-buffer-overflow@parse_header@main"

ASAN_STDERR = (
    "==1234==ERROR: AddressSanitizer: heap-buffer-overflow on address 0xdeadbeef\n"
    "    #0 0x4011a0 in parse_header /src/target.c:42\n"
    "    #1 0x401300 in main /src/target.c:99\n"
)


class _Fuzzer:
    """Minimal fuzzer surface read by CorpusManager.save_crash."""

    def __init__(self, tmp_path: Path):
        self.corpus_dir = tmp_path / "corpus"
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        self.crashes_dir = tmp_path / "crashes"
        self.crashes_dir.mkdir(parents=True, exist_ok=True)
        self.corpus = [b"seed-a" * 10, b"seed-b" * 10]
        self.seed_meta: dict = {}
        self.exec_count = 100
        self.target = "/bin/true"
        self._last_ops_used: list = []
        self._stats = types.SimpleNamespace(format_elapsed=lambda: "1s")
        self._last_regs = None
        self.ptrace_cov = None
        self._last_fault_addr = None
        self.crash_hashes: set[str] = set()
        self.crash_sigs: dict[str, int] = {}
        self.crash_frames: dict = {}
        self.crash_min_sizes: dict = {}
        self.save_smaller = False
        self.crash_blocklist: set = set()
        self.crash_allowlist: set = set()
        self.seen_hashes: set[str] = set()
        self.irreplaceable_hashes: set[str] = set()
        self.bloom = None


@pytest.fixture
def counted(monkeypatch):
    """Count the two expensive enrichment steps."""
    calls = {"nearest": 0, "gdb": 0}

    def fake_nearest(data, corpus, **kw):
        calls["nearest"] += 1
        return ("seed_0", 0.5, [1, 2, 3], "")

    def fake_gdb(f, data, returncode):
        calls["gdb"] += 1
        return "replay"

    monkeypatch.setattr(cm_mod, "find_nearest_corpus", fake_nearest)
    monkeypatch.setattr(cm_service, "_gdb_crash_replay", fake_gdb)
    return calls


class TestEnrichmentIsGatedOnNovelty:
    def test_novel_crash_is_enriched(self, tmp_path, counted):
        mgr = CorpusManager(_Fuzzer(tmp_path))
        assert mgr.save_crash(b"crash-input-1", -11, ASAN_STDERR)
        assert counted == {"nearest": 1, "gdb": 1}

    def test_repeat_signature_skips_enrichment(self, tmp_path, counted):
        f = _Fuzzer(tmp_path)
        mgr = CorpusManager(f)
        mgr.save_crash(b"crash-input-1", -11, ASAN_STDERR)
        counted["nearest"] = counted["gdb"] = 0

        # Same signature, different input bytes: this is the common case once
        # a crashing seed is in the corpus.
        for i in range(5):
            assert mgr.save_crash(b"crash-input-%d" % (i + 2), -11, ASAN_STDERR) is False
        assert counted == {"nearest": 0, "gdb": 0}
        assert f.crash_sigs[SIG] == 6

    def test_repeat_identical_input_skips_enrichment(self, tmp_path, counted):
        f = _Fuzzer(tmp_path)
        mgr = CorpusManager(f)
        mgr.save_crash(b"same-bytes", -11, ASAN_STDERR)
        counted["nearest"] = counted["gdb"] = 0
        assert mgr.save_crash(b"same-bytes", -11, ASAN_STDERR) is False
        assert counted == {"nearest": 0, "gdb": 0}
        # An exact input repeat is not counted as another sighting, matching
        # the behaviour before the split.
        assert f.crash_sigs[SIG] == 1

    def test_blocklisted_crash_skips_enrichment(self, tmp_path, counted):
        f = _Fuzzer(tmp_path)
        mgr = CorpusManager(f)
        verdict = classify_crash(b"blocked", -11, ASAN_STDERR, set(), {})
        f.crash_blocklist = {verdict.stack_hash}
        assert mgr.save_crash(b"blocked", -11, ASAN_STDERR) is False
        assert counted == {"nearest": 0, "gdb": 0}
        assert f.crash_sigs[verdict.signature] == 1

    def test_second_distinct_signature_is_enriched(self, tmp_path, counted):
        f = _Fuzzer(tmp_path)
        mgr = CorpusManager(f)
        mgr.save_crash(b"crash-input-1", -11, ASAN_STDERR)
        counted["nearest"] = counted["gdb"] = 0
        other = ASAN_STDERR.replace("parse_header", "decode_body").replace(
            "heap-buffer-overflow", "stack-overflow"
        )
        assert mgr.save_crash(b"crash-input-2", -11, other)
        assert counted == {"nearest": 1, "gdb": 1}

    def test_crash_frames_recorded_even_for_repeats(self, tmp_path, counted):
        """Repeats stay cheap, but the frame table must still be populated
        when the first sighting of a signature was filtered out elsewhere."""
        f = _Fuzzer(tmp_path)
        mgr = CorpusManager(f)
        mgr.save_crash(b"crash-input-1", -11, ASAN_STDERR)
        f.crash_frames.clear()
        mgr.save_crash(b"crash-input-2", -11, ASAN_STDERR)
        assert f.crash_frames[SIG]


class TestClassifyCrashIsPure:
    def test_classify_records_nothing(self):
        hashes: set[str] = set()
        sigs: dict[str, int] = {}
        v = classify_crash(b"x" * 32, -11, ASAN_STDERR, hashes, sigs)
        assert v.novel is True
        assert hashes == set() and sigs == {}

    def test_verdict_fields_for_repeat_signature(self):
        hashes: set[str] = set()
        sigs: dict[str, int] = {}
        first = classify_crash(b"a" * 32, -11, ASAN_STDERR, hashes, sigs)
        sigs[first.signature] = 1
        hashes.add(first.input_hash)
        second = classify_crash(b"b" * 32, -11, ASAN_STDERR, hashes, sigs)
        assert second.novel is False
        assert second.duplicate_input is False
        assert second.matched_signature == first.signature

    def test_fallback_signature_uses_fault_addr(self):
        v = classify_crash(b"z" * 16, -11, "", set(), {}, fault_addr=0xDEAD0000)
        assert v.signature == "signal:11@0xdead0000"
        assert v.report is None or not v.report.is_valid()


class TestSaveCrashHonoursPrecomputedVerdict:
    def test_verdict_is_reused_without_reparsing(self, tmp_path, monkeypatch):
        hashes: set[str] = set()
        sigs: dict[str, int] = {}
        v = classify_crash(b"q" * 32, -11, ASAN_STDERR, hashes, sigs)

        parses = {"n": 0}
        real_parse = fs_mod.SanitizerReport.parse

        def counting_parse(stderr):
            parses["n"] += 1
            return real_parse(stderr)

        monkeypatch.setattr(fs_mod.SanitizerReport, "parse", staticmethod(counting_parse))
        save_crash(b"q" * 32, -11, ASAN_STDERR, tmp_path, hashes, sigs, verdict=v)
        assert parses["n"] == 0
        assert sigs[v.signature] == 1
