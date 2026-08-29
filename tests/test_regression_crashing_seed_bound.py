"""Regression: protected crashing seeds are bounded per crash signature.

Every distinct crashing input was written under ``corpus/seeds/crashing/``
and marked irreplaceable, which no pruning path may remove. A cheap-to-reach
bug produces a different input on every crashing execution, so the directory
grew without bound and could never be reclaimed.

A novel crash is still always preserved; repeats of a signature stop once
``CRASHING_SEEDS_PER_SIG`` samples exist.
"""

import types
from pathlib import Path

from fuzzer_tool.services import corpus_manager as cm_service
from fuzzer_tool.services.corpus_manager import CRASHING_SEEDS_PER_SIG, CorpusManager

ASAN_STDERR = (
    "==1==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x1\n"
    "    #0 0x4011a0 in parse_header /src/target.c:42\n"
    "    #1 0x401300 in main /src/target.c:99\n"
)


class _Fuzzer:
    def __init__(self, tmp_path: Path):
        self.corpus_dir = tmp_path / "corpus"
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        self.crashes_dir = tmp_path / "crashes"
        self.crashes_dir.mkdir(parents=True, exist_ok=True)
        self.corpus: list[bytes] = []
        self.seed_meta: dict = {}
        self.exec_count = 0
        self.target = "/bin/true"
        self._last_ops_used: list = []
        self._stats = types.SimpleNamespace(format_elapsed=lambda: "0s")
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


def _crashing_files(f) -> int:
    root = f.corpus_dir / "seeds" / "crashing"
    return sum(1 for p in root.rglob("*") if p.is_file()) if root.exists() else 0


def _mgr(tmp_path, monkeypatch):
    monkeypatch.setattr(cm_service, "_gdb_crash_replay", lambda *a, **k: "")
    f = _Fuzzer(tmp_path)
    return f, CorpusManager(f)


class TestCrashingSeedsAreBounded:
    def test_repeats_stop_at_the_cap(self, tmp_path, monkeypatch):
        f, mgr = _mgr(tmp_path, monkeypatch)
        for i in range(CRASHING_SEEDS_PER_SIG + 50):
            mgr.save_crash(b"crash-%05d" % i, -11, ASAN_STDERR)
        assert _crashing_files(f) == CRASHING_SEEDS_PER_SIG

    def test_novel_signature_is_always_preserved(self, tmp_path, monkeypatch):
        f, mgr = _mgr(tmp_path, monkeypatch)
        for i in range(CRASHING_SEEDS_PER_SIG + 10):
            mgr.save_crash(b"crash-%05d" % i, -11, ASAN_STDERR)
        before = _crashing_files(f)
        other = ASAN_STDERR.replace("heap-buffer-overflow", "stack-use-after-return")
        mgr.save_crash(b"different-bug", -11, other)
        assert _crashing_files(f) == before + 1

    def test_each_signature_has_its_own_budget(self, tmp_path, monkeypatch):
        f, mgr = _mgr(tmp_path, monkeypatch)
        # A different error type, not just a different frame: sanitizer
        # signatures are fuzzy-matched at 0.8, and two overflows differing
        # only in the innermost frame name fold into ONE signature.
        second = ASAN_STDERR.replace("heap-buffer-overflow", "stack-use-after-return")
        for i in range(CRASHING_SEEDS_PER_SIG + 20):
            mgr.save_crash(b"a-%05d" % i, -11, ASAN_STDERR)
            mgr.save_crash(b"b-%05d" % i, -11, second)
        assert _crashing_files(f) == 2 * CRASHING_SEEDS_PER_SIG

    def test_fuzzy_matched_signature_shares_the_budget(self, tmp_path, monkeypatch):
        """Adversarial: a crash folded into an existing signature by fuzzy
        matching is counted under THAT signature, so it must draw on that
        signature's budget rather than reading its own count of zero."""
        f, mgr = _mgr(tmp_path, monkeypatch)
        near = ASAN_STDERR.replace("parse_header", "decode_body")
        for i in range(CRASHING_SEEDS_PER_SIG + 40):
            mgr.save_crash(b"a-%05d" % i, -11, ASAN_STDERR)
            mgr.save_crash(b"b-%05d" % i, -11, near)
        assert len(f.crash_sigs) == 1  # the two signatures folded together
        assert _crashing_files(f) == CRASHING_SEEDS_PER_SIG

    def test_preserved_seeds_are_still_irreplaceable(self, tmp_path, monkeypatch):
        f, mgr = _mgr(tmp_path, monkeypatch)
        mgr.save_crash(b"first-crasher", -11, ASAN_STDERR)
        from fuzzer_tool.adapters.filesystem import hash_data

        assert hash_data(b"first-crasher") in f.irreplaceable_hashes
