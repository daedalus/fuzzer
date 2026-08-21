"""Regressions for CRITICAL findings still open after 1f6a296.

CRITICAL #1 and #4 (truncated shmat pointers) are covered upstream by
adapters/libc_shm.py and are not retested here. What remains is #2, #5, and
the corpus-discovery bug that kept #5 hidden.

Each test below fails against 1f6a296.
"""

from __future__ import annotations

import inspect
from pathlib import Path


class TestParallelSignature:
    """C2 -- cmd_fuzz passed kwargs run_parallel did not accept, so every
    `--jobs > 1` run died with TypeError before spawning a worker."""

    def test_run_parallel_accepts_every_kwarg_cmd_fuzz_sends(self):
        from fuzzer_tool.services.parallel import run_parallel

        params = inspect.signature(run_parallel).parameters
        assert not any(p.kind == p.VAR_KEYWORD for p in params.values()), (
            "a **kwargs catch-all would hide this class of drift"
        )
        for kwarg in (
            "contextual",
            "contextual_alpha",
            "contextual_lambda",
            "lineage",
            "lineage_backtrack",
            "asan_target",
            "ubsan_target",
            "chi2_operator_interval",
        ):
            assert kwarg in params, f"run_parallel does not accept {kwarg}"

    def test_forwarded_kwargs_are_accepted_by_fuzzer(self):
        """Plumbing them into run_parallel is useless if Fuzzer rejects them."""
        from fuzzer_tool.services.fuzzer import Fuzzer

        fuzzer_params = inspect.signature(Fuzzer.__init__).parameters
        for kwarg in ("contextual", "contextual_alpha", "contextual_lambda",
                      "lineage", "lineage_backtrack"):
            assert kwarg in fuzzer_params


class TestMinimizeCorpusDiscovery:
    """The bug that masked C5: minimize used a flat iterdir(), so it never
    descended into the seeds/<hh>/ shards the writer produces and reported
    'Corpus is empty' against every real corpus."""

    def _files(self, root: Path):
        from fuzzer_tool.services.minimize import _discover_corpus_files

        return {p.name for p in _discover_corpus_files(root)}

    def test_finds_sharded_layout(self, tmp_path):
        (tmp_path / "seeds" / "00").mkdir(parents=True)
        (tmp_path / "seeds" / "ab").mkdir(parents=True)
        (tmp_path / "seeds" / "00" / "id_a").write_bytes(b"a")
        (tmp_path / "seeds" / "ab" / "id_b").write_bytes(b"b")
        assert self._files(tmp_path) == {"id_a", "id_b"}

    def test_finds_flat_layout(self, tmp_path):
        (tmp_path / "id_a").write_bytes(b"a")
        (tmp_path / "id_b").write_bytes(b"b")
        assert self._files(tmp_path) == {"id_a", "id_b"}

    def test_skips_pruned_at_every_level(self, tmp_path):
        (tmp_path / "seeds" / "00").mkdir(parents=True)
        (tmp_path / "seeds" / "pruned").mkdir(parents=True)
        (tmp_path / "seeds" / "00" / "id_live").write_bytes(b"a")
        (tmp_path / "seeds" / "pruned" / "id_dead").write_bytes(b"b")
        assert self._files(tmp_path) == {"id_live"}

    def test_skips_metadata_files(self, tmp_path):
        (tmp_path / "id_a").write_bytes(b"a")
        (tmp_path / "notes.txt").write_text("x")
        (tmp_path / "run.log").write_text("x")
        (tmp_path / "meta.json").write_text("{}")
        assert self._files(tmp_path) == {"id_a"}


class TestMinimizeCoverageBlackout:
    """C5 -- an all-zero bitmap set makes every file look redundant, so set
    cover deletes the entire corpus. That is a broken measurement (usually an
    uninstrumented target), not a worthless corpus."""

    def test_blackout_refuses_to_prune(self, tmp_path, monkeypatch):
        import fuzzer_tool.services.minimize as M

        corpus = tmp_path / "seeds" / "00"
        corpus.mkdir(parents=True)
        for i in range(5):
            (corpus / f"id_{i}").write_bytes(f"seed-{i}".encode())

        # Stub both the attach and the execution. Patching shmat_checked alone
        # is enough to produce the blackout, but minimize_corpus would still
        # fork a real child -- and under the full suite that fork happens in a
        # multithreaded interpreter, which segfaults (report finding E3).
        # Nothing here needs a real subprocess: the guard under test keys off
        # the bitmaps, so stubbing the runner keeps the test hermetic.
        # The runners are imported inside the function, so patch them at the
        # source module rather than on M.
        import fuzzer_tool.adapters.process as P

        monkeypatch.setattr(M.libc_shm, "shmat", lambda *_a, **_k: None)
        monkeypatch.setattr(P, "run_target_stdin", lambda *_a, **_k: (0, ""))
        monkeypatch.setattr(P, "run_target_file", lambda *_a, **_k: (0, ""))

        kept, removed = M.minimize_corpus(
            "/bin/true", str(tmp_path), use_coverage=True, timeout=1.0
        )
        assert removed == 0, "a coverage blackout must not delete the corpus"
        assert kept == 5
        assert len(list(corpus.glob("id_*"))) == 5
        assert not (tmp_path / "seeds" / "pruned").exists()
