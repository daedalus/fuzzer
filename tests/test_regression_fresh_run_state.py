"""Regression tests for finding 11 — persisted state on a non-resume run.

`Fuzzer.__init__` gated `self._state_store.load()` on `self.resume`, which
reads as correct and is not: `StateStore.get()` lazy-loads whenever the store
has not been loaded yet, so skipping `load()` DEFERRED the read to the first
`get()` rather than preventing it. A second "fresh" campaign in an existing
corpus directory therefore inherited the previous one's Markov model, Elo
ratings and crash-MI counters — silently, and fatally for any A/B comparison
of schedules, since the second arm starts pre-trained by the first.

The lazy load is right for the standalone readers (`report.py`, `tmin.py`,
`cli/commands.py`), so the store keeps it and a fresh run opts out through
`start_empty()`.

Second half: the GA restore and banner were nested inside the
`if self._diff_target:` block. Two consequences, tested here against the
source since constructing a Fuzzer needs a built target.
"""

import ast
import inspect
import textwrap
from pathlib import Path

from fuzzer_tool.core.state_store import StateStore


class TestFreshRunDoesNotInherit:
    def test_get_lazy_loads_by_default(self):
        """Document the behaviour the fix works around, so a future change to
        `get()` that removes the lazy load doesn't silently make
        `start_empty()` look redundant."""
        store = StateStore(Path("/nonexistent-corpus-dir"))
        assert store._loaded is False

    def test_skipping_load_is_not_enough(self, tmp_path):
        prev = StateStore(tmp_path)
        prev.set("markov", {"trained": "campaign-1"})
        assert prev.save()

        # What `if self.resume: load()` alone leaves behind.
        fresh = StateStore(tmp_path)
        assert fresh.get("markov") == {"trained": "campaign-1"}

    def test_start_empty_isolates_the_run(self, tmp_path):
        prev = StateStore(tmp_path)
        prev.set("markov", {"trained": "campaign-1"})
        prev.set("elo", {"ratings": {"havoc": 1900}})
        assert prev.save()

        fresh = StateStore(tmp_path)
        fresh.start_empty()
        assert fresh.get("markov") is None
        assert fresh.get("elo") is None
        assert fresh.get("elo", default={}) == {}

    def test_start_empty_still_saves(self, tmp_path):
        prev = StateStore(tmp_path)
        prev.set("markov", {"trained": "campaign-1"})
        assert prev.save()

        fresh = StateStore(tmp_path)
        fresh.start_empty()
        fresh.set("markov", {"trained": "campaign-2"})
        assert fresh.save() is True

        after = StateStore(tmp_path)
        after.load()
        assert after.get("markov") == {"trained": "campaign-2"}

    def test_start_empty_blocks_legacy_json_migration(self, tmp_path):
        """The legacy-JSON path is the same hazard by another route."""
        (tmp_path / "markov.json").write_text('{"trained": "legacy"}')

        lazy = StateStore(tmp_path)
        assert lazy.get("markov") == {"trained": "legacy"}

        fresh = StateStore(tmp_path)
        fresh.start_empty()
        assert fresh.get("markov") is None

    def test_resume_still_loads(self, tmp_path):
        prev = StateStore(tmp_path)
        prev.set("markov", {"trained": "campaign-1"})
        assert prev.save()

        resumed = StateStore(tmp_path)
        resumed.load()
        assert resumed.get("markov") == {"trained": "campaign-1"}


def _run_source():
    from fuzzer_tool.services.fuzzer import Fuzzer

    return inspect.getsource(Fuzzer.run)


def _block_body(test_attr: str) -> str:
    """Return the source of every `if self.<test_attr>:` block in `run()`.

    Asserted against the AST rather than by constructing a Fuzzer, which
    needs a built target and a live SHM segment. What went wrong here was
    purely a matter of which block the statements sat in, and that is
    exactly what the AST shows.
    """
    tree = ast.parse(textwrap.dedent(_run_source()))
    bodies = [
        "\n".join(ast.unparse(stmt) for stmt in node.body)
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Attribute)
        and node.test.attr == test_attr
    ]
    assert bodies, f"no `if self.{test_attr}:` block found in run()"
    return "\n".join(bodies)


class TestGaRestoreNesting:
    def test_ga_restore_lives_under_ga_enabled(self):
        """`--ga --resume` without a differential target silently restarted
        the population at generation 0, because the only `from_dict` call was
        nested one block over."""
        body = _block_body("_ga_enabled")
        assert "self.ga.from_dict" in body
        assert "'ga'" in body or '"ga"' in body

    def test_differential_block_does_not_touch_ga(self):
        """`--differential-target` without `--ga` reached the GA banner with
        `self.ga` still None and died on `.pop_size` before the first
        execution."""
        body = _block_body("_diff_target")
        assert "self.ga" not in body

    def test_ga_restore_is_gated_on_resume(self):
        body = _block_body("_ga_enabled")
        assert "self.resume" in body
