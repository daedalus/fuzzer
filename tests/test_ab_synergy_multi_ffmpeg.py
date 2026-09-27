"""tools/ab_synergy_multi_ffmpeg.py: plan, pairing and verdict logic.

Campaign execution and SHM replay are injected, so these run without
FFmpeg builds. Expected values derive from the synthetic inputs, not from
the code under test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import ab_synergy_multi_ffmpeg as ab  # noqa: E402

V = ["v8.0", "v8.1", "v9.0"]
MANIFEST = {"versions": V, "budget": 300}


# ── plan ──────────────────────────────────────────────────────────────


def test_plan_equal_total_budget():
    """Falsification: split arm spends exactly what multi spends, in total."""
    cells = ab.plan(V, seeds=[0, 1], budget=9000)

    for seed in (0, 1):
        multi = [c for c in cells if c.arm is ab.Arm.MULTI and c.seed == seed]
        split = [c for c in cells if c.arm is ab.Arm.SPLIT and c.seed == seed]
        assert len(multi) == 1
        assert multi[0].targets == tuple(V)
        assert sum(c.execs for c in split) == multi[0].execs == 9000
        assert sorted(c.targets[0] for c in split) == sorted(V)


def test_plan_full_and_control_arms():
    """FULL gives each version the whole budget; CONTROL repeats SPLIT on a disjoint seed."""
    cells = ab.plan(V, seeds=[3], budget=600)

    full = [c for c in cells if c.arm is ab.Arm.FULL]
    ctrl = [c for c in cells if c.arm is ab.Arm.CONTROL]
    split = [c for c in cells if c.arm is ab.Arm.SPLIT]
    assert {c.execs for c in full} == {600}
    assert [(c.targets, c.execs) for c in ctrl] == [(c.targets, c.execs) for c in split]
    assert all(c.run_seed != 3 for c in ctrl)
    assert all(c.run_seed == 3 for c in split)


@pytest.mark.parametrize(("versions", "budget"), [([], 900), (V, 2), (V, 0), (V, 5000)])
def test_plan_rejects_degenerate(versions, budget):
    """Adversarial: no targets, zero execs per version, or a budget V does not divide."""
    with pytest.raises(ValueError):
        ab.plan(versions, seeds=[0], budget=budget)


# ── scoring ───────────────────────────────────────────────────────────


def _results(multi, split, control=None, full=None):
    """Synthetic replay results: {cell_key: {version: edge-id set}}.

    *multi* / *split* / ... map seed -> {version: set}. A split corpus for
    version v is replayed on every version; only the given sets exist.
    """
    out = {}
    for arm, table in [
        (ab.Arm.MULTI, multi),
        (ab.Arm.SPLIT, split),
        (ab.Arm.CONTROL, control or split),
        (ab.Arm.FULL, full or split),
    ]:
        for seed, per in table.items():
            if arm is ab.Arm.MULTI:
                out[(arm, seed, None)] = per
                continue
            for owner in V:
                out[(arm, seed, owner)] = per[owner]
    return out


def _ids(n, offset=0):
    return frozenset(range(offset, offset + n))


def test_union_pools_split_corpora():
    """split_union on v = union of every split corpus replayed on v."""
    split = {0: {o: {v: _ids(10, offset=100 * V.index(o)) for v in V} for o in V}}
    res = _results(multi={0: {v: _ids(5) for v in V}}, split=split)

    scores = ab.score(res, V)

    assert scores[(ab.View.SPLIT_UNION, 0, "v9.0")] == 30
    assert scores[(ab.View.SPLIT_OWN, 0, "v9.0")] == 10
    assert scores[(ab.View.MULTI, 0, "v9.0")] == 5


def test_synergy_detected():
    """Falsification: multi beats pooled split by +50 on every seed -> significant win."""
    seeds = range(10)
    base = {s: {o: {v: _ids(100 + s) for v in V} for o in V} for s in seeds}
    multi = {s: {v: _ids(150 + s) for v in V} for s in seeds}

    rep = ab.analyse(ab.score(_results(multi, base), V), V)

    for v in V:
        row = rep[(ab.View.MULTI, ab.View.SPLIT_UNION, v)]
        assert row["wins"] == 10 and row["losses"] == 0
        assert row["median_delta"] == 50
        assert row["p"] < 0.05
    assert rep["control_ok"] is True


def test_identical_arms_null():
    """Control against itself: identical arms give p == 1 and no wins."""
    seeds = range(10)
    base = {s: {o: {v: _ids(100 + s) for v in V} for o in V} for s in seeds}
    multi = {s: {v: _ids(100 + s) for v in V} for s in seeds}

    rep = ab.analyse(ab.score(_results(multi, base), V), V)

    row = rep[(ab.View.MULTI, ab.View.SPLIT_UNION, "v8.0")]
    assert row["wins"] == row["losses"] == 0
    assert row["p"] == 1.0
    assert rep["control_ok"] is True


def test_broken_control_invalidates():
    """Adversarial: CONTROL differs from SPLIT systematically -> verdict unusable."""
    seeds = range(10)
    split = {s: {o: {v: _ids(100 + s) for v in V} for o in V} for s in seeds}
    ctrl = {s: {o: {v: _ids(300 + s) for v in V} for o in V} for s in seeds}
    multi = {s: {v: _ids(100 + s) for v in V} for s in seeds}

    rep = ab.analyse(ab.score(_results(multi, split, control=ctrl), V), V)

    assert rep["control_ok"] is False


def test_pairs_only_shared_seeds():
    """Adversarial: a seed missing from one arm is dropped, not zero-filled."""
    base = {s: {o: {v: _ids(100) for v in V} for o in V} for s in range(3)}
    multi = {s: {v: _ids(120) for v in V} for s in range(2)}

    rep = ab.analyse(ab.score(_results(multi, base), V), V)

    assert rep[(ab.View.MULTI, ab.View.SPLIT_UNION, "v8.0")]["n"] == 2


# ── execution wiring ──────────────────────────────────────────────────


def test_campaign_cmd_budget_and_targets(tmp_path):
    """Multi cell passes every target and --max-execs; single passes one."""
    cells = ab.plan(V, seeds=[7], budget=900)
    multi = next(c for c in cells if c.arm is ab.Arm.MULTI)
    split = next(c for c in cells if c.arm is ab.Arm.SPLIT)

    m = ab.campaign_cmd(multi, tmp_path / "c", tmp_path / "x")
    s = ab.campaign_cmd(split, tmp_path / "c", tmp_path / "x")

    assert all(t in m for t in V)
    assert m[m.index("--max-execs") + 1] == "900"
    assert s[s.index("--max-execs") + 1] == "300"
    assert s[s.index("-s") + 1] == "7"
    assert sum(t in s for t in V) == 1


def test_run_resumes_done_cells(tmp_path):
    """Resumable: a cell already in the results file is not re-run."""
    calls = []

    def fake_campaign(cell, workdir):
        calls.append(cell.key)
        corpus = workdir / "corpus"
        corpus.mkdir(parents=True)
        return corpus

    def fake_replay(target, corpus):
        return frozenset({1, 2, 3})

    out = tmp_path / "rows.pkl"
    cells = ab.plan(V, seeds=[0], budget=300)
    ab.run(cells, V, tmp_path / "work", out, fake_campaign, fake_replay, MANIFEST)
    first = len(calls)
    ab.run(cells, V, tmp_path / "work", out, fake_campaign, fake_replay, MANIFEST)

    assert first == len(cells)
    assert len(calls) == first
    assert len(ab.load(out)) == len(cells)
    assert not (tmp_path / "work").exists() or not any((tmp_path / "work").iterdir())


def _fake_replay(target, corpus):
    return frozenset({1})


def _fake_campaign(cell, workdir):
    corpus = workdir / "corpus"
    corpus.mkdir(parents=True)
    return corpus


def test_resume_rejects_other_manifest(tmp_path):
    """Adversarial: rows from another budget / binary set are never mixed in."""
    out = tmp_path / "rows.pkl"
    cells = ab.plan(V, seeds=[0], budget=300)
    ab.run(cells, V, tmp_path / "w", out, _fake_campaign, _fake_replay, MANIFEST)

    with pytest.raises(ValueError, match="manifest"):
        ab.run(
            cells, V, tmp_path / "w", out, _fake_campaign, _fake_replay, {**MANIFEST, "budget": 600}
        )


def test_fingerprint_tracks_binary_content(tmp_path):
    """A rebuilt binary at the same path changes the manifest."""
    b = tmp_path / "ffmpeg_read_1.0_asan"
    b.write_bytes(b"old")
    before = ab.fingerprint([str(b)])
    b.write_bytes(b"new")

    assert ab.fingerprint([str(b)]) != before
    assert ab.fingerprint([str(b)]) == ab.fingerprint([str(b)])


def test_failed_campaign_not_recorded(tmp_path, monkeypatch):
    """Adversarial: a non-zero fuzzer exit aborts before replay; the cell stays undone."""
    seeds = tmp_path / "seeds"
    seeds.mkdir()
    (seeds / "a.bin").write_bytes(b"x")
    monkeypatch.setattr(
        ab.subprocess,
        "run",
        lambda *a, **k: ab.subprocess.CompletedProcess(a, 1, "", "boom"),
    )
    out = tmp_path / "rows.pkl"
    cells = ab.plan(V, seeds=[0], budget=300)

    with pytest.raises(ab.CampaignError):
        ab.run(cells, V, tmp_path / "w", out, ab.make_campaign(seeds), _fake_replay, MANIFEST)
    assert not out.exists() or not ab.load(out)


def test_cli_no_versions_exits_2(tmp_path):
    """Adversarial: an empty --glob match is a clean exit 2, not a traceback."""
    import argparse

    seeds = tmp_path / "seeds"
    seeds.mkdir()
    args = argparse.Namespace(
        build_root=tmp_path,
        glob="nothing_*",
        seed_corpus=seeds,
        seeds=1,
        budget=300,
        out=tmp_path / "r.pkl",
        work=tmp_path / "w",
    )

    assert ab.cmd_run(args) == 2
