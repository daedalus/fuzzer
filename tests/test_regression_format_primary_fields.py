"""Seed picker reads the primary cluster's fields without summarizing every cluster.

``_format_learner_seed`` called ``FormatLearner.get_format_summary()`` for
its ``fields`` only; that summarizes every cluster (the primary twice),
merging each hypothesis's per-position histograms: 2.8 ms per cluster, 1.2 s
of a 2000-exec ``--hail-mary`` profile. ``primary_fields()`` summarizes the
primary cluster alone.
"""

import random

from fuzzer_tool.core.analyzers.analyzer_format_learner import FormatCluster, FormatLearner


def _learner(seed: int, formats: int) -> FormatLearner:
    """Transitions across *formats* signatures so several clusters form."""
    rnd = random.Random(seed)
    fl = FormatLearner()
    for i in range(400):
        head = bytes([0x89 + rnd.randrange(formats), 0x50, 0x4E, 0x47])
        data = head + rnd.randbytes(28)
        fl.record_transition(
            input_bytes=data,
            mutation_op=rnd.choice(("bit_flip", "arithmetic", "havoc")),
            mutation_offset=rnd.randrange(32),
            mutation_width=rnd.choice((1, 2, 4)),
            coverage_before=10,
            coverage_after=10 + rnd.randrange(3),
            new_edges={100 + i} if rnd.random() < 0.3 else set(),
            lost_edges=set(),
        )
    return fl


def test_regression_format_primary_fields(monkeypatch):
    """One cluster summary per call, however many clusters exist."""
    fl = _learner(0, formats=4)
    assert len(fl.clusters) >= 2
    calls = []
    real = FormatCluster.get_format_summary

    def _counting(self):
        calls.append(self)
        return real(self)

    monkeypatch.setattr(FormatCluster, "get_format_summary", _counting)
    fl.primary_fields()
    assert calls == [fl.primary_cluster]


def test_primary_fields_equal_summary_fields():
    """Falsification: same fields as the full summary's top level."""
    for seed in range(6):
        fl = _learner(seed, formats=1 + seed % 4)
        assert fl.primary_fields() == fl.get_format_summary()["fields"]


def test_no_clusters_gives_no_fields():
    """Adversarial: an empty learner returns [] like the summary's empty shape."""
    fl = FormatLearner()
    assert fl.primary_fields() == fl.get_format_summary()["fields"] == []


def test_observation_tie_picks_the_summary_cluster():
    """Adversarial: tied clusters resolve to the one the summary ranks first."""
    fl = _learner(1, formats=3)
    top = max(c.total_observations for c in fl.clusters.values())
    for c in fl.clusters.values():
        c.total_observations = top
    assert fl.primary_fields() == fl.get_format_summary()["fields"]
