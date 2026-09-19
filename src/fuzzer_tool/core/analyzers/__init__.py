"""Analyzer/detector components dispatched by :mod:`fuzzer_tool.core.analyzer_registry`.

Each module here is one pluggable analysis component -- a detector,
estimator, or online statistical tracker -- registered as an
``AnalyzerSpec`` in ``core/analyzer_registry.py`` and wired into
``Fuzzer.__init__`` via ``REGISTRY.wire_all(self)``.

Files follow the ``analyzer_<algo>.py`` naming convention, one module per
algorithm/technique (a few files host more than one registered analyzer
where the implementations share state or theory -- e.g.
``analyzer_critical_slowing.py`` holds both the ``csd`` and
``coverage_homogeneity`` specs). This mirrors ``core/mutations/`` and
``core/schedulers/`` as a themed subpackage of ``core``, consistent with
``core/operator_registry.py``'s mutation-operator dispatch pattern.
"""
