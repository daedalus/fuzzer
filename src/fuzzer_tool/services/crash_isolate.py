"""Fuzz-time failure-inducing-combination isolation for a novel crash.

``crash_explain`` says which fields of a crashing input *changed*; this asks
which of them (and which values) the crash *needs*, by replaying the target
through ``core.failure_inducing`` (FIC) over the fields ``core.field_map``
names -- no user field spec required, unlike ``root_cause --isolate-fields``.

Cost is bounded because it runs once per novel crash signature: at most
``MAX_PROBES`` target executions, and only for inputs ``map_fields``
recognises (PNG, gzip, ...). Replays are standalone subprocesses; they never
touch the fuzzer's coverage maps, cmplog state or exec counters.
"""

from __future__ import annotations

import logging
import os
import random
from collections.abc import Callable
from typing import Any

from fuzzer_tool.core import field_spec
from fuzzer_tool.core.field_map import FieldKind, FieldSpan, map_fields

log = logging.getLogger(__name__)

MAX_PROBES = 64
VERIFY_SAMPLES = 4
MAX_FIELDS = 24  # probes scale with parameter count; keep the search small
MAX_INPUT_BYTES = 1 << 16
_INT_KINDS = frozenset(
    {
        FieldKind.VALUE,
        FieldKind.FLAGS,
        FieldKind.LENGTH,
        FieldKind.COUNT,
        FieldKind.OFFSET,
        FieldKind.TAG,
    }
)


def fields_from_spans(spans: list[FieldSpan]) -> list[field_spec.FieldDef]:
    """Integer-like, non-overlapping spans as ``FieldDef`` parameters (<= MAX_FIELDS)."""
    out: list[field_spec.FieldDef] = []
    end = 0
    for s in sorted(spans, key=lambda s: s.offset):
        if s.kind not in _INT_KINDS or not 1 <= s.width <= 8 or s.offset < end:
            continue
        # field_map names ("IHDR[0].bit_depth") are already unique per chunk;
        # suffix the offset only if a name repeats.
        name = s.name
        if any(f.name == name for f in out):
            name = f"{name}@0x{s.offset:x}"
        out.append(field_spec.FieldDef(name, s.offset, s.width, s.endian.value == "little"))
        end = s.offset + s.width
        if len(out) >= MAX_FIELDS:
            break
    return out


def isolate_crash(
    data: bytes,
    fields: list[field_spec.FieldDef],
    replay: Callable[[bytes], tuple[int, str]],
    returncode: int,
    *,
    sanitizer: str = "",
    seed: int = 0,
) -> tuple[dict[str, int], str]:
    """``({field: value}, status)`` for the fields the crash needs.

    *replay* runs the target on bytes -> ``(returncode, stderr)``. A replay
    counts as the same failure when the return code matches and, for a
    sanitizer crash, the sanitizer name appears in stderr. ``status`` is the
    FIC status string, or ``"no_fields"`` / ``"too_large"``.
    """
    if not fields:
        return {}, "no_fields"
    if len(data) > MAX_INPUT_BYTES:
        return {}, "too_large"

    def fails(candidate: bytes) -> bool:
        rc, stderr = replay(candidate)
        return rc == returncode and (not sanitizer or sanitizer.lower() in stderr.lower())

    schema = field_spec.isolate_fields_failure(
        data,
        fails,
        fields,
        rng=random.Random(seed),
        max_probes=MAX_PROBES,
        verify_samples=VERIFY_SAMPLES,
    )
    if schema is None:
        return {}, "too_short"
    named = {fields[i].name: v for i, v in sorted(schema.params.items())}
    return named, str(getattr(schema.status, "value", schema.status))


def _isolated_inprocess_runner(f: Any) -> Any:
    """A private, coverage-free, process-isolated runner for an in-process target.

    Replaying through ``f._inprocess_runner`` would reset its coverage bitmap
    and fault/register relay (both read again after ``save_crash``), and in
    ``--inprocess-direct`` mode a wild replay could kill the fuzzer. A
    subprocess-loader runner of the same target/function has neither problem.
    Built once and cached on the fuzzer.
    """
    runner = getattr(f, "_isolate_runner", None)
    if runner is None:
        from fuzzer_tool.adapters.inprocess import InProcessRunner

        src = f._inprocess_runner
        runner = InProcessRunner(
            src.target,
            function_name=src.function_name,
            timeout=f.timeout,
            direct=False,
            cov=False,
        )
        f._isolate_runner = runner
    return runner


def isolate_for_fuzzer(f: Any, meta: Any, data: bytes, returncode: int) -> None:
    """Fill ``meta.failure_schema*`` for a novel crash; never raises into the crash path.

    The reference failure is re-measured with the replay backend itself
    (return codes differ between backends: direct in-process reports
    ``128 + signal``, a subprocess reports ``-signal``). If the crash does not
    reproduce there, the status is ``not_reproduced`` and nothing is isolated.
    """
    try:
        fields = fields_from_spans(map_fields(data).spans)
        if not fields:
            meta.failure_schema_status = "no_fields"
            return
        env = os.environ.copy()

        if getattr(f, "_inprocess_runner", None) is not None:
            replay = _isolated_inprocess_runner(f).run_one
        else:

            def replay(candidate: bytes) -> tuple[int, str]:
                from fuzzer_tool.adapters import process

                if f.file_mode:
                    rc, err, _ = process.run_target_file(
                        f.target, candidate, f.timeout, str(f._tmp_dir), f.target_args, env=env
                    )
                else:
                    rc, err, _ = process.run_target_stdin(f.target, candidate, f.timeout, env=env)
                return rc, err

        ref_rc, ref_err = replay(data)
        if ref_rc == 0:
            meta.failure_schema_status = "not_reproduced"
            return
        sanitizer = getattr(meta, "sanitizer", "") or ""
        if sanitizer.lower() not in ref_err.lower():
            sanitizer = ""
        meta.failure_schema, meta.failure_schema_status = isolate_crash(
            data, fields, replay, ref_rc, sanitizer=sanitizer
        )
    except Exception:
        log.warning("crash failure isolation failed", exc_info=True)
