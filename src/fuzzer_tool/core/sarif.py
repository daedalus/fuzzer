"""SARIF 2.1.0 log of sanitizer crashes, for code-scanning tools.

One result per crash cluster, anchored on the first project frame::

    crash .json sidecars ─► report_stacks(raw_stderr) ─► primary frame ─┐
                                                                         ├─► result
    exploitability ─► level, security-severity ─► rule ──────────────────┘

Pure: dicts in, dict out. Reading sidecars and writing the file is
``services.sarif_export``'s job.
"""

from __future__ import annotations

import posixpath
from enum import StrEnum
from typing import Any

from fuzzer_tool import __version__
from fuzzer_tool.core.sanitizer import Frame, report_stacks

SARIF_VERSION = "2.1.0"
SARIF_SCHEMA = "https://json.schemastore.org/sarif-2.1.0.json"
TOOL_NAME = "fuzzer-tool"
SRCROOT = "SRCROOT"  # uriBaseId the consumer resolves to its checkout
MAX_RESULTS = 5000  # GitHub code scanning per-run cap
FINGERPRINT_KEY = "crashCluster/v1"
UNKNOWN_RULE = "crash/unknown"

# Sanitizer runtime frames: never the bug's site.
_RUNTIME_PREFIXES = ("__asan", "__interceptor_", "__sanitizer", "__ubsan", "__msan", "__tsan")
_RUNTIME_DIRS = ("/compiler-rt/",)


class Level(StrEnum):
    """SARIF ``result.level``."""

    ERROR = "error"
    WARNING = "warning"
    NOTE = "note"


# exploitability tier -> (level, GitHub security-severity score 0-10)
_TIERS: dict[str, tuple[Level, float]] = {
    "CRITICAL": (Level.ERROR, 9.5),
    "HIGH": (Level.ERROR, 8.0),
    "MEDIUM": (Level.WARNING, 5.5),
    "LOW": (Level.NOTE, 3.0),
}
_DEFAULT_TIER = (Level.WARNING, 5.0)
_TIER_RANK = {t: i for i, t in enumerate(_TIERS)}


def _str(side: dict, key: str) -> str:
    """Sidecar string field; anything else reads as empty."""
    v = side.get(key)
    return v if isinstance(v, str) else ""


def _is_runtime(f: Frame) -> bool:
    return f.func.startswith(_RUNTIME_PREFIXES) or any(d in f.path for d in _RUNTIME_DIRS)


def _artifact(path: str, src_root: str) -> dict:
    """Relative to SRCROOT when *path* lies under *src_root*, else absolute.

    Paths are normalized first: ``/root/../etc`` is not under ``/root``.
    """
    norm = posixpath.normpath(path)
    if src_root and posixpath.isabs(norm):
        rel = posixpath.relpath(norm, src_root)
        if not rel.startswith(".."):
            return {"uri": rel, "uriBaseId": SRCROOT}

    if posixpath.isabs(norm):
        return {"uri": f"file://{norm}"}
    return {"uri": norm, "uriBaseId": SRCROOT} if src_root else {"uri": norm}


def _location(f: Frame, src_root: str) -> dict:
    """physicalLocation when the frame has a file, plus its function name."""
    loc: dict[str, Any] = {}
    if f.path:
        phys: dict[str, Any] = {"artifactLocation": _artifact(f.path, src_root)}
        if f.line > 0:
            region = {"startLine": f.line}
            if f.col > 0:
                region["startColumn"] = f.col
            phys["region"] = region
        loc["physicalLocation"] = phys

    name = f.func or f.module
    if name:
        loc["logicalLocations"] = [{"name": name, "kind": "function"}]
    return loc


def _primary(frames: list[Frame], src_root: str) -> Frame | None:
    """First project frame: under *src_root* if any, else first non-runtime with a file."""
    sited = [f for f in frames if f.path and not _is_runtime(f)]
    if src_root:
        root = src_root.rstrip("/") + "/"
        for f in sited:
            if posixpath.normpath(f.path).startswith(root):
                return f
    return sited[0] if sited else None


def _message(side: dict, sanitizer: str, error: str, site: Frame | None) -> str:
    """One line: "AddressSanitizer: heap-use-after-free READ of size 4 at 0x.. in foo"."""
    parts = [f"{sanitizer or 'crash'}: {error or 'unknown'}"]
    access, size = side.get("access_type"), side.get("access_size")
    if isinstance(access, str) and isinstance(size, int):
        parts.append(f"{access} of size {size}")
    addr = _str(side, "fault_addr")
    if addr:
        parts.append(f"at {addr}")
    if site and site.func:
        parts.append(f"in {site.func}")
    return " ".join(parts)


def _thread_flow(frames: list[Frame], src_root: str) -> dict:
    """Crash stack in execution order (outermost first), as SARIF wants."""
    steps = [{"location": _location(f, src_root)} for f in reversed(frames)]
    return {"threadFlows": [{"locations": steps}]}


def _result(side: dict, src_root: str, rules: dict[str, dict]) -> dict:
    """One SARIF result; registers its rule in *rules*."""
    sanitizer, error = _str(side, "sanitizer"), _str(side, "error_type")
    rule_id = f"{sanitizer}/{error}" if sanitizer and error else UNKNOWN_RULE
    tier = _str(side, "exploitability")
    level, score = _TIERS.get(tier, _DEFAULT_TIER)
    _add_rule(rules, rule_id, level, score)

    stacks = report_stacks(_str(side, "raw_stderr"))
    crash = stacks.get("crash", [])
    site = _primary(crash, src_root)

    # No project frame: anchor on the target binary, file-level.
    if site:
        locations = [_location(site, src_root)]
    else:
        target = _str(side, "target")
        locations = [_location(Frame("", target, 0, 0, ""), src_root)] if target else []

    res: dict[str, Any] = {
        "ruleId": rule_id,
        "ruleIndex": list(rules).index(rule_id),
        "level": level.value,
        "message": {"text": _message(side, sanitizer, error, site)},
        "locations": locations,
        "partialFingerprints": {FINGERPRINT_KEY: _str(side, "cluster_id") or rule_id},
        "properties": _properties(side),
    }
    if crash:
        res["codeFlows"] = [_thread_flow(crash, src_root)]
    if stacks:
        res["stacks"] = [
            {"message": {"text": key}, "frames": [{"location": _location(f, src_root)} for f in fr]}
            for key, fr in stacks.items()
        ]
    return res


def _add_rule(rules: dict[str, dict], rule_id: str, level: Level, score: float) -> None:
    """Register *rule_id*; its severity is the highest of its results."""
    rule = rules.get(rule_id)
    if rule is None:
        rules[rule_id] = {
            "id": rule_id,
            "name": rule_id.replace("/", "_"),
            "shortDescription": {"text": rule_id.replace("/", ": ")},
            "defaultConfiguration": {"level": level.value},
            "properties": {"tags": ["security", "fuzzing"], "security-severity": f"{score:.1f}"},
        }
        return

    props = rule["properties"]
    if score > float(props["security-severity"]):
        props["security-severity"] = f"{score:.1f}"
        rule["defaultConfiguration"]["level"] = level.value


def _properties(side: dict) -> dict:
    """Triage fields a reviewer needs to reproduce the crash."""
    keys = ("exploitability", "crash_input", "target", "parent_seed_hash", "cluster_id")
    props = {k: _str(side, k) for k in keys if _str(side, k)}
    ops = side.get("mutation_ops")
    if isinstance(ops, list) and ops:
        props["mutation_ops"] = [str(o) for o in ops]

    # --replay-n score, once finished; flaky = reproduced below 100%.
    repro = side.get("repro")
    if isinstance(repro, dict) and isinstance(repro.get("rate"), (int, float)):
        props["repro_rate"] = float(repro["rate"])
        props["flaky"] = bool(repro.get("flaky"))
    return props


def _rank(side: dict) -> int:
    """Sort key: CRITICAL first, unknown tiers last."""
    return _TIER_RANK.get(_str(side, "exploitability"), len(_TIER_RANK))


def build_sarif(sidecars: list[dict], src_root: str) -> dict:
    """SARIF log for crash *sidecars* (``CrashMetadata.to_dict()`` shape).

    Args:
        sidecars: Crash sidecar dicts; a ``crash_input`` key names the input.
        src_root: Absolute source root frames are made relative to; "" keeps
            absolute ``file://`` URIs.

    Returns:
        SARIF 2.1.0 log: one run, one result per cluster, most severe first,
        at most ``MAX_RESULTS``.
    """
    src_root = posixpath.normpath(src_root) if src_root else ""

    # One result per cluster; first sidecar wins.
    seen: dict[str, dict] = {}
    for side in sorted(sidecars, key=_rank):
        key = _str(side, "cluster_id") or str(len(seen))
        seen.setdefault(key, side)

    rules: dict[str, dict] = {}
    results = [_result(s, src_root, rules) for s in list(seen.values())[:MAX_RESULTS]]

    run: dict[str, Any] = {
        "tool": {
            "driver": {
                "name": TOOL_NAME,
                "version": __version__,
                "rules": list(rules.values()),
            }
        },
        "results": results,
    }
    if src_root:
        run["originalUriBaseIds"] = {SRCROOT: {"uri": f"file://{src_root.rstrip('/')}/"}}
    return {"$schema": SARIF_SCHEMA, "version": SARIF_VERSION, "runs": [run]}
