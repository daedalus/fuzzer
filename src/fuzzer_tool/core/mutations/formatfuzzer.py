"""FormatFuzzer-backed structure-aware mutators.

Wraps external FormatFuzzer generators/parsers (Dutra, Gopinath, Zeller —
ACM TOSEM 2023) as ``MutatorBase`` operators so the Elo/bandit/MCTS
schedulers can use high-validity structural mutations.

Design (see ``docs/handover/handover_formatfuzzer_integration_2026-09-06.md``):

* One operator instance per template (``ff_png``, ``ff_zip``, …).
* Category ``"format"`` — peers of the hand-written ``*_chunk_mutate`` ops.
* Gated by ``--formatfuzzer`` (``context.formatfuzzer_enabled``). When the
  flag is off, or the corresponding binary is missing, ``is_available``
  returns False and the operator is invisible to schedulers.
* External process first (simple, no C++ build dependency). The concrete
  CLI is intentionally thin; adjust ``_run_ff`` once the upstream
  FormatFuzzer interface is confirmed in Phase 0 of the handover.

Smart mutation kinds (paper §6.2):
  abstract, replace, insert, delete, random
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path

from fuzzer_tool.core.mutator_interface import MutationContext, MutatorBase
from fuzzer_tool.core.operator_registry import REGISTRY

log = logging.getLogger(__name__)

# Default search path for FormatFuzzer binaries. Overridable via env or
# the --ff-bin-dir CLI flag (propagated through MutationContext later).
_DEFAULT_BIN_DIR = Path(
    os.environ.get("FORMATFUZZER_BIN", "/usr/local/lib/formatfuzzer")
)

# Templates enabled when --formatfuzzer is set and no explicit list is given.
_DEFAULT_TEMPLATES = ("png", "zip", "isobmff", "jpeg")

# Magic-byte prefixes used for the cheap is_available gate. Templates with
# None accept any input (useful for generation-from-scratch attempts).
_MAGICS: dict[str, bytes | None] = {
    "png": b"\x89PNG\r\n\x1a\n",
    "jpeg": b"\xff\xd8\xff",
    "jpg": b"\xff\xd8\xff",
    "zip": b"PK\x03\x04",
    "isobmff": None,  # ftyp box; more expensive to probe
    "mp4": None,
    "gif": b"GIF8",
    "bmp": b"BM",
    "wav": b"RIFF",
    "avi": b"RIFF",
    "pcap": b"\xd4\xc3\xb2\xa1",  # little-endian magic
    "midi": b"MThd",
}

_SMART_OPS = ("abstract", "replace", "insert", "delete", "random")

# Hard ceiling on the external process. FormatFuzzer is normally << 10 ms;
# anything longer is treated as a decline so a stuck binary cannot stall
# the fuzz loop.
_SUBPROCESS_TIMEOUT = 2.0


def _looks_like(template: str, data: bytes) -> bool:
    """Cheap magic-byte check. Returns True when no magic is registered."""
    if len(data) < 4:
        return False
    magic = _MAGICS.get(template)
    if magic is None:
        return True
    return data.startswith(magic)


class FormatFuzzerMutator(MutatorBase):
    """Smart structural mutations via an external FormatFuzzer binary.

    Parameters
    ----------
    template:
        Format name that maps to a binary (e.g. ``png`` → ``png_fuzzer``).
    name:
        Operator name exposed to schedulers. Defaults to ``ff_<template>``.
    bin_dir:
        Directory that contains the FormatFuzzer executables.
    """

    category = "format"

    def __init__(
        self,
        template: str = "png",
        name: str | None = None,
        bin_dir: Path | str | None = None,
    ) -> None:
        self.template = template.strip().lower()
        self.name = name or f"ff_{self.template}"
        self.bin_dir = Path(bin_dir) if bin_dir else _DEFAULT_BIN_DIR
        # Common naming conventions used by FormatFuzzer builds. The first
        # existing executable wins.
        candidates = [
            self.bin_dir / f"{self.template}_fuzzer",
            self.bin_dir / f"ff_{self.template}",
            self.bin_dir / self.template,
            self.bin_dir / f"formatfuzzer_{self.template}",
        ]
        self._bin: Path | None = next(
            (c for c in candidates if c.is_file() and os.access(c, os.X_OK)),
            None,
        )
        self._bin_available = self._bin is not None
        if not self._bin_available:
            log.debug(
                "FormatFuzzer binary for template %r not found under %s",
                self.template,
                self.bin_dir,
            )

    # ------------------------------------------------------------------
    # MutatorBase contract
    # ------------------------------------------------------------------

    def is_available(self, context: MutationContext, data: bytes) -> bool:
        """Gate on feature flag + binary presence + optional magic bytes."""
        if not getattr(context, "formatfuzzer_enabled", False):
            return False
        if not self._bin_available:
            return False
        # Allow very short inputs so pure generation (op=random) can still
        # fire; otherwise require a matching magic.
        if len(data) < 8:
            return True
        return _looks_like(self.template, data)

    def mutate(
        self,
        data: bytes,
        rng,
        max_len: int = 0,
        *,
        context: MutationContext | None = None,
        **kwargs,
    ) -> bytes | None:
        """Apply one FormatFuzzer smart mutation; return None on decline."""
        if not self._bin_available or self._bin is None:
            return None

        op = rng.choice(_SMART_OPS)
        try:
            return self._run_ff(data, op, max_len or 0, rng)
        except Exception:  # noqa: BLE001 — never break the fuzz loop
            log.debug(
                "FormatFuzzer %s op=%s failed",
                self.template,
                op,
                exc_info=True,
            )
            return None

    def on_new_coverage(self, seed: bytes, new_edges: int) -> None:
        # Reserved for later: per-(template, op) success counters that can
        # bias rng.choice. Keeping the hook present so notify_new_coverage
        # has a stable target.
        return None

    # ------------------------------------------------------------------
    # External process bridge
    # ------------------------------------------------------------------

    def _run_ff(
        self,
        data: bytes,
        op: str,
        max_len: int,
        rng,
    ) -> bytes | None:
        """Invoke the FormatFuzzer binary.

        The CLI contract below is a stable placeholder matching the
        handover sketch. Once Phase 0 confirms the real upstream interface,
        only this method needs to change.

        Expected behaviour of the binary:
          - exit 0 + stdout bytes  → success
          - non-zero / empty stdout → decline (return None)
        """
        assert self._bin is not None

        # Write input to a temp file (many FormatFuzzer builds expect a path).
        suffix = f".{self.template}"
        try:
            with tempfile.NamedTemporaryFile(
                suffix=suffix, delete=False
            ) as tmp:
                tmp.write(data)
                tmp_path = tmp.name
        except OSError:
            return None

        try:
            cmd = [
                str(self._bin),
                "--mutate",
                "--op",
                op,
                "--seed",
                str(rng.randint(0, 2**31 - 1)),
            ]
            if max_len > 0:
                cmd.extend(["--max-size", str(max_len)])
            cmd.append(tmp_path)

            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=_SUBPROCESS_TIMEOUT,
                check=False,
            )
            if result.returncode != 0 or not result.stdout:
                return None
            out = result.stdout
            if max_len > 0 and len(out) > max_len:
                out = out[:max_len]
            return out
        except subprocess.TimeoutExpired:
            log.debug(
                "FormatFuzzer %s timed out (>%ss)",
                self.template,
                _SUBPROCESS_TIMEOUT,
            )
            return None
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def __repr__(self) -> str:
        status = "ready" if self._bin_available else "no-binary"
        return (
            f"<FormatFuzzerMutator name={self.name!r} "
            f"template={self.template!r} {status}>"
        )


# ---------------------------------------------------------------------------
# Registration helpers
# ---------------------------------------------------------------------------

def register_formatfuzzer_mutators(
    templates: list[str] | tuple[str, ...] | None = None,
    bin_dir: Path | str | None = None,
) -> list[FormatFuzzerMutator]:
    """Register one FormatFuzzerMutator per template.

    Safe to call multiple times: existing names are skipped.
    Returns the list of mutator instances (new and pre-existing).
    """
    templates = templates or _DEFAULT_TEMPLATES
    registered: list[FormatFuzzerMutator] = []
    existing = set(REGISTRY.names())

    for t in templates:
        t = t.strip().lower()
        if not t:
            continue
        name = f"ff_{t}"
        if name in existing:
            # Already present — find the live instance if possible.
            for m in REGISTRY.mutators():
                if getattr(m, "name", None) == name:
                    registered.append(m)  # type: ignore[arg-type]
                    break
            continue
        mut = FormatFuzzerMutator(template=t, bin_dir=bin_dir)
        REGISTRY.register_mutator(mut)
        registered.append(mut)
        existing.add(name)
        log.info(
            "Registered FormatFuzzer operator %s (binary %s)",
            name,
            "found" if mut._bin_available else "MISSING",
        )
    return registered


def _register_on_import() -> None:
    """Import-time registration (mirrors weizz_structural / fractal_voronoi).

    Operators stay dark until ``--formatfuzzer`` sets
    ``context.formatfuzzer_enabled``. Binary absence also keeps them
    unavailable, so a missing install is harmless.
    """
    # Honour an explicit env list so tests / CI can restrict templates.
    env = os.environ.get("FORMATFUZZER_TEMPLATES")
    templates = (
        tuple(t.strip() for t in env.split(",") if t.strip())
        if env
        else _DEFAULT_TEMPLATES
    )
    register_formatfuzzer_mutators(templates)


_register_on_import()

__all__ = [
    "FormatFuzzerMutator",
    "register_formatfuzzer_mutators",
]
