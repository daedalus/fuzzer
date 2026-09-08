"""FormatFuzzer-backed structure-aware mutators.

Wraps external FormatFuzzer generators/parsers (Dutra, Gopinath, Zeller —
ACM TOSEM 2023) as ``MutatorBase`` operators so the Elo/bandit/MCTS
schedulers can use high-validity structural mutations.

Design (see ``docs/handover/handover_done_2026-09-06.md``):

* One operator instance per template (``ff_png``, ``ff_zip``, …).
* Category ``"format"`` — peers of the hand-written ``*_chunk_mutate`` ops.
* Gated by ``--formatfuzzer`` (``context.formatfuzzer_enabled``). When the
  flag is off, or the corresponding binary is missing, ``is_available``
  returns False and the operator is invisible to schedulers.
* External process (simple, no C++ build dependency).

Upstream CLI, confirmed against github.com/uds-se/FormatFuzzer -- the
first skeleton guessed at it and every guess was wrong, so the details
matter:

* the executable is ``<format>-fuzzer`` with a **hyphen** (``make
  gif-fuzzer`` / ``./build.sh gif``), not ``gif_fuzzer``;
* it takes a **command** as its first positional argument (``fuzz``,
  ``parse``, ``decode``).  There is no ``--mutate``, no ``--op`` and no
  ``--max-size``;
* output goes to a **file named on the command line**, never to stdout.

There is no single-shot mutate command.  Mutation is the decision-file
round trip the README describes:

    <fmt>-fuzzer parse --decisions in.dec input      # decisions -> in.dec
    (perturb bytes of in.dec)
    <fmt>-fuzzer fuzz  --decisions in.dec out        # decisions -> file

Each decision byte selects one parsing alternative, so perturbing a few
of them yields a *structurally valid* neighbour of the input -- which is
the whole point of the operator.  Pure generation is a single
``fuzz out`` with no decision file.

Cost: this is two fork+execs per mutation against a target that executes
in well under a millisecond, so the operator is deliberately gated and
the cost-aware schedulers are expected to price it accordingly.  The
cheap path is upstream's per-format shared library (``make gif.so``),
which is what AFL++ loads; wiring that is a separate piece of work.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from fuzzer_tool.core.mutator_interface import MutationContext, MutatorBase
from fuzzer_tool.core.operator_registry import REGISTRY

log = logging.getLogger(__name__)

# Default search path for FormatFuzzer binaries. Overridable via env or
# the --ff-bin-dir CLI flag (propagated through MutationContext later).
_DEFAULT_BIN_DIR = Path(os.environ.get("FORMATFUZZER_BIN", "/usr/local/lib/formatfuzzer"))

# Templates enabled when --formatfuzzer is set and no explicit list is given.
# The spellings are kept as they shipped so the operator names (and the
# import-time OPERATOR_CATEGORIES snapshot built from them) do not move;
# _TEMPLATE_ALIASES below is what makes them find the right binary.
_DEFAULT_TEMPLATES = ("png", "zip", "isobmff", "jpeg")

# Names people reach for that are not what upstream calls the template.
# The operator keeps the requested name (so a registry inventory taken at
# import time stays valid); only binary lookup and magic detection use the
# canonical one.  Upstream ships AVI, BMP, GIF, JPG, MIDI, MP3, MP4, PCAP,
# PNG, WAV and ZIP out of the box -- there is no "isobmff.bt" or "jpeg.bt",
# so the original defaults named two binaries that cannot exist.
_TEMPLATE_ALIASES = {
    "isobmff": "mp4",
    "mpeg4": "mp4",
    "jpeg": "jpg",
    "midi": "mid",
    "tiff": "tif",
}

# Magic-byte prefixes used for the cheap is_available gate. Templates with
# None accept any input (useful for generation-from-scratch attempts).
_MAGICS: dict[str, bytes | None] = {
    "png": b"\x89PNG\r\n\x1a\n",
    "jpg": b"\xff\xd8\xff",
    "zip": b"PK\x03\x04",
    "mp4": None,  # ftyp box lives at offset 4; more expensive to probe
    "gif": b"GIF8",
    "bmp": b"BM",
    "wav": b"RIFF",
    "avi": b"RIFF",
    "pcap": b"\xd4\xc3\xb2\xa1",  # little-endian magic
    "mid": b"MThd",
}

# How many decision bytes one mutation perturbs, as a fraction of the
# decision file.  Small: each byte is a whole parsing alternative, so a
# large perturbation is closer to regeneration than to mutation.
_DECISION_MUTATION_FRACTION = 0.05
_DECISION_MUTATIONS_MIN = 1
_DECISION_MUTATIONS_MAX = 16

# Hard ceiling on the external process. FormatFuzzer is normally << 10 ms;
# anything longer is treated as a decline so a stuck binary cannot stall
# the fuzz loop.
_SUBPROCESS_TIMEOUT = 2.0


def canonical_template(template: str) -> str:
    """Upstream's name for *template* (``isobmff`` -> ``mp4``, ``jpeg`` -> ``jpg``)."""
    t = template.strip().lower()
    return _TEMPLATE_ALIASES.get(t, t)


def _looks_like(template: str, data: bytes) -> bool:
    """Cheap magic-byte check. Returns True when no magic is registered."""
    if len(data) < 4:
        return False
    magic = _MAGICS.get(canonical_template(template))
    if magic is None:
        return True
    return data.startswith(magic)


def _resolve_binary(fmt: str, bin_dir: Path) -> Path | None:
    """Locate the FormatFuzzer executable for *fmt*.

    Upstream's own build (``make <fmt>-fuzzer`` / ``./build.sh <fmt>``)
    produces ``<fmt>-fuzzer`` with a hyphen; that name goes first.  The
    underscore and prefixed spellings are kept only because a packager
    might rename, and PATH is searched last because ``--formatfuzzer``'s
    help text promises it.
    """
    qualified = (
        f"{fmt}-fuzzer",
        f"{fmt}_fuzzer",
        f"ff_{fmt}",
        f"formatfuzzer_{fmt}",
    )
    # The bare format name is only honoured inside an explicitly named
    # directory. Searching PATH for it would happily pick up /usr/bin/zip,
    # /usr/bin/gif2webp and friends -- unrelated programs that exit 0 and
    # would be driven as if they were generators.
    for name in (*qualified, fmt):
        candidate = bin_dir / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    for name in qualified:
        found = shutil.which(name)
        if found:
            return Path(found)
    return None


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
        # Upstream's spelling, used for binary lookup and magic detection.
        # The operator name keeps the requested spelling so a registry
        # inventory taken at import time stays valid.
        self.format = canonical_template(self.template)
        self.name = name or f"ff_{self.template}"
        self._bin: Path | None = None
        self._bin_available = False
        self.rebind(bin_dir)

    def rebind(self, bin_dir: Path | str | None) -> bool:
        """Point this operator at *bin_dir* and re-resolve its executable.

        Registration is idempotent by name, so an operator registered at
        import time (with the default directory) used to keep that binding
        forever and ``--ff-bin-dir`` silently did nothing for every default
        template.  Re-resolving on registration is what makes the flag real.

        Returns True when an executable was found.
        """
        self.bin_dir = Path(bin_dir) if bin_dir else _DEFAULT_BIN_DIR
        self._bin = _resolve_binary(self.format, self.bin_dir)
        self._bin_available = self._bin is not None
        # Rebinding invalidates the probe: a different path is a different
        # program and has to earn its place again.
        self._probed: bool | None = None
        if not self._bin_available:
            log.debug(
                "FormatFuzzer binary for template %r (upstream name %r) not "
                "found under %s or on PATH",
                self.template,
                self.format,
                self.bin_dir,
            )
        return self._bin_available

    # ------------------------------------------------------------------
    # MutatorBase contract
    # ------------------------------------------------------------------

    def _probe(self) -> bool:
        """One-time check that the resolved binary really is a FormatFuzzer.

        A path match is not proof: an executable called ``zip`` or ``png``
        in the search directory is an unrelated program that would be
        driven as if it were a generator. Require the thing to honour the
        upstream contract -- ``fuzz OUT`` exits 0 and leaves bytes in OUT --
        before letting it near the fuzz loop. Cached; a failure disables the
        operator for the rest of the run rather than re-probing per mutation.
        """
        if self._probed is not None:
            return self._probed
        self._probed = False
        if self._bin is None:
            return False
        tmpdir = tempfile.mkdtemp(prefix="ff_probe_")
        try:
            out = Path(tmpdir) / f"probe.{self.format}"
            if self._exec(["fuzz", str(out)]) and out.is_file() and out.stat().st_size:
                self._probed = True
            else:
                log.warning(
                    "FormatFuzzer: %s did not behave like a generator "
                    "('fuzz OUT' produced nothing); disabling %s",
                    self._bin,
                    self.name,
                )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
        return self._probed

    def is_available(self, context: MutationContext, data: bytes) -> bool:
        """Gate on feature flag + a working binary + optional magic bytes."""
        if not getattr(context, "formatfuzzer_enabled", False):
            return False
        if not self._bin_available or not self._probe():
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
        """Apply one FormatFuzzer structural mutation; return None on decline."""
        if not self._bin_available or self._bin is None:
            return None
        try:
            return self._run_ff(data, max_len or 0, rng)
        except Exception:  # noqa: BLE001 — never break the fuzz loop
            log.debug("FormatFuzzer %s failed", self.format, exc_info=True)
            return None

    def on_new_coverage(self, seed: bytes, new_edges: int) -> None:
        # Reserved for later: per-template success counters that can bias
        # the generate-vs-mutate split. Keeping the hook present so
        # notify_new_coverage has a stable target.
        return None

    # ------------------------------------------------------------------
    # External process bridge
    # ------------------------------------------------------------------

    def _exec(self, args: list[str]) -> bool:
        """Run the fuzzer binary; True on a clean exit.

        The binary reports through its exit status and writes its output to
        the file it was given.  Nothing useful arrives on stdout, so nothing
        reads stdout -- the first cut did, which is why it declined every
        single mutation even when the binary ran.
        """
        try:
            result = subprocess.run(
                [str(self._bin), *args],
                capture_output=True,
                timeout=_SUBPROCESS_TIMEOUT,
                check=False,
            )
        except subprocess.TimeoutExpired:
            log.debug(
                "FormatFuzzer %s timed out (>%ss): %s",
                self.format,
                _SUBPROCESS_TIMEOUT,
                args,
            )
            return False
        except OSError:
            return False
        return result.returncode == 0

    def _perturb_decisions(self, dec: bytearray, rng) -> None:
        """Flip a few parsing decisions in place.

        Each byte selects one parsing alternative, so a handful of changed
        bytes is a structurally valid neighbour; changing many is closer to
        regeneration than to mutation.
        """
        n = len(dec)
        if n == 0:
            return
        count = int(n * _DECISION_MUTATION_FRACTION)
        count = max(_DECISION_MUTATIONS_MIN, min(_DECISION_MUTATIONS_MAX, count))
        for _ in range(count):
            idx = rng.randrange(n)
            # Small deltas keep the choice near the original alternative;
            # a full random byte is the occasional larger jump.
            if rng.random() < 0.75:
                dec[idx] = (dec[idx] + rng.choice((-2, -1, 1, 2))) & 0xFF
            else:
                dec[idx] = rng.randrange(256)

    def _run_ff(self, data: bytes, max_len: int, rng) -> bytes | None:
        """One decision-file round trip, or a from-scratch generation.

        ``parse --decisions D IN`` records the alternatives taken while
        parsing IN; perturbing D and running ``fuzz --decisions D OUT``
        replays them, so OUT is a valid file near IN.  When IN does not
        parse -- or when there is no meaningful IN -- fall back to a plain
        ``fuzz OUT``, which is the generator upstream advertises.
        """
        assert self._bin is not None

        tmpdir = tempfile.mkdtemp(prefix="ff_")
        try:
            base = Path(tmpdir)
            src = base / f"input.{self.format}"
            dec = base / "decisions.bin"
            out = base / f"output.{self.format}"

            mutated = False
            if len(data) >= 8:
                try:
                    src.write_bytes(data)
                except OSError:
                    return None
                if self._exec(["parse", "--decisions", str(dec), str(src)]) and (
                    dec.is_file() and dec.stat().st_size > 0
                ):
                    buf = bytearray(dec.read_bytes())
                    self._perturb_decisions(buf, rng)
                    dec.write_bytes(bytes(buf))
                    mutated = self._exec(["fuzz", "--decisions", str(dec), str(out)])

            if not mutated and not self._exec(["fuzz", str(out)]):
                return None

            if not out.is_file():
                return None
            try:
                result = out.read_bytes()
            except OSError:
                return None
            if not result:
                return None
            if max_len > 0 and len(result) > max_len:
                result = result[:max_len]
            return result
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def __repr__(self) -> str:
        status = "ready" if self._bin_available else "no-binary"
        return (
            f"<FormatFuzzerMutator name={self.name!r} "
            f"template={self.template!r} format={self.format!r} {status}>"
        )


# ---------------------------------------------------------------------------
# Registration helpers
# ---------------------------------------------------------------------------


def register_formatfuzzer_mutators(
    templates: list[str] | tuple[str, ...] | None = None,
    bin_dir: Path | str | None = None,
) -> list[FormatFuzzerMutator]:
    """Register one FormatFuzzerMutator per template.

    Safe to call multiple times.  A name that already exists is *rebound*
    to *bin_dir* rather than skipped: the module self-registers the default
    templates at import time with the default directory, so skipping made
    ``--ff-bin-dir`` a no-op for exactly the templates people use.

    Returns the list of mutator instances (new and pre-existing).
    """
    templates = templates or _DEFAULT_TEMPLATES
    registered: list[FormatFuzzerMutator] = []
    by_name = {getattr(m, "name", None): m for m in REGISTRY.mutators()}

    for raw in templates:
        t = raw.strip().lower()
        if not t:
            continue
        name = f"ff_{t}"
        existing = by_name.get(name)
        if existing is not None:
            if bin_dir is not None and hasattr(existing, "rebind"):
                existing.rebind(bin_dir)
            registered.append(existing)  # type: ignore[arg-type]
            continue
        mut = FormatFuzzerMutator(template=t, bin_dir=bin_dir)
        REGISTRY.register_mutator(mut)
        registered.append(mut)
        by_name[name] = mut
        log.debug(
            "Registered FormatFuzzer operator %s (binary %s)",
            name,
            mut._bin if mut._bin_available else "MISSING",
        )
    return registered


def report_availability(mutators: list[FormatFuzzerMutator]) -> int:
    """Log which operators found a binary; return how many did.

    Called once from ``Fuzzer.__init__`` when ``--formatfuzzer`` is set.
    It exists because the registration path is silent by design (it runs on
    every import), so enabling the feature with nothing installed used to
    produce no message at all and four operators that never fired.
    """
    ready = [m for m in mutators if getattr(m, "_bin_available", False)]
    if ready:
        log.info(
            "FormatFuzzer: %d/%d operator(s) ready (%s)",
            len(ready),
            len(mutators),
            ", ".join(sorted(m.name for m in ready)),
        )
    if len(ready) != len(mutators):
        missing = sorted(m.name for m in mutators if m not in ready)
        searched = mutators[0].bin_dir if mutators else _DEFAULT_BIN_DIR
        log.warning(
            "FormatFuzzer enabled but no binary found for %s. Looked in %s "
            "and on PATH for <format>-fuzzer. Build them with "
            "'./build.sh <format>' in a FormatFuzzer checkout and point "
            "--ff-bin-dir at it; these operators stay unavailable until then.",
            ", ".join(missing),
            searched,
        )
    return len(ready)


def _register_on_import() -> None:
    """Import-time registration (mirrors weizz_structural / fractal_voronoi).

    Operators stay dark until ``--formatfuzzer`` sets
    ``context.formatfuzzer_enabled``. Binary absence also keeps them
    unavailable, so a missing install is harmless.
    """
    # Honour an explicit env list so tests / CI can restrict templates.
    env = os.environ.get("FORMATFUZZER_TEMPLATES")
    templates = tuple(t.strip() for t in env.split(",") if t.strip()) if env else _DEFAULT_TEMPLATES
    # No bin_dir: import time must not pin a directory, or a later
    # --ff-bin-dir would have nothing to rebind against.
    register_formatfuzzer_mutators(templates)


_register_on_import()

__all__ = [
    "FormatFuzzerMutator",
    "canonical_template",
    "register_formatfuzzer_mutators",
    "report_availability",
]
