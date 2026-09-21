"""Single registry of every shipped from-scratch seed generator, by format name.

The fuzzer already ships 35 ``_generate_random_<fmt>`` builders (one per
structure-aware mutator in :mod:`fuzzer_tool.core.mutations`) plus the six
hand-verified constant builders in :mod:`fuzzer_tool.core.minimal_seeds`.
Only eleven of those 41 are reachable from a live campaign's cold-start path
today: the six in ``MINIMAL_SEEDS``, and five wired into
``SeedPicker._GENERATED_SEED_FORMATS`` (webp/webm/zip/protobuf/riff) --
because that table is deliberately scoped to formats
``target_profiler.py::_infer_format`` can actually name (see that table's
own docstring). The other ~24 generators are real and tested but sit
unused: nothing in the live fuzz loop can name "adts" or "flac" as a
``format_signature`` today.

This module doesn't touch that live path (extending format *detection* is a
separate, riskier change). It just gives every generator a name so a caller
that already knows what format it wants -- ``fuzzer-tool genseed``, a test,
a notebook -- can ask for one directly, instead of the ~24 extra builders
staying reachable only via a private import path buried in a mutation
module.

:data:`FORMATS` is the full sorted list of names :func:`generate` accepts.
"""

from __future__ import annotations

import importlib

from fuzzer_tool.core.minimal_seeds import MINIMAL_SEEDS

# fmt name -> (mutations submodule, class name, generator method name).
# Every entry here is a real, tested `_generate_random_<fmt>` method already
# shipped in core/mutations/ -- this table only names them, it writes none
# of the generation logic itself. Resolved lazily (imported inside
# `generate()`) so asking for one format never imports the other 34.
GENERATED_SEED_FORMATS: dict[str, tuple[str, str, str]] = {
    # Already reachable from SeedPicker._format_aware_seed (kept here too so
    # this module is a complete, single source of truth for callers that
    # don't want to know about that separate, narrower table).
    "webp": ("webp", "WebpMutator", "_generate_random_webp"),
    "webm": ("webm", "WebmMutator", "_generate_random_webm"),
    "zip": ("zip", "ZipMutator", "_generate_random_zip"),
    "protobuf": ("protobuf", "ProtobufMutator", "_generate_random_protobuf"),
    "riff": ("riff", "RiffMutator", "_generate_random_riff"),
    # Shipped generators with no live cold-start path yet -- see module
    # docstring. Names match the `format_signature`/mutation-module
    # vocabulary already used elsewhere in the repo (e.g. "av1" not
    # "av1_rtp", to match the format rather than the module filename).
    "adts": ("adts", "AdtsMutator", "_generate_random_adts"),
    "arm": ("arm", "ArmMutator", "_generate_random_arm"),
    "asf": ("asf", "AsfMutator", "_generate_random_asf"),
    "av1": ("av1_rtp", "Av1RtpMutator", "_generate_random_av1"),
    "avif": ("avif", "AvifMutator", "_generate_random_avif"),
    "cfhd": ("cfhd", "CfhdMutator", "_generate_random_cfhd"),
    "der": ("der", "DerMutator", "_generate_random_der"),
    "dvbsub": ("dvbsub", "DvbsubMutator", "_generate_random_dvbsub"),
    "ffconcat": ("ffconcat", "FfconcatMutator", "_generate_random_ffconcat"),
    "flac": ("flac", "FlacMutator", "_generate_random_flac"),
    "flv": ("flv", "FlvMutator", "_generate_random_flv"),
    "isobmff": ("isobmff", "IsobmffMutator", "_generate_random_isobmff"),
    "jpeg2000": ("jpeg2000", "Jpeg2000Mutator", "_generate_random_jpeg2000"),
    "magicyuv": ("magicyuv", "MagicYUVMutator", "_generate_random_magicyuv"),
    "mp3": ("mp3", "Mp3Mutator", "_generate_random_mp3"),
    "mpegts": ("mpegts", "MpegtsMutator", "_generate_random_ts"),
    "nal": ("nal", "NalMutator", "_generate_random_nal_stream"),
    "ogg": ("ogg", "OggMutator", "_generate_random_ogg"),
    "pgs": ("pgs", "PgsMutator", "_generate_random_pgs"),
    "rasc": ("rasc", "RascMutator", "_generate_random_rasc"),
    "shorten": ("shorten", "ShnMutator", "_generate_random_shn"),
    "sqlite": ("sqlite", "SqliteMutator", "_generate_random_sqlite"),
    "tiff": ("tiff", "TiffMutator", "_generate_random_tiff"),
    "x86": ("x86", "X86Mutator", "_generate_random_x86"),
}

#: Every format name :func:`generate` accepts, sorted. Union of the six
#: constant builders in ``MINIMAL_SEEDS`` and the generator table above --
#: on overlap (there is none today; MINIMAL_SEEDS and this table cover
#: disjoint formats) MINIMAL_SEEDS would win, per :func:`generate`.
FORMATS: tuple[str, ...] = tuple(sorted(set(MINIMAL_SEEDS) | set(GENERATED_SEED_FORMATS)))


def generate(fmt: str, max_len: int = 4096, rng=None) -> bytes | None:
    """Build one from-scratch seed for *fmt*, or ``None`` if *fmt* is unknown.

    ``MINIMAL_SEEDS`` wins on overlap: those six builders are hand-verified
    against each format's own real-world decoder (Pillow/zlib/gzip -- see
    :mod:`fuzzer_tool.core.minimal_seeds`'s docstring), while a mutation
    module's own ``_generate_random_<fmt>`` was written to seed in-loop
    mutation, not validated as a decoder-accepted cold-start seed in
    isolation. They also ignore *rng* (they're constant), since that
    verification was against one specific fixed byte string, not a family.

    *rng*, when given, should be a :class:`fuzzer_tool.core.rand_pool.RandPool`
    (the mutators' own dependency, per Hard Rule 16 -- never the stdlib
    ``random`` module) -- passed straight through, so ``None`` lets a
    generator fall back to its own freshly-seeded pool.
    """
    builder = MINIMAL_SEEDS.get(fmt)
    if builder is not None:
        seed = builder()
        return seed[:max_len] if max_len > 0 else seed

    spec = GENERATED_SEED_FORMATS.get(fmt)
    if spec is None:
        return None
    mod, cls, meth = spec
    mutator_cls = getattr(importlib.import_module(f"fuzzer_tool.core.mutations.{mod}"), cls)
    mutator = mutator_cls()
    return getattr(mutator, meth)(max_len=max_len, rng=rng)
