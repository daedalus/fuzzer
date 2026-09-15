"""Structure-aware / grammar mutations for structured inputs.

Provides a simple grammar specification format and mutation operators
that generate structurally valid inputs for protocols, file formats, etc.

Grammar format (S-expression style):
    <rule_name> = <alt1> | <alt2> | ...
    <rule_name> = <literal>          # fixed bytes
    <rule_name> = <rule_ref>{N}      # repeat N times
    <rule_name> = <rule_ref>+        # repeat 1-8 times
    <rule_name> = <rule_ref>*        # repeat 0-8 times
    <rule_name> = <rule_ref>{N,M}    # repeat N to M times

Example (HTTP request):
    request = method SP uri SP version CRLF headers CRLF body
    method  = GET | POST | PUT | DELETE | HEAD
    uri     = / | /api/v1 | /index.html
    version = HTTP/1.0 | HTTP/1.1 | HTTP/2
    CRLF    = \r\n
    SP      = \x20
    headers = header*
    header  = name ": " value CRLF
    name    = Host | Content-Type | Accept | Authorization
    value   = localhost | application/json | text/html | Bearer
    body    = {} | {"key":"value"} | <html></html>
"""

import hashlib
import logging
import random
import re
from pathlib import Path

from fuzzer_tool.core.rand_pool import RandPool

log = logging.getLogger(__name__)

_SIMPLE_ESCAPES = {"t": 9, "r": 13, "n": 10, "0": 0}


def decode_quoted_literal(text: str) -> bytes:
    """Decode escapes inside a quoted grammar literal.

    Recognises ``\\xNN``, ``\\\\``, ``\\"``, ``\\'`` and ``\\t`` / ``\\r`` /
    ``\\n`` / ``\\0``; anything else keeps its backslash, so a literal that
    was never meant as an escape survives unchanged.

    Quoted literals previously kept their backslash text VERBATIM, which
    made a binary literal inexpressible in quotes -- ``"\\xFF\\xD8"`` was
    eight ASCII characters, not the two-byte JPEG SOI marker. That is not a
    corner case: all 35 rules of the shipped ``dictionaries/jpeg.gram``
    are quoted marker definitions, so the whole grammar generated the
    literal text ``\\xFF\\xD8`` and could never produce a JPEG.

    Single pass rather than a regex sweep, for the same reason the
    dictionary parser needs one: the escapes are not independent, and a
    sweep for ``\\x[0-9a-f]{2}`` matches inside an escaped backslash, so
    ``\\\\x41`` would decode as backslash + ``A`` instead of backslash +
    ``x41``.
    """
    out = bytearray()
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "\\":
            j = text.find("\\", i)
            if j == -1:
                j = n
            out.extend(text[i:j].encode("utf-8"))
            i = j
            continue
        if i + 1 >= n:  # trailing lone backslash
            out.append(0x5C)
            break
        nxt = text[i + 1]
        if nxt == "x" and i + 3 < n:
            try:
                out.append(int(text[i + 2 : i + 4], 16))
                i += 4
                continue
            except ValueError:
                pass
        if nxt in _SIMPLE_ESCAPES:
            out.append(_SIMPLE_ESCAPES[nxt])
            i += 2
        elif nxt in ("\\", '"', "'"):
            out.append(ord(nxt))
            i += 2
        else:
            out.append(0x5C)
            i += 1
    return bytes(out)


# Ceiling on one generate() call when the caller names no max_len.  Per-token
# repeats are clamped to _MAX_REPEAT, but their PRODUCT was not: chained
# `{32}` rules multiply, and the default max_depth of 10 puts the ceiling at
# 32**10.  Truncation used to happen after the full expansion, so asking for
# 16 bytes could still cost gigabytes (finding #24).
GENERATION_BYTE_CAP = 1 << 20  # 1 MiB


class Grammar:
    """Simple grammar-based generator and mutator.

    Rules are stored as a dict of rule_name -> list of alternatives.
    Each alternative is a list of tokens (literals, refs, or quantifiers).
    """

    def __init__(self, seed=None):
        self.rules: dict[str, list[list]] = {}
        self.max_depth = 10
        rng = RandPool(seed=seed)
        self._rng = rng
        # Live byte budget for one generate() call; see _expand_tokens.
        self._budget = GENERATION_BYTE_CAP
        self._produced = 0

    def merge(self, other: "Grammar") -> "Grammar":
        """Merge rules from another Grammar into this one.

        If a rule name already exists, alternatives are appended
        (later values take precedence on potential overlaps).
        """
        for name, alts in other.rules.items():
            if name in self.rules:
                self.rules[name].extend(alts)
            else:
                self.rules[name] = alts.copy()
        return self

    def parse(self, spec: str):
        """Parse a grammar specification string.

        Args:
            spec: Grammar specification in the S-expression format.
        """
        for line in spec.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            name, body = line.split("=", 1)
            name = name.strip()
            body = body.strip()

            alternatives = []
            for alt in body.split("|"):
                alt = alt.strip()
                if not alt:
                    continue
                alternatives.append(self._parse_alternative(alt))

            if alternatives:
                self.rules[name] = alternatives

    def parse_file(self, path: str):
        """Parse a grammar file.

        Args:
            path: Path to grammar specification file.
        """
        text = Path(path).read_text(errors="replace")
        self.parse(text)

    def _parse_alternative(self, alt: str) -> list:
        """Parse a single alternative into a list of tokens.

        Both quoted and unquoted ``\\xNN`` / ``\\t`` / ``\\r`` / ``\\n``
        escapes expand to literal byte tokens, so ``SP = \\x20`` and
        ``SOI = "\\xFF\\xD8"`` both work. Quoted literals used to keep
        their backslash text verbatim; see ``decode_quoted_literal``.

        A BARE word is a rule reference, per normal grammar syntax -- it is
        not a literal. Literal ASCII inside a binary rule must therefore be
        quoted: ``signature = \\x89 "PNG" \\r\\n\\x1a\\n``, not
        ``\\x89PNG\\r\\n\\x1a\\n``, which asks for a nonterminal named PNG.
        An undefined reference expands to ``b"?"`` and now logs a warning.
        """
        # Match: "literal", rule_ref, rule_ref{N}, rule_ref{N,M}, rule_ref+, rule_ref*
        pattern = re.compile(
            r'"([^"]*)"'  # quoted literal
            r"|'([^']*)'"  # single-quoted literal
            r"|(\w+)\{(\d+),(\d+)\}"  # {N,M}
            r"|(\w+)\{(\d+)\}"  # {N}
            r"|(\w+)\+"  # +
            r"|(\w+)\*"  # *
            r"|(\w+)"  # bare rule_ref
        )
        _MAX_REPEAT = 32

        def match_segment(segment: str) -> list:
            tokens = []
            for m in pattern.finditer(segment):
                if m.group(1) is not None:
                    tokens.append(("lit", decode_quoted_literal(m.group(1))))
                elif m.group(2) is not None:
                    tokens.append(("lit", decode_quoted_literal(m.group(2))))
                elif m.group(3) is not None:
                    lo, hi = int(m.group(4)), int(m.group(5))
                    clamped_lo = min(lo, _MAX_REPEAT)
                    clamped_hi = min(hi, _MAX_REPEAT)
                    tokens.append(("repeat", m.group(3), clamped_lo, max(clamped_hi, clamped_lo)))
                elif m.group(6) is not None:
                    n = min(int(m.group(7)), _MAX_REPEAT)
                    tokens.append(("repeat", m.group(6), n, n))
                elif m.group(8) is not None:
                    tokens.append(("repeat", m.group(8), 1, 8))
                elif m.group(9) is not None:
                    tokens.append(("repeat", m.group(9), 0, 8))
                elif m.group(10) is not None:
                    tokens.append(("ref", m.group(10)))
            return tokens

        # Split the alternative at unquoted escapes; everything else goes
        # through the tokenizer unchanged.
        tokens: list = []
        simple_escapes = {"t": 9, "r": 13, "n": 10}
        seg_start = 0
        in_quote: str | None = None
        i = 0
        n = len(alt)
        while i < n:
            ch = alt[i]
            if in_quote is not None:
                if ch == in_quote:
                    in_quote = None
                i += 1
                continue
            if ch in "\"'":
                in_quote = ch
                i += 1
                continue
            if ch == "\\" and i + 1 < n:
                nxt = alt[i + 1]
                byte = None
                if nxt == "x" and i + 3 < n:
                    try:
                        v = int(alt[i + 2 : i + 4], 16)
                        if 0 <= v <= 255:
                            byte = v
                    except ValueError:
                        byte = None
                elif nxt in simple_escapes:
                    byte = simple_escapes[nxt]
                if byte is not None:
                    if seg_start < i:
                        tokens.extend(match_segment(alt[seg_start:i]))
                    tokens.append(("lit", bytes((byte,))))
                    i += 4 if nxt == "x" else 2
                    seg_start = i
                    continue
            i += 1
        if seg_start < n:
            tokens.extend(match_segment(alt[seg_start:]))
        return tokens

    def generate(
        self,
        rule: str = None,
        max_depth: int = None,
        max_len: int = 0,
        boltzmann: bool = False,
        target_size: float | None = None,
    ) -> bytes:
        """Generate a random input from the grammar.

        Args:
            rule: Starting rule name. If None, uses the first rule.
            max_depth: Maximum recursion depth.
            max_len: If > 0, truncate output to this length. Also caps the
                work done getting there — see below.
            boltzmann: If True, use :meth:`generate_boltzmann` instead of
                the plain recursive-descent expansion. See that method's
                docstring for what this buys you and why the default
                expansion below is biased.
            target_size: Only used when ``boltzmann=True``; forwarded to
                :meth:`generate_boltzmann`.

        Returns:
            Generated bytes.

        Expansion stops as soon as ``max_len`` bytes exist (or
        ``GENERATION_BYTE_CAP`` when no ``max_len`` is given), rather than
        expanding the whole tree and slicing afterwards.

        The returned bytes are unchanged by this. Expansion is a strictly
        left-to-right concatenation, so the first N bytes depend only on
        expansions that already happened; abandoning the rest cannot alter
        them, and everything abandoned was about to be sliced away. What
        changes is only the cost: a grammar of chained ``{32}`` rules asked
        for 16 bytes took 22s and 560MB at four levels of chaining and grew
        by 32x per level after that, on a grammar file the fuzzer accepts
        from ``--grammar``.

        Picking an alternative uniformly at random (``self._rng.choice(alts)``
        below) and a repeat count uniformly in ``[lo, hi]`` is *not* a
        uniform sample over derivation trees of a given size — see
        ``docs/handover/handover_trees.md`` §5 for the same bias measured on
        Dyck paths (a naive greedy walk over-samples deep, thin shapes) and
        §7.4 for why it generalizes to any grammar with a recursive rule
        (``expr = "(" expr ")" | "x"`` and similar): each recursive choice is
        made without regard to how many completions of each eventual size
        sit below it, so whichever branch happens to admit deeper expansion
        gets sampled disproportionately more often as depth increases.
        ``boltzmann=True`` corrects this; see :meth:`generate_boltzmann`.
        """
        if not self.rules:
            return b""

        if rule is None:
            rule = next(iter(self.rules))

        if max_depth is None:
            max_depth = self.max_depth

        if boltzmann:
            return self.generate_boltzmann(
                rule=rule, max_depth=max_depth, max_len=max_len, target_size=target_size
            )

        self._budget = max_len if max_len > 0 else GENERATION_BYTE_CAP
        self._produced = 0
        result = self._expand_rule(rule, max_depth)
        if max_len > 0:
            result = result[:max_len]
        return result

    def _expand_rule(self, name: str, depth: int) -> bytes:
        """Expand a rule into bytes."""
        if depth <= 0 or name not in self.rules:
            if depth <= 0:
                log.warning(
                    "Grammar recursion depth exhausted at rule '%s' — possible cyclic grammar", name
                )
            else:
                # Warning, not debug: the b"?" below is silently substituted
                # into generated output, so an undefined reference corrupts
                # every generation without any other symptom. png.gram
                # shipped with five (IHDR, IDAT, IEND, PLTE, PNG) and
                # rar.gram with two, unnoticed.
                log.warning(
                    "Grammar unknown rule '%s' — expanding to b'?' and corrupting "
                    "the generated output. Bare words are rule references; quote "
                    "literal text.",
                    name,
                )
            self._produced += 1
            return b"?"

        alts = self.rules[name]
        alt = self._rng.choice(alts)
        return self._expand_tokens(alt, depth)

    def _expand_tokens(self, tokens: list, depth: int) -> bytes:
        """Expand a list of tokens into bytes, stopping once the budget is spent.

        ``self._produced`` counts every byte this generate() call has emitted
        anywhere in the tree, so the check below bounds the whole expansion and
        not just this one rule. Charging happens at the leaves (literals and
        the ``b"?"`` substitute), which is where bytes are actually created, so
        nothing is counted twice on the way back up.
        """
        result = b""
        for token in tokens:
            if self._produced >= self._budget:
                break
            kind = token[0]
            if kind == "lit":
                result += token[1]
                self._produced += len(token[1])
            elif kind == "ref":
                result += self._expand_rule(token[1], depth - 1)
            elif kind == "repeat":
                _, name, lo, hi = token
                count = self._rng.randint(lo, hi)
                for _ in range(count):
                    if self._produced >= self._budget:
                        break
                    result += self._expand_rule(name, depth - 1)
        return result

    # ------------------------------------------------------------------
    # Boltzmann sampling (Duchon-Flajolet-Louchard-Schaeffer), §7.4 of
    # docs/handover/handover_trees.md
    # ------------------------------------------------------------------
    #
    # ``_expand_rule``/``_expand_tokens`` above choose an alternative
    # uniformly and a repeat count uniformly, at every branch, independent
    # of what sits below it. That is exactly the "naive greedy walk" bias
    # documented for Dyck paths in §5 of the same handover, generalized
    # from balanced-bracket strings to an arbitrary grammar: a recursive
    # rule's alternatives don't all admit the same number of completions of
    # a given size, so sampling them with equal probability over-represents
    # whichever branch happens to keep admitting deeper expansions.
    #
    # A Boltzmann sampler fixes this by choosing each alternative (and each
    # repeat count) with probability proportional to the *number of ways it
    # can be completed*, summed over every completion size and weighted by
    # x^size for a fixed tuning parameter x. Concretely, treat every rule as
    # an unlabelled combinatorial class and track its ordinary generating
    # function y_name(x) = sum_size (#derivations of that size) * x^size.
    # Because generation here is already bounded by ``max_depth`` (an
    # unknown/undefined reference or a rule reached at depth 0 always
    # becomes the single-byte ``b"?"`` atom — see ``_expand_rule``), y is a
    # genuine finite polynomial in x for every (rule, depth) pair rather
    # than a function needing a singularity analysis: it can be evaluated
    # exactly by recursing down through depth to 0 and summing back up.
    #
    # Given y(x) at every branch point, DFLS says: pick an alternative with
    # probability y_i(x) / sum_j y_j(x), and pick a bounded repeat count k
    # in [lo, hi] with probability y_atom(x)^k / sum_j y_atom(x)^j. Sampling
    # this way makes the probability of any *specific* derivation tree
    # exactly x^size(tree) / y(x) — which depends on the tree only through
    # its size, so conditioned on the realized size, every tree of that
    # size is equally likely. That's the property the plain generator
    # lacks and can't be patched to have without this reweighting.
    #
    # x is a free parameter that trades off expected output size: small x
    # favors small derivations, large x favors large ones. It's tuned by
    # bisection so the *expected* size of the top-level rule lands near
    # ``target_size`` (using the standard identity E[size](x) = x*y'(x)/y(x),
    # so y and its derivative are tracked together throughout).

    def generate_boltzmann(
        self,
        rule: str = None,
        max_depth: int = None,
        max_len: int = 0,
        target_size: float | None = None,
    ) -> bytes:
        """Generate from the grammar via Boltzmann sampling.

        Unlike :meth:`generate`, every derivation tree of a given size is
        equally likely to be produced (conditioned on the size that comes
        out) — see the block comment above this method for why the plain
        recursive-descent path doesn't have that property and how this one
        gets it. Best suited to grammars with recursive rules (parenthesis-
        like nesting, recursive `value` rules in JSON-like grammars, etc);
        for a grammar with no recursive rule this reduces to the same
        distribution ``generate()`` already produces.

        Args:
            rule: Starting rule name. If None, uses the first rule.
            max_depth: Maximum recursion depth (same meaning as in
                ``generate()`` — a rule reached at depth 0, or an unknown
                rule, always expands to the single-byte ``b"?"`` atom).
            max_len: If > 0, truncate output to this length and cap the
                byte budget while generating, exactly as in ``generate()``.
            target_size: Desired *expected* output size in bytes, used to
                tune the sampling parameter. Defaults to ``max_len`` if
                given, else 64. This is a target for the mean, not a
                guarantee — actual sizes vary around it, same as sampling
                any random-sized structure.

        Returns:
            Generated bytes.
        """
        if not self.rules:
            return b""
        if rule is None:
            rule = next(iter(self.rules))
        if max_depth is None:
            max_depth = self.max_depth
        if target_size is None:
            target_size = float(max_len) if max_len > 0 else 64.0
        target_size = max(float(target_size), 1e-9)

        x = self._tune_boltzmann_x(rule, max_depth, target_size)
        memo: dict[tuple[str, int], tuple[float, float]] = {}

        self._budget = max_len if max_len > 0 else GENERATION_BYTE_CAP
        self._produced = 0
        result = self._boltzmann_sample_rule(rule, max_depth, x, memo)
        if max_len > 0:
            result = result[:max_len]
        return result

    def _gf_and_deriv(
        self, name: str, depth: int, x: float, memo: dict[tuple[str, int], tuple[float, float]]
    ) -> tuple[float, float]:
        """Return ``(y(x), y'(x))`` for rule *name* at recursion *depth*.

        Mirrors ``_expand_rule`` exactly: depth <= 0 or an unknown rule is
        the fixed-size ``b"?"`` atom (y=x, y'=1), and otherwise y is the sum
        over alternatives of each alternative's own generating function.
        Memoized per (name, depth) since the same pair recurs constantly in
        any grammar with shared sub-rules or repeats.
        """
        key = (name, depth)
        cached = memo.get(key)
        if cached is not None:
            return cached
        if depth <= 0 or name not in self.rules:
            value = (x, 1.0)
        else:
            y_total = 0.0
            yp_total = 0.0
            for alt in self.rules[name]:
                ay, ayp = self._gf_token_seq(alt, depth, x, memo)
                y_total += ay
                yp_total += ayp
            value = (y_total, yp_total)
        memo[key] = value
        return value

    def _gf_token_seq(
        self, tokens: list, depth: int, x: float, memo: dict[tuple[str, int], tuple[float, float]]
    ) -> tuple[float, float]:
        """``(y(x), y'(x))`` for one alternative, i.e. a product of tokens.

        Combined via the product rule: for a product of factors
        ``y = y1*y2*...``, ``y' = y1'*y2*...  +  y1*y2'*...  + ...``, updated
        incrementally so no factor is ever revisited.
        """
        y = 1.0
        yp = 0.0
        for token in tokens:
            kind = token[0]
            if kind == "lit":
                length = len(token[1])
                ty = x**length
                typ = length * x ** (length - 1) if length > 0 else 0.0
            elif kind == "ref":
                ty, typ = self._gf_and_deriv(token[1], depth - 1, x, memo)
            elif kind == "repeat":
                _, name2, lo, hi = token
                ty, typ = self._gf_repeat(name2, lo, hi, depth, x, memo)
            else:  # pragma: no cover - defensive, all token kinds covered above
                ty, typ = 1.0, 0.0
            yp = yp * ty + y * typ
            y = y * ty
        return y, yp

    def _gf_repeat(
        self,
        name: str,
        lo: int,
        hi: int,
        depth: int,
        x: float,
        memo: dict[tuple[str, int], tuple[float, float]],
    ) -> tuple[float, float]:
        """``(y(x), y'(x))`` for a bounded repeat ``name{lo,hi}``.

        A repeat of exactly k copies has generating function ``ay(x)^k``
        where ``ay`` is the sub-rule's own GF; the repeat as a whole is the
        finite sum over k in [lo, hi]. Both the sum and its derivative are
        accumulated in one pass by carrying ``ay^k`` and its derivative
        forward from k to k+1 rather than recomputing the power each time.
        """
        ay, ayp = self._gf_and_deriv(name, depth - 1, x, memo)
        y_total = 0.0
        yp_total = 0.0
        ak = 1.0  # ay ** 0
        akp = 0.0  # d/dx of ay ** 0
        for k in range(hi + 1):
            if k >= lo:
                y_total += ak
                yp_total += akp
            akp = akp * ay + ak * ayp
            ak = ak * ay
        return y_total, yp_total

    def _tune_boltzmann_x(self, rule: str, depth: int, target_size: float) -> float:
        """Bisect for the x with expected size of *rule* ~= target_size.

        E[size](x) = x*y'(x)/y(x) is monotonically non-decreasing in x for
        any polynomial y with non-negative coefficients (it's a coefficient-
        weighted average of the exponents, and raising x shifts weight
        toward the higher-degree, i.e. larger-size, terms) — which is
        exactly what every y here is, so plain bisection suffices; no
        Newton step or singularity search needed.
        """
        memo: dict[tuple[str, int], tuple[float, float]] = {}

        def expected_size(x: float) -> float:
            memo.clear()
            y, yp = self._gf_and_deriv(rule, depth, x, memo)
            if y <= 0.0:
                return 0.0
            return x * yp / y

        lo, hi = 1e-6, 1.0
        expansions = 0
        while expected_size(hi) < target_size and expansions < 80:
            hi *= 2.0
            expansions += 1

        for _ in range(60):
            mid = (lo * hi) ** 0.5
            if expected_size(mid) < target_size:
                lo = mid
            else:
                hi = mid
        return (lo * hi) ** 0.5

    def _boltzmann_sample_rule(
        self, name: str, depth: int, x: float, memo: dict[tuple[str, int], tuple[float, float]]
    ) -> bytes:
        """Sample bytes for rule *name*, weighting each alternative by its GF."""
        if depth <= 0 or name not in self.rules:
            self._produced += 1
            return b"?"

        alts = self.rules[name]
        if len(alts) == 1:
            chosen = alts[0]
        else:
            weights = [self._gf_token_seq(alt, depth, x, memo)[0] for alt in alts]
            chosen = self._rng.weighted_choice(alts, weights)
        return self._boltzmann_expand_tokens(chosen, depth, x, memo)

    def _boltzmann_expand_tokens(
        self, tokens: list, depth: int, x: float, memo: dict[tuple[str, int], tuple[float, float]]
    ) -> bytes:
        """Like ``_expand_tokens``, but repeats are drawn from their true
        size-weighted distribution instead of uniformly over [lo, hi]."""
        result = b""
        for token in tokens:
            if self._produced >= self._budget:
                break
            kind = token[0]
            if kind == "lit":
                result += token[1]
                self._produced += len(token[1])
            elif kind == "ref":
                result += self._boltzmann_sample_rule(token[1], depth - 1, x, memo)
            elif kind == "repeat":
                _, name2, lo, hi = token
                if lo == hi:
                    count = lo
                else:
                    ay, _ = self._gf_and_deriv(name2, depth - 1, x, memo)
                    ks = list(range(lo, hi + 1))
                    weights = [ay**k for k in ks]
                    count = self._rng.weighted_choice(ks, weights)
                for _ in range(count):
                    if self._produced >= self._budget:
                        break
                    result += self._boltzmann_sample_rule(name2, depth - 1, x, memo)
        return result

    def mutate(self, data: bytes, max_len: int = 4096, rng=None) -> bytes:
        """Mutate an input using grammar-aware operations.

        Applies grammar-specific mutations that respect structure:
        1. Subtree replacement: regenerate a random subexpression
        2. Literal perturbation: flip/corrupt bytes in literals
        3. Repetition count change: add/remove repeated elements
        4. Field truncation: shorten a field
        5. Field extension: lengthen a field

        Args:
            data: Input bytes to mutate.
            max_len: Maximum output length.

        Returns:
            Mutated bytes.
        """
        self._rng = rng or self._rng
        if not self.rules or not data:
            return data

        op = self._rng.randint(0, 4)
        if op == 0:
            return self._mutate_literal(data, max_len)
        elif op == 1:
            return self._mutate_truncate(data)
        elif op == 2:
            return self._mutate_extend(data, max_len)
        elif op == 3:
            return self._mutate_insert(data, max_len)
        else:
            return self._mutate_replace_section(data, max_len)

    def _mutate_literal(self, data: bytes, max_len: int) -> bytes:
        """Flip or replace bytes in the input."""
        if not data:
            return self.generate(max_len=max_len)

        buf = bytearray(data)
        idx = self._rng.randint(0, len(buf) - 1)
        op = self._rng.randint(0, 3)
        if op == 0:
            buf[idx] ^= 1 << self._rng.randint(0, 7)
        elif op == 1:
            buf[idx] = self._rng.randint(0, 255)
        elif op == 2:
            buf[idx] = self._rng.choice([0, 1, 0x7F, 0x80, 0xFF])
        else:
            # Replace with a printable ASCII character
            buf[idx] = self._rng.randint(0x20, 0x7E)
        return bytes(buf[:max_len])

    def _mutate_truncate(self, data: bytes) -> bytes:
        """Truncate the input at a random point."""
        if len(data) <= 1:
            return data
        cut = self._rng.randint(1, len(data) - 1)
        return data[:cut]

    def _mutate_extend(self, data: bytes, max_len: int) -> bytes:
        """Extend the input with generated bytes."""
        # max_len=64 rather than an unbounded generate() sliced to 64: the
        # slice discards the rest, so asking for it only paid for it.
        extra = self.generate(max_len=64)
        if len(data) + len(extra) > max_len:
            extra = extra[: max_len - len(data)]
        pos = self._rng.randint(0, len(data))
        return data[:pos] + extra + data[pos:]

    def _mutate_insert(self, data: bytes, max_len: int) -> bytes:
        """Insert a generated fragment at a random position."""
        fragment = self.generate(max_len=32)
        if len(data) + len(fragment) > max_len:
            fragment = fragment[: max_len - len(data)]
        pos = self._rng.randint(0, len(data))
        return data[:pos] + fragment + data[pos:]

    def _mutate_replace_section(self, data: bytes, max_len: int) -> bytes:
        """Replace a random section with generated bytes."""
        if len(data) <= 1:
            return self.generate(max_len=max_len)
        start = self._rng.randint(0, len(data) - 1)
        end = self._rng.randint(start + 1, min(start + 32, len(data)))
        replacement = self.generate(max_len=end - start)
        return data[:start] + replacement + data[end:]


# Built-in grammar specs for common formats
GRAMMARS = {
    "json": """
# Simple JSON grammar
json    = value
value   = object | array | string | number | "true" | "false" | "null"
object  = "{" "}"
array   = "[" "]"
string  = "\\"" text "\\""
text    = word* | ""
word    = letter | digit | space
letter  = "a" | "b" | "c" | "d" | "e" | "f" | "g" | "h" | "i" | "j" | "k" | "l" | "m" | "n" | "o" | "p" | "q" | "r" | "s" | "t" | "u" | "v" | "w" | "x" | "y" | "z"
digit   = "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
space   = \\x20 | \\t
""",
    "http_request": """
# HTTP request grammar
request = method SP uri SP version CRLF headers CRLF body
method  = GET | POST | PUT | DELETE | HEAD | PATCH
uri     = / | /api | /api/v1 | /index.html | /health | /debug
version = HTTP/1.0 | HTTP/1.1
CRLF    = \\r\\n
SP      = \\x20
headers = header*
header  = name ":" SP value CRLF
name    = Host | Content-Type | Accept | Authorization | X-Request-ID | User-Agent
value   = localhost | application/json | text/html | application/octet-stream | Bearer | close
body    = {} | {"key":"value"} | data=12345
""",
    "elf": """
# Minimal ELF header
magic   = \\x7f ELF
class   = \\x01 | \\x02
data    = \\x01
version = \\x01
osabi   = \\x00 | \\x03 | \\x06 | \\x09
padding = \\x00{8}
""",
}


def load_grammar(spec: str | Path) -> Grammar:
    """Load a grammar from a spec string or file path.

    If the spec is a known name (e.g., 'json', 'http_request', 'elf'),
    loads the built-in grammar. If it's a file path, reads the file.

    Args:
        spec: Grammar name or file path.

    Returns:
        Parsed Grammar object.
    """
    g = Grammar()
    if isinstance(spec, str) and spec in GRAMMARS:
        g.parse(GRAMMARS[spec])
    elif isinstance(spec, Path) or (isinstance(spec, str) and Path(spec).is_file()):
        g.parse_file(str(spec))
    else:
        # Try as inline spec
        g.parse(str(spec))
    return g


# ---------------------------------------------------------------------------
# Tree-level AST and mutations (Superion/Nautilus-style)
# ---------------------------------------------------------------------------


class TreeNode:
    """Node in a parse tree for grammar-aware mutation.

    A leaf node holds raw bytes. An interior node holds a rule name and
    a list of child TreeNodes. The tree mirrors the grammar's structure:
    each interior node corresponds to a nonterminal expansion.
    """

    __slots__ = ("rule", "children", "data")

    def __init__(self, rule: str = "", children: list | None = None, data: bytes = b""):
        self.rule = rule
        self.children = children or []
        self.data = data  # only for leaf nodes

    @property
    def is_leaf(self) -> bool:
        return not self.children and bool(self.data)

    def serialize(self) -> bytes:
        """Serialize the tree back to bytes."""
        if self.is_leaf:
            return self.data
        parts = []
        for child in self.children:
            parts.append(child.serialize())
        return b"".join(parts)

    def size(self) -> int:
        """Number of nodes in this subtree."""
        if self.is_leaf:
            return 1
        return 1 + sum(c.size() for c in self.children)

    def depth(self) -> int:
        if self.is_leaf:
            return 0
        return 1 + max(c.depth() for c in self.children) if self.children else 0

    def collect_interior(self, rule: str | None = None) -> list["TreeNode"]:
        """Collect all interior nodes (optionally filtered by rule name)."""
        result = []
        if not self.is_leaf:
            if rule is None or self.rule == rule:
                result.append(self)
            for child in self.children:
                result.extend(child.collect_interior(rule))
        return result

    def canonical_hash(self) -> bytes:
        """Canonical structural hash of this single subtree.

        Aho-Hopcroft-Ullman-style: two subtrees hash identically iff they
        have the same shape -- same rule label at every position, same
        number/order of children, and (at leaves) the same raw bytes.
        Unlike classic AHU tree-isomorphism hashing, child order is
        preserved rather than sorted, since these are ordered (plane)
        parse trees, not unordered trees -- a grammar production's child
        order is semantically meaningful here, not an artifact to
        normalize away.

        For hashing every node of a tree (not just one), use
        ``collect_interior_hashes()`` instead: calling this method once
        per node, from the outside, recomputes every descendant's hash
        once per ancestor -- O(n) per call, O(n^2) across a whole tree.
        This method itself is the O(n) single-subtree building block that
        ``collect_interior_hashes()`` amortizes across all nodes in one
        pass.
        """
        if self.is_leaf:
            return hashlib.sha1(
                b"L|" + self.rule.encode("utf-8", "surrogateescape") + b"|" + self.data
            ).digest()
        child_hashes = b"".join(c.canonical_hash() for c in self.children)
        return hashlib.sha1(
            b"N|" + self.rule.encode("utf-8", "surrogateescape") + b"|" + child_hashes
        ).digest()

    def collect_interior_hashes(
        self, rule: str | None = None
    ) -> list[tuple["TreeNode", bytes]]:
        """Collect ``(node, canonical_hash)`` pairs for interior nodes.

        Hashes every node in this subtree bottom-up in a single O(n) pass
        (mirrors the fix for the same per-node-recompute trap that
        ``_collect_nodes_with_sizes`` addresses for size-weighted sampling,
        see docs/handover/handover_trees.md §3.2/§7.3), then returns the
        canonical hash for each interior node alongside the node itself --
        the same filter ``collect_interior()`` applies, plus its hash as a
        byproduct of the traversal rather than a second O(n) walk.
        """
        result: list[tuple[TreeNode, bytes]] = []

        def visit(node: "TreeNode") -> bytes:
            if node.is_leaf:
                return hashlib.sha1(
                    b"L|" + node.rule.encode("utf-8", "surrogateescape") + b"|" + node.data
                ).digest()
            child_hashes = b"".join(visit(c) for c in node.children)
            h = hashlib.sha1(
                b"N|" + node.rule.encode("utf-8", "surrogateescape") + b"|" + child_hashes
            ).digest()
            if rule is None or node.rule == rule:
                result.append((node, h))
            return h

        visit(self)
        return result

    def collect_leaves(self) -> list["TreeNode"]:
        if self.is_leaf:
            return [self]
        result = []
        for child in self.children:
            result.extend(child.collect_leaves())
        return result

    def all_nodes(self) -> list["TreeNode"]:
        result = [self]
        for child in self.children:
            result.extend(child.all_nodes())
        return result

    def _find_path(self, target: "TreeNode", path: list | None = None) -> list | None:
        """Find the path from root to target node."""
        if path is None:
            path = []
        if self is target:
            return path
        for i, child in enumerate(self.children):
            found = child._find_path(target, path + [i])
            if found is not None:
                return found
        return None

    def __repr__(self):
        if self.is_leaf:
            preview = self.data[:20]
            return f"Leaf({self.rule!r}, {preview!r}...)"
        return f"Node({self.rule!r}, children={len(self.children)})"


def _weighted_choice(nodes: list["TreeNode"], rng) -> "TreeNode":
    """Pick a node with probability proportional to its subtree size.

    Plain uniform selection over-samples the many small subtrees near a
    Catalan-distributed tree's leaves and under-samples the few large
    subtrees near the root — the same bias Koza's genetic-programming
    literature addresses with a 90/10 internal/leaf crossover-point split.
    Weighting by ``TreeNode.size()`` corrects for it directly.
    """
    weights = [n.size() for n in nodes]
    total = sum(weights)
    if total <= 0:
        return rng.choice(nodes)
    r = rng.randint(0, total - 1)
    acc = 0
    for node, w in zip(nodes, weights):
        acc += w
        if r < acc:
            return node
    return nodes[-1]


class SubtreePopulation:
    """Global pool of subtrees harvested across many corpus entries.

    Port of the "subtree-population crossover" idea (GRIIN, ASE '23;
    Grammarinator x AFL++, 2026): grammar-aware crossover is far more
    productive when the replacement subtree can come from *any* corpus
    entry that shares the target's rule, not only a freshly generated
    subtree or a clone from within the same tree. This class keeps a
    bounded, per-rule reservoir of interior nodes so ``TreeMutator``
    can splice in subtrees seen elsewhere in the corpus.

    Reservoir sampling (Algorithm R) bounds memory to ``max_per_rule``
    nodes per rule regardless of corpus size, while still giving every
    harvested node an equal chance of ending up in the pool.

    Plain reservoir sampling treats every harvested node as distinct, so
    exact structural duplicates (the same small JSON object shape
    recurring across many corpus entries, say) compete for reservoir
    slots on equal footing with genuinely novel shapes -- duplicates can
    and do crowd out donors that would add real shape diversity. Each
    rule's pool tracks a canonical-hash (AHU-style, see
    ``TreeNode.canonical_hash``) count of the shapes it currently holds;
    a newly harvested node whose shape is already represented in that
    rule's pool is skipped rather than spending a slot on it, turning the
    population into a shape-coverage set rather than a pure size-biased
    random sample (docs/handover/handover_trees.md §7.3). Distinct shapes
    still compete for the remaining slots via ordinary reservoir sampling.
    """

    def __init__(self, max_per_rule: int = 64):
        self.max_per_rule = max_per_rule
        self._pools: dict[str, list[TreeNode]] = {}
        # Parallel to _pools[rule]: canonical hash of the node at each slot.
        self._pool_hashes: dict[str, list[bytes]] = {}
        # Per-rule count of how many pool slots currently hold each shape
        # hash -- lets us tell in O(1) whether a shape is already represented.
        self._shape_counts: dict[str, dict[bytes, int]] = {}
        self._seen: dict[str, int] = {}

    def add(self, tree: TreeNode, rng=None) -> None:
        """Harvest every interior node of *tree* into the population."""
        rand = rng or random
        for node, node_hash in tree.collect_interior_hashes():
            rule = node.rule
            pool = self._pools.setdefault(rule, [])
            pool_hashes = self._pool_hashes.setdefault(rule, [])
            shape_counts = self._shape_counts.setdefault(rule, {})
            seen = self._seen.get(rule, 0)
            self._seen[rule] = seen + 1

            if shape_counts.get(node_hash, 0) > 0:
                # This exact shape is already sitting in the pool for this
                # rule -- skip it rather than spend a reservoir slot on a
                # structural duplicate.
                continue

            if len(pool) < self.max_per_rule:
                pool.append(node)
                pool_hashes.append(node_hash)
                shape_counts[node_hash] = shape_counts.get(node_hash, 0) + 1
                continue
            j = rand.randint(0, seen)
            if j < self.max_per_rule:
                old_hash = pool_hashes[j]
                shape_counts[old_hash] -= 1
                if shape_counts[old_hash] <= 0:
                    del shape_counts[old_hash]
                pool[j] = node
                pool_hashes[j] = node_hash
                shape_counts[node_hash] = shape_counts.get(node_hash, 0) + 1

    def sample(self, rule: str, rng=None) -> "TreeNode | None":
        """Return a random subtree previously harvested for *rule*, or None."""
        pool = self._pools.get(rule)
        if not pool:
            return None
        rand = rng or random
        return pool[rand.randint(0, len(pool) - 1)]

    def __len__(self) -> int:
        return sum(len(pool) for pool in self._pools.values())


class TreeMutator:
    """Parse inputs against a grammar into trees and mutate at the tree level.

    Produces mutations that respect the grammar's structure — swapping two
    subtrees of the same nonterminal type, duplicating a node, deleting a
    subtree, or splicing in a subtree from a different corpus entry.
    """

    def __init__(self, grammar: Grammar, seed=None):
        self.grammar = grammar
        rng = RandPool(seed=seed)
        self._rng = rng
        # Known structural delimiters for heuristic parsing
        self._delimiters: dict[str, tuple[bytes, bytes]] = {
            "json": (b"{", b"}"),
            "array": (b"[", b"]"),
        }

    def parse(
        self, data: bytes, rule: str | None = None, chunk_size: int | None = None
    ) -> TreeNode:
        """Heuristic parse of input bytes into a tree structure.

        Uses grammar rule knowledge to identify structural boundaries.
        For well-known formats (JSON, XML), uses delimiter matching.
        For unknown formats, segments by fixed-size chunks from grammar
        quantifiers (or ``chunk_size`` when given, e.g. an inferred record
        stride from ``estimate_record_size``).
        """
        if rule is None:
            rule = next(iter(self.grammar.rules)) if self.grammar.rules else "root"

        if not data:
            return TreeNode(rule=rule, data=b"")

        # Try structured parsing for known formats
        tree = self._parse_structured(data, rule)
        if tree is not None:
            return tree

        # Fallback: segment by grammar-inferred chunk sizes
        return self._parse_chunked(data, rule, chunk_size)

    def _parse_structured(self, data: bytes, rule: str) -> TreeNode | None:
        """Parse structured formats using delimiter matching."""
        # JSON-like: { ... } or [ ... ]
        if data.startswith(b"{") and data.endswith(b"}"):
            return self._parse_braced(data, rule, b"{", b"}")
        if data.startswith(b"[") and data.endswith(b"]"):
            return self._parse_braced(data, rule, b"[", b"]")
        return None

    def _parse_braced(self, data: bytes, rule: str, open_b: bytes, close_b: bytes) -> TreeNode:
        """Parse braced content into a tree with children for each element."""
        children = []
        # Opening delimiter
        children.append(TreeNode(rule="delim", data=open_b))

        # Parse contents: split by top-level commas (for JSON objects/arrays)
        inner = data[len(open_b) : -len(close_b)] if len(data) > len(open_b) + len(close_b) else b""
        if inner:
            elements = self._split_top_level(inner)
            for elem in elements:
                elem = elem.strip()
                if not elem:
                    continue
                child_rule = self._infer_rule(elem)
                children.append(self.parse(elem, child_rule))

        # Closing delimiter
        children.append(TreeNode(rule="delim", data=close_b))
        return TreeNode(rule=rule, children=children)

    def _split_top_level(self, data: bytes) -> list[bytes]:
        """Split by comma at the top nesting level only."""
        result = []
        depth = 0
        current = bytearray()
        in_string = False
        escape_next = False
        for b in data:
            if escape_next:
                current.append(b)
                escape_next = False
                continue
            if b == ord("\\") and in_string:
                current.append(b)
                escape_next = True
                continue
            if b == ord('"'):
                in_string = not in_string
            if not in_string:
                if b in (ord("{"), ord("[")):
                    depth += 1
                elif b in (ord("}"), ord("]")):
                    depth -= 1
                elif b == ord(",") and depth == 0:
                    result.append(bytes(current))
                    current = bytearray()
                    continue
            current.append(b)
        if current:
            result.append(bytes(current))
        return result

    def _infer_rule(self, data: bytes) -> str:
        """Infer which grammar rule a byte fragment likely matches."""
        data = data.strip()
        if data.startswith(b'"') and data.endswith(b'"'):
            return "string"
        if data.startswith(b"{"):
            return "object"
        if data.startswith(b"["):
            return "array"
        if data in (b"true", b"false", b"null"):
            return "value"
        try:
            float(data)
            return "number"
        except (ValueError, UnicodeDecodeError):
            pass
        return "value"

    def _parse_chunked(self, data: bytes, rule: str, chunk_size: int | None = None) -> TreeNode:
        """Fallback: segment input into fixed-size chunks based on grammar."""
        # Infer chunk size from grammar quantifiers, unless a record stride
        # (from periodicity detection) was supplied.
        chunk_size = chunk_size or self._infer_chunk_size()
        if chunk_size <= 0 or len(data) <= chunk_size:
            return TreeNode(rule=rule, data=data)

        children = []
        for i in range(0, len(data), chunk_size):
            chunk = data[i : i + chunk_size]
            child_rule = self._infer_rule(chunk)
            children.append(TreeNode(rule=child_rule, data=chunk))
        return TreeNode(rule=rule, children=children)

    def _infer_chunk_size(self) -> int:
        """Infer a reasonable chunk size from grammar quantifiers."""
        for alts in self.grammar.rules.values():
            for alt in alts:
                for token in alt:
                    if token[0] == "repeat":
                        # Use the repeat body's typical expansion as chunk size
                        return 16  # reasonable default for structured fields
        return 16

    # ------------------------------------------------------------------
    # Tree-level mutations
    # ------------------------------------------------------------------

    def mutate_tree(
        self,
        tree: TreeNode,
        max_len: int = 4096,
        rng=None,
        population: SubtreePopulation | None = None,
    ) -> bytes:
        """Apply a random tree-level mutation and serialize back to bytes.

        Operations:
        1. Subtree swap: replace a node with a freshly generated subtree
           of the same rule type
        2. Subtree delete: remove a node (replace with empty)
        3. Subtree duplicate: clone a node and insert the copy nearby
        4. Subtree splice: replace a node with a subtree from another
           corpus entry's tree (see ``SubtreePopulation``); falls back to
           subtree swap when no *population* is supplied or no donor of a
           matching rule has been harvested yet
        5. Rule substitution: replace a node with a different alternative
           from the same grammar rule

        Args:
            population: Optional cross-corpus subtree pool for op 4.
                Callers should keep one long-lived ``SubtreePopulation``
                per fuzzer run and feed it every parsed corpus tree.
        """
        self._rng = rng or self._rng
        if tree.is_leaf:
            return self._mutate_leaf(tree, max_len)

        op = self._rng.randint(0, 5)
        if op == 0:
            return self._tree_swap(tree, max_len)
        elif op == 1:
            return self._tree_delete(tree, max_len)
        elif op == 2:
            return self._tree_duplicate(tree, max_len)
        elif op == 3:
            return self._tree_splice(tree, max_len, population)
        elif op == 4:
            return self._tree_rule_sub(tree, max_len)
        else:
            return self._mutate_leaf(tree, max_len)

    def _tree_swap(self, tree: TreeNode, max_len: int) -> bytes:
        """Replace a random interior node with a freshly generated subtree."""
        targets = tree.collect_interior()
        if not targets:
            return tree.serialize()[:max_len]
        target = _weighted_choice(targets, self._rng)
        # Generate a replacement of the same rule type
        replacement_bytes = self.grammar.generate(target.rule, max_len=max_len)
        replacement = TreeNode(rule=target.rule, data=replacement_bytes)
        # Replace in parent
        self._replace_in_tree(tree, target, replacement)
        return tree.serialize()[:max_len]

    def _tree_delete(self, tree: TreeNode, max_len: int) -> bytes:
        """Remove a random non-root interior node."""
        # Collect interior nodes that aren't the root
        all_interior = tree.collect_interior()
        candidates = [n for n in all_interior if n is not tree and n.children]
        if not candidates:
            return tree.serialize()[:max_len]
        target = _weighted_choice(candidates, self._rng)
        # Replace with empty leaf
        self._replace_in_tree(tree, target, TreeNode(rule=target.rule, data=b""))
        return tree.serialize()[:max_len]

    def _tree_duplicate(self, tree: TreeNode, max_len: int) -> bytes:
        """Clone a random node and insert the copy as a sibling."""
        # Find a parent with multiple children
        all_interior = tree.collect_interior()
        parents_with_children = [n for n in all_interior if len(n.children) >= 2]
        if not parents_with_children:
            return tree.serialize()[:max_len]
        parent = _weighted_choice(parents_with_children, self._rng)
        idx = self._rng.randint(0, len(parent.children) - 1)
        clone = self._clone_tree(parent.children[idx])
        # Insert after the original
        parent.children.insert(idx + 1, clone)
        return tree.serialize()[:max_len]

    def _tree_splice(
        self, tree: TreeNode, max_len: int, population: SubtreePopulation | None
    ) -> bytes:
        """Replace a random interior node with a same-rule subtree donated
        by a different corpus entry (subtree-population crossover).

        Falls back to ``_tree_swap`` (freshly-generated subtree) when no
        population was supplied or it hasn't harvested a matching rule yet
        — that keeps this op always productive instead of a silent no-op.
        """
        if population is None or not len(population):
            return self._tree_swap(tree, max_len)
        targets = tree.collect_interior()
        if not targets:
            return tree.serialize()[:max_len]
        rng = self._rng
        # Try a bounded number of random targets rather than shuffling the
        # whole list — RandPool doesn't implement shuffle, and one match is
        # all a single mutation needs.
        tries = min(len(targets), 8)
        for _ in range(tries):
            target = _weighted_choice(targets, rng)
            donor = population.sample(target.rule, rng=rng)
            if donor is None or donor is target:
                continue
            if target is tree:
                # Root itself is the splice point: it has no parent to
                # rewrite in place, so the donor subtree simply becomes the
                # whole output.
                return self._clone_tree(donor).serialize()[:max_len]
            self._replace_in_tree(tree, target, self._clone_tree(donor))
            return tree.serialize()[:max_len]
        return self._tree_swap(tree, max_len)

    def _tree_rule_sub(self, tree: TreeNode, max_len: int) -> bytes:
        """Replace a node with a different alternative from the same rule."""
        targets = tree.collect_interior()
        # Filter to rules with multiple alternatives
        multi_targets = [
            n
            for n in targets
            if n.rule in self.grammar.rules and len(self.grammar.rules[n.rule]) > 1
        ]
        if not multi_targets:
            return tree.serialize()[:max_len]
        target = _weighted_choice(multi_targets, self._rng)
        replacement_bytes = self.grammar.generate(target.rule, max_len=max_len)
        self._replace_in_tree(tree, target, TreeNode(rule=target.rule, data=replacement_bytes))
        return tree.serialize()[:max_len]

    def _mutate_leaf(self, tree: TreeNode, max_len: int) -> bytes:
        """Byte-level mutation on a leaf node."""
        data = tree.serialize()
        if not data:
            return self.grammar.generate(max_len=max_len)
        buf = bytearray(data)
        idx = self._rng.randint(0, len(buf) - 1)
        buf[idx] ^= 1 << self._rng.randint(0, 7)
        return bytes(buf[:max_len])

    def _replace_in_tree(self, root: TreeNode, old: TreeNode, new: TreeNode):
        """Replace old node with new in root's tree."""
        for i, child in enumerate(root.children):
            if child is old:
                root.children[i] = new
                return True
            if self._replace_in_tree(child, old, new):
                return True
        return False

    def _clone_tree(self, node: TreeNode) -> TreeNode:
        """Deep-clone a tree node."""
        if node.is_leaf:
            return TreeNode(rule=node.rule, data=node.data)
        return TreeNode(
            rule=node.rule,
            children=[self._clone_tree(c) for c in node.children],
        )

    def _path_copy_replace(
        self, root: TreeNode, path: list[int], replacement: TreeNode
    ) -> TreeNode:
        """Return a copy of *root* with the node at *path* swapped for *replacement*.

        Trees guarantee a unique root-to-node path (see docs/handover/handover_trees.md
        §1), so a full deep clone isn't needed to relocate one node: only the
        nodes *on* the path from root to the target need copying (their
        `children` list is rebuilt so the original tree is left untouched).
        Every sibling subtree not on the path is reused by reference. This is
        the standard "path copying" technique for persistent tree updates
        (Okasaki, *Purely Functional Data Structures*) and turns a per-candidate
        cost of O(n) (full clone) into O(depth).
        """
        if not path:
            return replacement
        idx, rest = path[0], path[1:]
        new_children = list(root.children)
        new_children[idx] = self._path_copy_replace(root.children[idx], rest, replacement)
        return TreeNode(rule=root.rule, children=new_children)

    # ------------------------------------------------------------------
    # Hierarchical delta debugging for tmin
    # ------------------------------------------------------------------

    def hierarchical_shrink(self, data: bytes, still_crashes, max_rounds: int = 64) -> bytes:
        """Shrink a crashing input by removing whole tree-level chunks first.

        Tries removing each nonterminal subtree before falling back to
        byte-level reduction. Converges to a minimal reproducer that's
        structurally meaningful.

        Args:
            data: The crashing input.
            still_crashes: Callable(bytes) -> bool — returns True if input still crashes.
            max_rounds: Maximum reduction rounds.

        Returns:
            Minimized bytes.
        """
        best = data
        for _ in range(max_rounds):
            tree = self.parse(best)
            # Collect all non-root interior nodes with children
            candidates = [n for n in tree.collect_interior() if n is not tree and n.children]
            if not candidates:
                break

            improved = False
            for node in candidates:
                # Try removing this subtree. Only the nodes on the root->node
                # path need copying (path-copying, not a full O(n) clone) —
                # see _path_copy_replace.
                path = tree._find_path(node)
                if path is None:
                    continue
                replacement = TreeNode(rule=node.rule, data=b"")
                candidate_tree = self._path_copy_replace(tree, path, replacement)
                candidate = candidate_tree.serialize()
                if candidate and candidate != best and still_crashes(candidate):
                    best = candidate
                    improved = True
                    break  # restart with smaller tree

            if not improved:
                break

        return best


# ---------------------------------------------------------------------------
# PNG format-aware mutations
# ---------------------------------------------------------------------------


# Re-export PNG classes/functions from dedicated module for backwards compatibility


# Re-export PngChunkMutator from dedicated module for backwards compatibility
