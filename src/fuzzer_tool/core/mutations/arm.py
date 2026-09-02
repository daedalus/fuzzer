"""Structure-aware ARM (A32/A64/Thumb-2) mutator.

Word-level mutation over fixed-width instruction words:
  a32/a64 — 4-byte words (mode determined by detection heuristics)
  t32_16  — 2-byte Thumb halfword
  t32_32  — 4-byte Thumb-2 doubleword
  raw     — trailing bytes that do not fit a word (len % 4 in A32 mode)

Thumb detection: a 16-bit halfword whose top 5 bits are 0b11101/0b11110/
0b11111 is the first half of a 32-bit Thumb-2 instruction; otherwise it
is a 16-bit Thumb instruction.
"""

from __future__ import annotations

from fuzzer_tool.core.mutations.generic import _swap_pair

import random
import struct
from dataclasses import dataclass

# Interesting word values for arithmetic mutation
WORD_VALUES = [0x00000000, 0x00000001, 0x0000FFFF, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF]

# A32 condition nibble values (top 4 bits)
COND_VALUES = [0x0, 0x1, 0xA, 0xB, 0xE, 0xF]  # eq/ne/ge/lt/al/nv

# Thumb conditional branches: 0xD0-0xDF are BEQ..BAL (16-bit)
T16_COND = list(range(0xD0, 0xE0))

# Interesting A32 branch offsets (imm24 field)
BRANCH_IMM24_VALUES = [0, 1, 0x7FFFFF, 0x800000, 0xFFFFFF]

# NOP word (A32) and BX LR word (A32) for generation
A32_NOP = 0xE1A00000
A32_BXLR = 0xE12FFF1E


@dataclass
class Word:
    """A single fixed-width instruction word."""

    raw: bytes
    kind: str  # a32 | a64 | t32_16 | t32_32 | raw
    value: int = 0  # little-endian word value (for a32/a64/t32_32)


def _is_thumb_stream(data: bytes) -> bool:
    """Heuristic: a stream whose first halfword looks like Thumb code.

    TODO follow-up: A64 vs A32 mode detection unsupported — this only
    distinguishes Thumb (T32) from 4-byte-word streams; A64 words are
    mutated as generic 32-bit words.
    """
    if len(data) < 2:
        return False
    hw = struct.unpack_from("<H", data, 0)[0]
    return (hw >> 11) in (0x1D, 0x1E, 0x1F) or (hw & 0xFF00) in (0xD000, 0xE000, 0xF000)


def parse_arm(data: bytes) -> list[Word] | None:
    """Parse a byte stream into fixed-width words.

    Returns None for an empty buffer. Thumb mode is inferred from the
    first halfword; a32/a64 words are 4 bytes, with the len % 4 tail
    stashed as a raw word.
    """
    if not data:
        return None

    words: list[Word] = []
    if _is_thumb_stream(data):
        pos = 0
        while pos + 2 <= len(data):
            hw = struct.unpack_from("<H", data, pos)[0]
            if (hw >> 11) in (0x1D, 0x1E, 0x1F):
                # 32-bit Thumb-2 instruction
                if pos + 4 <= len(data):
                    words.append(
                        Word(
                            raw=data[pos : pos + 4],
                            kind="t32_32",
                            value=struct.unpack_from("<I", data, pos)[0],
                        )
                    )
                    pos += 4
                    continue
                words.append(Word(raw=data[pos : pos + 2], kind="t32_16", value=hw))
                pos += 2
                continue
            words.append(Word(raw=data[pos : pos + 2], kind="t32_16", value=hw))
            pos += 2
        if pos < len(data):
            words.append(Word(raw=data[pos:], kind="raw"))
    else:
        pos = 0
        while pos + 4 <= len(data):
            words.append(
                Word(
                    raw=data[pos : pos + 4],
                    kind="a32",
                    value=struct.unpack_from("<I", data, pos)[0],
                )
            )
            pos += 4
        if pos < len(data):
            words.append(Word(raw=data[pos:], kind="raw"))

    return words


def serialize_arm(words: list[Word]) -> bytes:
    """Concatenate word bytes back into a stream."""
    return b"".join(w.raw for w in words)


def _kind_value(word: Word) -> tuple[str, int] | None:
    """Return (kind, value) for mutatable words."""
    if word.kind in ("a32", "a64", "t32_32"):
        return word.kind, word.value
    if word.kind == "t32_16":
        return word.kind, word.value
    return None


def _set_value(word: Word, value: int) -> None:
    """Patch a word's raw bytes and value from a new integer."""
    if word.kind in ("a32", "a64", "t32_32"):
        word.value = value & 0xFFFFFFFF
        word.raw = struct.pack("<I", word.value)
    elif word.kind == "t32_16":
        word.value = value & 0xFFFF
        word.raw = struct.pack("<H", word.value)


class ArmMutator:
    """Structure-aware ARM mutator."""

    _rng = random

    def mutate(self, data: bytes, max_len: int = 4096, rng=None) -> bytes:
        self._rng = rng or random
        words = parse_arm(data)
        if words is None:
            return self._generate_random_arm(max_len=max_len, rng=self._rng)

        op = self._rng.randint(0, 11)
        mutators = [
            self._word_bitflip,
            self._word_arith,
            self._word_interesting,
            self._cond_nibble_mutate,
            self._branch_imm24_mutate,
            self._reg_field_flip,
            self._word_swap,
            self._word_duplicate,
            self._word_delete,
            self._truncate_word,
            self._t32_pair_mutate,
            self._generate_random_arm,
        ]
        result = mutators[op](words, max_len)
        if isinstance(result, list):
            return serialize_arm(result)[:max_len]
        return result[:max_len]

    def _word_bitflip(self, words: list[Word], max_len: int) -> list[Word]:
        candidates = [w for w in words if _kind_value(w) is not None]
        if candidates:
            target = self._rng.choice(candidates)
            kind, value = _kind_value(target)
            bit = self._rng.randint(0, 31)
            _set_value(target, value ^ (1 << bit))
        return words

    def _word_arith(self, words: list[Word], max_len: int) -> list[Word]:
        candidates = [w for w in words if _kind_value(w) is not None]
        if candidates:
            target = self._rng.choice(candidates)
            kind, value = _kind_value(target)
            _set_value(target, (value * self._rng.choice([2, 3, 0x100, 0x10001])) & 0xFFFFFFFF)
        return words

    def _word_interesting(self, words: list[Word], max_len: int) -> list[Word]:
        candidates = [w for w in words if _kind_value(w) is not None]
        if candidates:
            target = self._rng.choice(candidates)
            kind, _value = _kind_value(target)
            _set_value(target, self._rng.choice(WORD_VALUES + [self._rng.randint(0, 0xFFFFFFFF)]))
        return words

    def _cond_nibble_mutate(self, words: list[Word], max_len: int) -> list[Word]:
        """Mutate the condition nibble of A32 instructions."""
        candidates = [w for w in words if w.kind == "a32"]
        if candidates:
            target = self._rng.choice(candidates)
            cond = (target.value >> 28) & 0xF
            new_cond = self._rng.choice([c for c in COND_VALUES if c != cond])
            _set_value(target, (target.value & 0x0FFFFFFF) | (new_cond << 28))
        return words

    def _branch_imm24_mutate(self, words: list[Word], max_len: int) -> list[Word]:
        """Mutate the imm24 field of A32 B/BL instructions."""
        branches = [w for w in words if w.kind == "a32" and ((w.value >> 26) & 3) == 2]
        if branches:
            target = self._rng.choice(branches)
            imm24 = self._rng.choice(BRANCH_IMM24_VALUES + [self._rng.randint(0, 0xFFFFFF)])
            _set_value(target, (target.value & 0xFC000000) | (imm24 & 0xFFFFFF))
        return words

    def _reg_field_flip(self, words: list[Word], max_len: int) -> list[Word]:
        """Flip a register field (bits 16-20 or bits 0-3) of an A32 word."""
        candidates = [w for w in words if w.kind == "a32"]
        if candidates:
            target = self._rng.choice(candidates)
            if self._rng.random() < 0.5:
                _set_value(target, target.value ^ (self._rng.randint(0, 7) << 16))
            else:
                _set_value(target, target.value ^ self._rng.randint(0, 7))
        return words

    def _word_swap(self, words: list[Word], max_len: int) -> list[Word]:
        swapable = [i for i, w in enumerate(words) if w.kind != "raw"]
        if (pair := _swap_pair(swapable, self._rng)) is not None:
            i, j = pair
            words[i], words[j] = words[j], words[i]
        return words

    def _word_duplicate(self, words: list[Word], max_len: int) -> list[Word]:
        dupable = [i for i, w in enumerate(words) if w.kind != "raw"]
        if dupable:
            idx = self._rng.choice(dupable)
            orig = words[idx]
            dup = Word(raw=orig.raw[:], kind=orig.kind, value=orig.value)
            words.insert(idx + 1, dup)
        return words

    def _word_delete(self, words: list[Word], max_len: int) -> list[Word]:
        delable = [i for i, w in enumerate(words) if w.kind != "raw"]
        if len(delable) > 1:
            words.pop(self._rng.choice(delable))
        return words

    def _truncate_word(self, words: list[Word], max_len: int) -> list[Word]:
        delable = [i for i, w in enumerate(words) if w.kind != "raw"]
        if delable:
            words.pop(self._rng.choice(delable))
        return words

    def _t32_pair_mutate(self, words: list[Word], max_len: int) -> list[Word]:
        """Mutate a 16-bit Thumb instruction's condition/opcode."""
        t16 = [w for w in words if w.kind == "t32_16"]
        if not t16:
            return words
        target = self._rng.choice(t16)
        if self._rng.random() < 0.5:
            # Conditional branch range: 0xD0-0xDF
            _set_value(target, (self._rng.choice(T16_COND) << 8) | (target.value & 0xFF))
        else:
            # imm8 field of B/BL: bits 0-7
            _set_value(target, (target.value & 0xFF00) | self._rng.randint(0, 0xFF))
        return words

    def _generate_random_arm(self, _words=None, max_len: int = 4096, rng=None) -> bytes:
        """Generate a random ARM stream of NOPs and BX LR returns."""
        # An int in the first slot is a max_len passed positionally. Without
        # this the cap lands in the vestigial placeholder and is dropped, and
        # the generator silently falls back to its own default -- the same
        # overload bmp/gzip/jpeg/zlib already handle and document.
        if isinstance(_words, int):
            max_len = _words
        self._rng = rng or self._rng
        out = bytearray()
        for _ in range(self._rng.randint(1, 16)):
            word = self._rng.choice([A32_NOP, A32_BXLR, self._rng.randint(0, 0xFFFFFFFF)])
            out.extend(struct.pack("<I", word))
        return bytes(out)[:max_len]
