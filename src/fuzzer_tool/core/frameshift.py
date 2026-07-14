"""FrameShift: automatic length field tracking for structure-aware fuzzing.

Ports AFL++'s FrameShift algorithm that automatically detects and adjusts
length/count fields in structured inputs. When mutations insert or delete
bytes, FrameShift updates nearby length fields to keep the input valid.

Core concept: a "relation" tracks a length field at (pos, size) that is
computed from an anchor point. When bytes are inserted/deleted before the
anchor, the relation's value adjusts automatically.

Relations are discovered by observing how insertions at various positions
affect the target's execution behavior.
"""

import logging
import random
import struct
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class Relation:
    """A tracked length/count field in the input.

    Attributes:
        pos: Byte position of the length field in the input.
        size: Size of the field in bytes (1, 2, or 4).
        anchor: Byte position that defines the start of the measured region.
            The length field = (insert_point - anchor) adjusted for val.
        insert_point: Byte position that defines the end of the measured region.
        val: Current computed value of the length field.
        le: Whether the field is little-endian.
        enabled: Whether this relation is still valid.
    """

    pos: int
    size: int
    anchor: int
    insert_point: int
    val: int
    le: bool = True
    enabled: bool = True
    # Saved state for restore
    _old_pos: int = 0
    _old_val: int = 0
    _old_anchor: int = 0
    _old_insert_point: int = 0


class FrameShift:
    """Track and adjust length fields during mutations.

    Maintains a list of discovered relations (length fields) and
    updates them when bytes are inserted or deleted.

    Args:
        max_relations: Maximum number of relations to track.
    """

    def __init__(self, max_relations: int = 64):
        self.relations: list[Relation] = []
        self.max_relations = max_relations
        # Blocked points: byte positions that are part of a relation field
        self.blocked_points: set[int] = set()

    def add_relation(self, rel: Relation) -> bool:
        """Add a new relation if under the limit.

        Returns:
            True if added, False if at capacity.
        """
        if len(self.relations) >= self.max_relations:
            return False
        self.relations.append(rel)
        for i in range(rel.pos, rel.pos + rel.size):
            self.blocked_points.add(i)
        return True

    def on_insert(self, idx: int, data_size: int, ignore_invalid: bool = True) -> bool:
        """Update all relations after an insertion at position idx.

        Args:
            idx: Byte position where data was inserted.
            data_size: Number of bytes inserted.
            ignore_invalid: If True, disable invalid relations instead of failing.

        Returns:
            True on success, False if an invalid insertion occurred.
        """
        for rel in self.relations:
            if not rel.enabled:
                continue
            if not self._rel_insert_invalid(rel, idx, data_size):
                continue
            if ignore_invalid:
                rel.enabled = False
            else:
                return False
        return True

    def on_delete(self, idx: int, data_size: int) -> None:
        """Update all relations after a deletion at position idx.

        Args:
            idx: Byte position where data was deleted.
            data_size: Number of bytes deleted.
        """
        for rel in self.relations:
            if not rel.enabled:
                continue
            self._rel_on_remove(rel, idx, data_size)

    def apply_to_buffer(self, buf: bytearray) -> None:
        """Write all relation values into the buffer.

        Updates the bytes at each relation's position with its current value.
        """
        for rel in self.relations:
            if not rel.enabled:
                continue
            val = rel.val
            if rel.le:
                for i in range(rel.size):
                    buf[rel.pos + i] = (val >> (i * 8)) & 0xFF
            else:
                for i in range(rel.size):
                    buf[rel.pos + rel.size - 1 - i] = (val >> (i * 8)) & 0xFF

    def save(self) -> None:
        """Save current state of all relations for later restore."""
        for rel in self.relations:
            rel._old_pos = rel.pos
            rel._old_val = rel.val
            rel._old_anchor = rel.anchor
            rel._old_insert_point = rel.insert_point

    def restore(self) -> None:
        """Restore all relations to their last saved state."""
        for rel in self.relations:
            rel.pos = rel._old_pos
            rel.val = rel._old_val
            rel.anchor = rel._old_anchor
            rel.insert_point = rel._old_insert_point
            rel.enabled = True

    def _rel_insert_invalid(self, rel: Relation, idx: int, size: int) -> bool:
        """Update a relation after an insertion. Returns True if invalid."""
        # Error if insert is inside the field
        if rel.pos < idx < rel.pos + rel.size:
            return True

        # Check if we should update the value
        if rel.anchor <= idx <= rel.insert_point:
            old_val = rel.val
            rel.val += size
            # Check overflow
            mask = (1 << (rel.size * 8)) - 1
            if rel.val > mask:
                return True

        # Move the field position
        if idx <= rel.pos:
            rel.pos += size

        # Move the anchor point (0 = locked)
        if rel.anchor > 0 and idx < rel.anchor:
            rel.anchor += size

        # Move the insert point
        if idx <= rel.insert_point:
            rel.insert_point += size

        return False

    def _rel_on_remove(self, rel: Relation, idx: int, size: int) -> bool:
        """Update a relation after a deletion. Returns True if invalid."""
        # Error if remove overlaps the field
        if idx < rel.pos + rel.size and idx + size > rel.pos:
            return True

        # Compute overlap between [idx, idx+size) and [anchor, insert_point)
        a = max(idx, rel.anchor)
        b = min(idx + size, rel.insert_point)
        overlap = max(0, b - a)

        if overlap > rel.val:
            return True
        rel.val -= overlap

        # Adjust positions
        if idx < rel.pos:
            rel.pos -= min(rel.pos - idx, size)
        if rel.anchor > 0 and idx < rel.anchor:
            rel.anchor -= min(rel.anchor - idx, size)
        if idx < rel.insert_point:
            rel.insert_point -= min(rel.insert_point - idx, size)

        return False

    def discover_relations(
        self,
        data: bytes,
        exec_fn,
        max_relations: int = 8,
        max_execs: int = 200,
    ) -> int:
        """Discover length fields by observing insertion effects.

        Inserts bytes at various positions and observes whether the
        execution path changes. Positions where insertion causes a path
        change that correlates with a nearby length field are candidates.

        This is a simplified version of AFL++'s frameshift analysis.

        Args:
            data: Input to analyze.
            exec_fn: Callable(bytes) -> int, returns execution checksum.
            max_relations: Maximum relations to discover.
            max_execs: Maximum executions.

        Returns:
            Number of relations discovered.
        """
        if not data:
            return 0

        length = len(data)
        exec_count = 0

        # Get baseline
        baseline = exec_fn(data)
        exec_count += 1

        # Test insertions at regular intervals
        step = max(1, length // 32)
        candidates: list[tuple[int, int]] = []  # (position, effect_strength)

        for pos in range(0, length, step):
            if exec_count >= max_execs or len(self.relations) >= max_relations:
                break

            # Insert 4 bytes at this position
            insert_data = bytes(random.randint(0, 255) for _ in range(4))
            modified = bytearray(data)
            modified[pos:pos] = insert_data

            cksum = exec_fn(bytes(modified))
            exec_count += 1

            if cksum != baseline:
                candidates.append((pos, 1))

        # Try to create relations from candidates
        for pos, strength in candidates:
            if len(self.relations) >= max_relations:
                break

            # Create a relation at this position
            # Heuristic: if position is near the start, it's likely a header length
            # If near the end, it's likely a payload length
            field_size = 4  # default to 4-byte length field
            if pos + field_size > length:
                field_size = min(2, length - pos)
            if field_size < 1:
                continue

            rel = Relation(
                pos=pos,
                size=field_size,
                anchor=0,  # start of input
                insert_point=pos,
                val=0,
                le=True,
            )
            self.add_relation(rel)

        log.debug(
            "FrameShift: discovered %d relations from %d execs",
            len(self.relations),
            exec_count,
        )
        return len(self.relations)
