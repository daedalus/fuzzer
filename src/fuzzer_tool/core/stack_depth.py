"""Stack-depth novelty (honggfuzz instrumentCheckStackDepth).

    shim: base frame - lowest frame ──> SHM offset 0 ──> StackMax.observe()
                                                         deeper than max -> admit

An input that recurses deeper than any before it is kept, so the next one
can recurse deeper still (nested containers, expression parsers). Strict
``>``, not a growth factor: a path's frames are deterministic, and each
report is a new recursion level, bounded by how deep the target can go.
"""


class StackMax:
    """Campaign maximum stack depth in bytes; reports a deeper run."""

    def __init__(self) -> None:
        self.top = 0
        self.hits = 0

    def observe(self, depth: int) -> bool:
        """Whether *depth* beats every depth seen before.

        Example: max 4096 -> 4096 is not new, 4352 reports.
        The first nonzero depth is a baseline, not novelty; 0 (no
        per-edge callback in the build) is never a sample.
        """
        if depth <= self.top:
            return False

        baseline = self.top == 0
        self.top = depth
        if baseline:
            return False

        self.hits += 1
        return True
