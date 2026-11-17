"""Plateau groups: seeds whose mutants exit to each other's exact path.

soksolv fess_moves.go:341-415 (reversible-group exit dedup): states one
reversible move apart are one group, explored once. Here a boring mutant of
seed A that takes seed B's path (same rolling path hash, so same edges and
counts) puts A and B on one plateau:

    mutant(A) boring, path_hash == path_hash(B)  ->  union(A, B)

    {A, B, C} {D} {E, F}   group() = root key; the lottery's "group"
                           currency then backs each plateau once.

Union-find with path compression and union by size, bounded at max_keys.
A full table is rebuilt over the live keys (dead ones dropped, groups
kept); with no room after that a new union is skipped.
"""

from __future__ import annotations

from collections.abc import Mapping

# Keys held by default; ~100 B each.
MAX_KEYS = 65536
# Rebuild only when live keys fit in this share of the table, so a rebuild
# frees at least 1/8 of it and cannot run on every boring exec.
RETAIN_FRAC = 0.875


class PlateauGroups:
    """Bounded union-find over seed keys plus a path-hash -> seed index."""

    def __init__(self, max_keys: int = MAX_KEYS):
        self._max = max_keys
        self._parent: dict[str, str] = {}
        self._size: dict[str, int] = {}  # roots only
        # path hash -> seed key, rebuilt when the tracker's size moves or
        # an entry fails revalidation.
        self._by_hash: dict[int, str] = {}
        self._indexed = -1

    def __len__(self) -> int:
        return len(self._parent)

    def group(self, key: str) -> str:
        """Root of *key*'s plateau; an unseen key is its own."""
        parent = self._parent
        if key not in parent:
            return key

        root = key
        while parent[root] != root:
            root = parent[root]
        # Path compression: point the walked chain at the root.
        while parent[key] != root:
            parent[key], key = root, parent[key]
        return root

    def size(self, root: str) -> int:
        return self._size.get(root, 1)

    def observe(self, parent_key: str, path_hash: int, seed_hashes: Mapping[str, int]) -> bool:
        """A boring mutant of *parent_key* took *path_hash*; True on a union."""
        # Inlined miss path (the common case): one size check, one get.
        if len(seed_hashes) != self._indexed:
            self._reindex(seed_hashes)
        other = self._by_hash.get(path_hash)
        if other is None:
            return False
        if seed_hashes.get(other) != path_hash:
            self._reindex(seed_hashes)
            return False
        if other == parent_key:
            return False
        return self._union(parent_key, other, seed_hashes)

    def _reindex(self, seed_hashes: Mapping[str, int]) -> None:
        self._by_hash = {h: k for k, h in seed_hashes.items()}
        self._indexed = len(seed_hashes)

    def _union(self, a: str, b: str, live: Mapping[str, int]) -> bool:
        new = (a not in self._parent) + (b not in self._parent)
        if len(self._parent) + new > self._max:
            if len(live) > self._max * RETAIN_FRAC:
                return False
            self._retain(live)
            if len(self._parent) + new > self._max:
                return False

        ra, rb = self._root(a), self._root(b)
        if ra == rb:
            return False
        if self._size[ra] < self._size[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        self._size[ra] += self._size.pop(rb)
        return True

    def _root(self, key: str) -> str:
        # Root of *key*, inserting it as a singleton first.
        if key not in self._parent:
            self._parent[key] = key
            self._size[key] = 1
            return key
        return self.group(key)

    def _retain(self, live: Mapping[str, int]) -> None:
        """Rebuild over live keys: dead keys go, live members stay grouped."""
        members: dict[str, list[str]] = {}
        for key in self._parent:
            if key in live:
                members.setdefault(self.group(key), []).append(key)

        self._parent, self._size = {}, {}
        for keys in members.values():
            root = keys[0]
            for key in keys:
                self._parent[key] = root
            self._size[root] = len(keys)

    def to_dict(self) -> dict:
        return {"parent": dict(self._parent), "size": dict(self._size)}

    @classmethod
    def from_dict(cls, state: dict, max_keys: int = MAX_KEYS) -> PlateauGroups:
        """Restore :meth:`to_dict`; a malformed payload starts empty."""
        g = cls(max_keys)
        parent, size = state.get("parent"), state.get("size")
        if not isinstance(parent, dict) or not isinstance(size, dict):
            return g
        g._parent, g._size = dict(parent), dict(size)
        return g
