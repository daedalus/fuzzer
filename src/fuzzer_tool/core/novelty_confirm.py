"""Decide which of an execution's "new" edges reproduce (handover F2).

An execution can report edge ids that no later execution of the same input
reproduces. On the default path (one persistent table, ASLR off) about 6.5%
of executions carry some, they make up 12-18% of the "new coverage"
successes, and they are most of the singleton edges. Every scheduler that
reads rarity or credits an operator for a discovery reads them.

The evidence that an id is real is that it fires again. :func:`confirm` takes
the original execution's edge set and one rerun and keeps the intersection.
It never adopts an id only the rerun saw: F2 also loses real ids on the first
run, but those are not novelty this execution demonstrated.

Pure: no I/O, no shared memory. The caller runs the rerun.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass


@dataclass(frozen=True)
class Confirmation:
    edge_ids: frozenset[int]  # original & rerun
    phantoms: frozenset[int]  # original - rerun
    has_new: bool  # novelty that survives confirmation


def confirm(
    has_new: bool,
    new_ids: frozenset[int],
    old_bucket_novel: bool,
    edge_ids: Collection[int],
    rerun_ids: Collection[int],
) -> Confirmation:
    """Confirm one execution's novelty against one rerun.

    ``new_ids`` are the ids the scan saw for the first time;
    ``old_bucket_novel`` says a previously seen edge reached a new hit-count
    bucket. Those are the only two things that make ``has_new`` true, so it
    survives when either a new id reproduced or the bucket event happened. A
    bucket event on an already-seen edge does not depend on a new id, which is
    why a phantom beside it must not withdraw it. With no new id the verdict is
    left as reported.
    """
    original = frozenset(edge_ids)
    reproduced = original & frozenset(rerun_ids)
    # No new id at all means the cause is not an id, so there is nothing to withdraw.
    still_new = has_new and (bool(new_ids & reproduced) or old_bucket_novel or not new_ids)
    return Confirmation(reproduced, original - reproduced, still_new)
