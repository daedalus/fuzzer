"""--tcp-mode direct: received TCP seeds go straight to the target.

The default (``round``) hands one queued seed per fuzz round to
``OperatorEngine.mutate()``, which returns it unmutated; the round then does
its scheduler / operator bookkeeping around it. ``direct`` skips the round:
before each loop iteration every queued seed is run as is, its edges are
recorded, and it joins the corpus only if it showed new coverage.

    inbox --pop--> run unmutated --> crash?   save_crash
                                     timeout? save_timeout
                                     new edges? save_to_corpus + record_edges
                                     else       boring (not kept)

    [*] Ran seed id_3f2a9c0d1e4b5a67 (direct): rc=0, +12 edges, admitted

--admit-all keeps every seed that ran cleanly (new coverage or not);
crashes and timeouts are still only saved as such.
"""

from __future__ import annotations

import enum
import sys
from typing import TYPE_CHECKING

from fuzzer_tool.services.fuzz_round import RoundOutcome

if TYPE_CHECKING:
    from fuzzer_tool.adapters.tcp_seed import TcpSeed, TcpSeedInbox
    from fuzzer_tool.services.fuzzer import Fuzzer

# _run_target's timeout sentinel.
TIMEOUT_RC = -1


class TcpMode(enum.Enum):
    """How --tcp-listen-port seeds reach the target (--tcp-mode)."""

    ROUND = "round"  # one per fuzz round, through mutate(), round bookkeeping
    DIRECT = "direct"  # all queued, before each round, no mutate()/round


class Admission(enum.Enum):
    """Which direct-mode seeds join the corpus (--admit-all)."""

    NEW_COVERAGE = "new-coverage"  # only seeds that showed new edges
    ALL = "all"  # every seed that ran cleanly; crashes/timeouts never


class TcpIngest:
    """Runs every queued TCP seed unmutated; keeps those per *admission*."""

    def __init__(
        self, fuzzer: Fuzzer, inbox: TcpSeedInbox, admission: Admission = Admission.NEW_COVERAGE
    ) -> None:
        self._f = fuzzer
        self._inbox = inbox
        self._admission = admission

    def drain(self) -> int:
        """Run every seed queued right now; returns how many ran."""
        ran = 0
        while (seed := self._inbox.pop_seed()) is not None:
            self._ingest(seed)
            ran += 1
        return ran

    def _ingest(self, seed: TcpSeed) -> None:
        """One unmutated execution, its verdict, and the 'Ran seed' line."""
        f = self._f
        rc, stderr = f._run_target(seed.data)
        f.exec_count += 1
        outcome, gained = self._judge(seed.data, rc, stderr)

        f._tcp_ran += 1
        # One write: the listener thread prints "Received" lines concurrently.
        sys.stdout.write(
            f"[*] Ran seed id_{seed.seed_id} (direct): rc={rc}, +{gained} edges, {outcome.value}\n"
        )
        sys.stdout.flush()

    def _judge(self, data: bytes, rc: int, stderr: str) -> tuple[RoundOutcome, int]:
        """(outcome, new edges): crashes and timeouts are saved as such, never admitted."""
        f = self._f
        if f._is_crash(rc, stderr):
            f.save_crash(data, rc, stderr)
            return RoundOutcome.CRASH, 0
        if rc == TIMEOUT_RC:
            f.timeout_count += 1
            f._corpus_manager.save_timeout(data)
            return RoundOutcome.TIMEOUT, 0

        has_new, edge_ids = f._unmutated_coverage(data)
        if not has_new and self._admission is Admission.NEW_COVERAGE:
            return RoundOutcome.BORING, 0
        # A root seed: no parent, like an initial corpus file. --admit-all
        # keeps no-gain seeds too, with whatever edges they reached.
        f.save_to_corpus(data)
        gained = f._record_direct_edges(data, edge_ids) if edge_ids else 0
        return RoundOutcome.ADMITTED, gained
