"""IJON channel: the max-map segment plus its campaign tracker.

    open()    allocate IjonMaxShm, export $__AFL_IJON_SHM_ID
    observe() per exec: dirty? -> read + zero -> clean run? -> IjonMax
    pick()    per seed pick: the paper's IJON_SCHEDULE_MAXMAP draw (§IV-A4)

The set/inc annotations need nothing here: they are synthetic edges and
reach the corpus through ordinary coverage.
"""

import logging
import os
from enum import Enum

import numpy as np

from fuzzer_tool.adapters.shm import IjonMaxShm
from fuzzer_tool.core.ijon import IJON_MAP_SLOTS, IjonMax

log = logging.getLogger(__name__)

IJON_ENV = "__AFL_IJON_SHM_ID"

# IJON_SCHEDULE_MAXMAP: percent of picks left to the normal queue. The
# paper names the knob but gives no default; half keeps both queues fed.
DEFAULT_SCHEDULE_MAXMAP = 50
SCHEDULE_DRAW_MAX = 100


class Exec(Enum):
    """How the execution ended, as far as its slot values are concerned."""

    CLEAN = "clean"
    TRUNCATED = "truncated"  # crash or timeout: values are partial


class IjonChannel:
    """Reads the max-map after each exec and reports new maxima."""

    def __init__(
        self,
        shm: IjonMaxShm,
        tracker: IjonMax,
        schedule_maxmap: int = DEFAULT_SCHEDULE_MAXMAP,
    ) -> None:
        self._shm = shm
        self.tracker = tracker
        self.schedule_maxmap = schedule_maxmap
        self._buf = np.zeros(shm.slots, dtype=np.uint64)

    @classmethod
    def open(cls, schedule_maxmap: int = DEFAULT_SCHEDULE_MAXMAP) -> "IjonChannel | None":
        """Allocate the segment and export it to targets spawned after this."""
        try:
            shm = IjonMaxShm(slots=IJON_MAP_SLOTS)
        except OSError as e:
            log.warning("IJON disabled: %s", e)
            return None

        os.environ[IJON_ENV] = shm.env_id
        return cls(shm, IjonMax(slots=IJON_MAP_SLOTS), schedule_maxmap)

    def observe(self, run: Exec, data: bytes | None = None) -> bool:
        """Whether this exec raised any slot past the campaign max.

        *data* (the executed input) becomes the input of each raised slot;
        None leaves the max queue as it is.

        Always zeroes a dirty segment, so a truncated run's partial values
        never carry into the next exec.
        """
        if not self._shm.dirty():
            return False

        self._shm.read_into(self._buf)
        if run is Exec.TRUNCATED:
            return False
        return self.tracker.observe(self._buf, data)

    def pick(self, rng) -> bytes | None:
        """A max-queue input, or None to let the normal queue pick.

        Paper §IV-A4: below IJON_SCHEDULE_MAXMAP the usual scheduling runs,
        otherwise a random slot with an input is fuzzed. Drawn from 0..99,
        not the paper's 1..100, so 0 and 100 mean "always" and "never".
        Example: schedule 50, draw 50 -> slot input; draw 49 -> None.
        """
        if rng.randint(0, SCHEDULE_DRAW_MAX - 1) < self.schedule_maxmap:
            return None
        return self.tracker.pick_input(rng.choice)

    def close(self) -> None:
        self._shm.cleanup()
