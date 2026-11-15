"""IJON channel: the max-map segment plus its campaign tracker.

    open()    allocate IjonMaxShm, export $__AFL_IJON_SHM_ID
    observe() per exec: dirty? -> read + zero -> clean run? -> IjonMax

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


class Exec(Enum):
    """How the execution ended, as far as its slot values are concerned."""

    CLEAN = "clean"
    TRUNCATED = "truncated"  # crash or timeout: values are partial


class IjonChannel:
    """Reads the max-map after each exec and reports new maxima."""

    def __init__(self, shm: IjonMaxShm, tracker: IjonMax) -> None:
        self._shm = shm
        self.tracker = tracker
        self._buf = np.zeros(shm.slots, dtype=np.uint64)

    @classmethod
    def open(cls) -> "IjonChannel | None":
        """Allocate the segment and export it to targets spawned after this."""
        try:
            shm = IjonMaxShm(slots=IJON_MAP_SLOTS)
        except OSError as e:
            log.warning("IJON disabled: %s", e)
            return None

        os.environ[IJON_ENV] = shm.env_id
        return cls(shm, IjonMax(slots=IJON_MAP_SLOTS))

    def observe(self, run: Exec) -> bool:
        """Whether this exec raised any slot past the campaign max.

        Always zeroes a dirty segment, so a truncated run's partial values
        never carry into the next exec.
        """
        if not self._shm.dirty():
            return False

        self._shm.read_into(self._buf)
        if run is Exec.TRUNCATED:
            return False
        return self.tracker.observe(self._buf)

    def close(self) -> None:
        self._shm.cleanup()
