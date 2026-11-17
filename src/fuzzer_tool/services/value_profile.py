"""Value-profile channel: the shim's distance ids, switched from Python.

    open(mode)   OFF: nothing. ON/STALL: allocate ValueProfileShm, export
                 $__AFL_VP_SHM_ID; ON switches it on now.
    on_stall()   STALL: switch on at the first stall, then keep it on
                 (AFL++ ``-r -1``: once active, active for the run)

The ids themselves need no Python: they are synthetic edges, so novelty,
admission and rewards see them through ordinary coverage.
"""

from __future__ import annotations

import logging
import os
from enum import Enum

from fuzzer_tool.adapters.shm import ValueProfileShm

log = logging.getLogger(__name__)

VP_ENV = "__AFL_VP_SHM_ID"


class ValueProfile(Enum):
    """When distance ids are marked (--value-profile)."""

    OFF = "off"
    ON = "on"
    STALL = "stall"


class ValueProfileChannel:
    """Owns the switch segment for one campaign."""

    def __init__(self, shm: ValueProfileShm, mode: ValueProfile) -> None:
        self.shm = shm
        self._mode = mode

    @classmethod
    def open(cls, mode: ValueProfile) -> ValueProfileChannel | None:
        """Allocate and export the segment; None for OFF or on failure."""
        if mode is ValueProfile.OFF:
            return None
        try:
            shm = ValueProfileShm()
        except OSError as e:
            log.warning("value profile disabled: %s", e)
            return None

        os.environ[VP_ENV] = shm.env_id
        if mode is ValueProfile.ON:
            shm.enable()
        return cls(shm, mode)

    def on_stall(self) -> None:
        """Stall recovery engaged: STALL mode switches on for good."""
        if self._mode is not ValueProfile.STALL or self.shm.is_on():
            return
        self.shm.enable()
        print("[*] --value-profile stall: distance feedback on")

    def cleanup(self) -> None:
        self.shm.cleanup()
