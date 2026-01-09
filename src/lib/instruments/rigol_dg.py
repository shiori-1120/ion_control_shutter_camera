"""Rigol DG-series (e.g., DG922 Pro) minimal SCPI wrapper via pyVISA.

This module intentionally implements only the small surface we need:
- identify (IDN)
- set frequency
- set amplitude (Vpp)
- output on/off

NOTE:
- SCPI for Rigol DG series is typically of the form:
    :SOUR1:FREQ <Hz>
    :OUTP1 ON|OFF
  If your unit differs, adjust command strings here.
"""

from __future__ import annotations

from dataclasses import dataclass

import pyvisa
try:
    from pyvisa import constants as _visa_constants
except Exception:  # pragma: no cover
    _visa_constants = None


@dataclass(frozen=True)
class RigolDgConfig:
    visa_resource: str
    channel: int = 1
    timeout_ms: int = 5000


class RigolDG:
    def __init__(self, cfg: RigolDgConfig) -> None:
        self.cfg = cfg
        self._rm: pyvisa.ResourceManager | None = None
        self._inst = None

    def open(self) -> None:
        if self._inst is not None:
            return
        self._rm = pyvisa.ResourceManager()
        self._inst = self._rm.open_resource(self.cfg.visa_resource, timeout=int(self.cfg.timeout_ms))
        # DG series generally works well with \n termination.
        try:
            self._inst.write_termination = "\n"
            self._inst.read_termination = "\n"
        except Exception:
            pass

    def close(self) -> None:
        try:
            if self._inst is not None:
                # Best-effort: return to LOCAL (front panel) control.
                try:
                    self.local()
                except Exception:
                    pass
                self._inst.close()
        finally:
            self._inst = None
            try:
                if self._rm is not None:
                    self._rm.close()
            finally:
                self._rm = None

    def local(self) -> None:
        """Return instrument to LOCAL (front panel) mode.

        Some instruments stay in remote mode even after the VISA session is
        closed unless we explicitly release it.
        """
        if self._inst is None:
            return
        # SCPI: many Rigol instruments support SYST:LOC.
        try:
            self._inst.write(":SYST:LOC")
            return
        except Exception:
            pass

        # VISA REN line operation (if supported by backend/instrument).
        try:
            if _visa_constants is not None:
                self._inst.control_ren(_visa_constants.RENLineOperation.go_to_local)
        except Exception:
            pass

    def query(self, cmd: str) -> str:
        if self._inst is None:
            raise RuntimeError("RigolDG is not open")
        return str(self._inst.query(cmd)).strip()

    def write(self, cmd: str) -> None:
        if self._inst is None:
            raise RuntimeError("RigolDG is not open")
        self._inst.write(cmd)

    def idn(self) -> str:
        return self.query("*IDN?")

    def set_frequency_hz(self, freq_hz: float) -> None:
        ch = int(self.cfg.channel)
        f = float(freq_hz)
        if f <= 0:
            raise ValueError("freq_hz must be > 0")
        self.write(f":SOUR{ch}:FREQ {f}")

    def set_amplitude_vpp(self, vpp: float) -> None:
        """Set output amplitude in Vpp.

        Note: Rigol DG series typically accepts Vpp via :SOUR<ch>:VOLT.
        """
        ch = int(self.cfg.channel)
        a = float(vpp)
        if a <= 0:
            raise ValueError("vpp must be > 0")
        self.write(f":SOUR{ch}:VOLT {a}")

    def output(self, on: bool) -> None:
        ch = int(self.cfg.channel)
        self.write(f":OUTP{ch} {'ON' if on else 'OFF'}")
