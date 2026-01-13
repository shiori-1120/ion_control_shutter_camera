from __future__ import annotations

from typing import Any

from .fg_iface import FgDevice


class RigolFgDevice(FgDevice):
    """Adapter around the existing RigolDG handle."""

    def __init__(self, *, channel: int = 1, timeout_ms: int = 5000) -> None:
        self._channel = int(channel)
        self._timeout_ms = int(timeout_ms)
        self._handle: Any | None = None

    def open(self, resource: str) -> None:
        from src.lib.instruments.rigol_dg import RigolDG, RigolDgConfig

        rig = RigolDG(RigolDgConfig(visa_resource=resource, channel=self._channel, timeout_ms=self._timeout_ms))
        rig.open()
        self._handle = rig

    def apply(self, cfg: dict[str, float | str | bool]) -> None:
        if self._handle is None:
            raise RuntimeError("FG not opened")
        amp = cfg.get("amp_vpp")
        if amp is not None:
            self._handle.set_amplitude_vpp(float(amp))
        try:
            _ = self._handle.idn()
        except Exception:
            pass

    def close(self) -> None:
        if self._handle is None:
            return
        try:
            self._handle.output(False)
        except Exception:
            pass
        try:
            self._handle.close()
        except Exception:
            pass
        self._handle = None
