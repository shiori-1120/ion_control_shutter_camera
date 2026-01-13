from __future__ import annotations

from typing import Any

from ..clients.daq_client import DaqClient
from .daq_iface import DaqDevice, DaqSequenceCommand


class DaqClientDevice(DaqDevice):
    """Adapter for existing DaqClient queue protocol."""

    def __init__(self, client: DaqClient) -> None:
        self._client = client

    def open(self, device: str) -> None:
        if not self._client.connected:
            raise RuntimeError("DAQ client is not connected")

    def set_do(self, value: int) -> None:
        self._client.request({"cmd": "set_do", "value": int(value)}, timeout=2.0)

    def run_sequence_once(self, spec: DaqSequenceCommand) -> None:
        self._client.request(
            {
                "cmd": "run_sequence_once",
                "do_sequence": list(spec.do_sequence),
                "insert_index": int(spec.ao_insert_index),
                "ao_width_ms": float(spec.ao_width_ms),
                "ao_rate_hz": float(spec.ao_rate_hz),
                "ao_v_high": float(spec.ao_v_high),
                "ao_v_low": float(spec.ao_v_low),
            },
            timeout=5.0,
        )

    def all_off(self) -> None:
        self._client.request({"cmd": "all_off"}, timeout=2.0)

    def close(self) -> None:
        self._client.try_close()


class DaqQueueDevice(DaqDevice):
    """Adapter for raw DAQ worker queues."""

    def __init__(self, *, cmd_q: Any, resp_q: Any) -> None:
        self._cmd_q = cmd_q
        self._resp_q = resp_q

    def open(self, device: str) -> None:
        if self._cmd_q is None or self._resp_q is None:
            raise RuntimeError("DAQ queues are not ready")

    def _request(self, cmd: dict, *, timeout: float) -> dict:
        self._cmd_q.put(cmd)
        resp = self._resp_q.get(timeout=timeout)
        if not isinstance(resp, dict):
            raise RuntimeError(f"Invalid DAQ response: {resp!r}")
        if not resp.get("ok"):
            raise RuntimeError(resp.get("error", "DAQ error"))
        return resp

    def set_do(self, value: int) -> None:
        self._request({"cmd": "set_do", "value": int(value)}, timeout=2.0)

    def run_sequence_once(self, spec: DaqSequenceCommand) -> None:
        self._request(
            {
                "cmd": "run_sequence_once",
                "do_sequence": list(spec.do_sequence),
                "insert_index": int(spec.ao_insert_index),
                "ao_width_ms": float(spec.ao_width_ms),
                "ao_rate_hz": float(spec.ao_rate_hz),
                "ao_v_high": float(spec.ao_v_high),
                "ao_v_low": float(spec.ao_v_low),
            },
            timeout=5.0,
        )

    def all_off(self) -> None:
        self._request({"cmd": "all_off"}, timeout=2.0)

    def close(self) -> None:
        try:
            if self._cmd_q is not None:
                self._cmd_q.put({"cmd": "close"})
        except Exception:
            pass
