"""Headless DAQ worker process.

Goal:
- Keep NI-DAQ control in a lightweight, headless process.
- GUI (or other runners) talk to this worker over a local connection.

Protocol:
- Uses multiprocessing.connection Listener/Client on localhost.
- Messages are Python dicts (pickled).

Run:
  python -m src.shutter_camera_trigger.daq_worker

Notes:
- This worker intentionally does NOT import tkinter/matplotlib.
- On any error/exit, it attempts to set DO to ALL_OFF.
"""

from __future__ import annotations

import argparse
import threading
import time
from multiprocessing.connection import Listener

import numpy as np
import nidaqmx
from nidaqmx.constants import AcquisitionType, LineGrouping, RegenerationMode
from nidaqmx.stream_writers import DigitalSingleChannelWriter


# -------------------------
# Defaults / protocol
# -------------------------
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 58888
AUTHKEY = b"ion_control_shutter_camera"


# -------------------------
# DO bit mapping (port1/line0:3)
# -------------------------
ALL_OFF = 0b0000


# -------------------------
# AO timing
# -------------------------
AO_RATE_HZ = 5000.0
AO_EDGE_LOW_SAMPLES = 1


class DaqSession:
    def __init__(self, *, device: str) -> None:
        self.device = device
        self.port_range = f"{device}/port1/line0:3"
        self.ao_ch = f"{device}/ao0"

        self._do_task = nidaqmx.Task()
        self._ao_task: nidaqmx.Task | None = None
        self._write_port = None

        self._ao_rate_hz: float | None = None
        self._ao_width_ms: float | None = None
        self._ao_v_high: float | None = None
        self._ao_v_low: float | None = None

        self.ao_high_s: float | None = None
        self.ao_total_s: float | None = None
        self.ao_actual_width_ms: float | None = None

        self._setup_do()
        self.set_do(ALL_OFF)

    def _setup_do(self) -> None:
        self._do_task.do_channels.add_do_chan(
            self.port_range,
            line_grouping=LineGrouping.CHAN_FOR_ALL_LINES,
        )
        do_writer = DigitalSingleChannelWriter(self._do_task.out_stream)
        self._write_port = do_writer.write_one_sample_port_uint16

    def ensure_ao_config(
        self,
        *,
        width_ms: float,
        rate_hz: float,
        v_high: float,
        v_low: float,
    ) -> None:
        width_ms = float(width_ms)
        rate_hz = float(rate_hz)
        v_high = float(v_high)
        v_low = float(v_low)

        if width_ms <= 0:
            raise ValueError("AO width_ms must be > 0")
        if rate_hz <= 0:
            raise ValueError("AO rate_hz must be > 0")

        if (
            self._ao_task is not None
            and self._ao_width_ms == width_ms
            and self._ao_rate_hz == rate_hz
            and self._ao_v_high == v_high
            and self._ao_v_low == v_low
        ):
            return

        if self._ao_task is not None:
            try:
                self._ao_task.close()
            except Exception:
                pass
            self._ao_task = None

        n_high = max(1, int(round((width_ms / 1000.0) * rate_hz)))
        self.ao_high_s = n_high / rate_hz
        self.ao_total_s = (n_high + 2 * AO_EDGE_LOW_SAMPLES) / rate_hz
        self.ao_actual_width_ms = (n_high / rate_hz) * 1000.0

        ao_wave = np.concatenate(
            [
                np.full(AO_EDGE_LOW_SAMPLES, v_low, dtype=np.float64),
                np.full(n_high, v_high, dtype=np.float64),
                np.full(AO_EDGE_LOW_SAMPLES, v_low, dtype=np.float64),
            ]
        )

        ao_task = nidaqmx.Task()
        ao_task.ao_channels.add_ao_voltage_chan(self.ao_ch, min_val=0.0, max_val=5.0)
        ao_task.timing.cfg_samp_clk_timing(
            rate_hz,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=len(ao_wave),
        )
        ao_task.out_stream.regen_mode = RegenerationMode.ALLOW_REGENERATION
        ao_task.write(ao_wave, auto_start=False)

        self._ao_task = ao_task
        self._ao_width_ms = width_ms
        self._ao_rate_hz = rate_hz
        self._ao_v_high = v_high
        self._ao_v_low = v_low

    def set_do(self, value: int) -> None:
        if self._write_port is None:
            raise RuntimeError("DO writer not initialized")
        self._write_port(int(value))

    def pulse_ao_once(self) -> None:
        if self._ao_task is None:
            raise RuntimeError("AO is not configured")
        self._ao_task.start()
        self._ao_task.wait_until_done(timeout=5.0)
        self._ao_task.stop()

    def close(self) -> None:
        try:
            if self._write_port is not None:
                self._write_port(ALL_OFF)
        except Exception:
            pass

        if self._ao_task is not None:
            try:
                self._ao_task.close()
            except Exception:
                pass
            self._ao_task = None

        try:
            self._do_task.close()
        except Exception:
            pass


class WorkerState:
    def __init__(self) -> None:
        self.session: DaqSession | None = None
        self.seq_thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.running = False
        self.last_step_id = 0

    def status(self) -> dict:
        s = self.session
        return {
            "connected": s is not None,
            "device": getattr(s, "device", None),
            "running": self.running,
            "last_step_id": self.last_step_id,
            "ao_high_s": getattr(s, "ao_high_s", None),
            "ao_total_s": getattr(s, "ao_total_s", None),
            "ao_actual_width_ms": getattr(s, "ao_actual_width_ms", None),
        }


def wait_s(duration_s: float) -> None:
    """Software-timed wait with reduced drift."""
    end_t = time.perf_counter() + max(0.0, float(duration_s))
    while True:
        remaining = end_t - time.perf_counter()
        if remaining <= 0:
            return
        if remaining > 0.005:
            time.sleep(remaining - 0.002)
        else:
            time.sleep(0)


def _start_sequence_thread(
    st: WorkerState,
    *,
    do_sequence: list[tuple[int, float]],
    insert_index: int,
    ao_width_ms: float,
) -> None:
    if st.session is None:
        raise RuntimeError("Not connected")

    st.session.ensure_ao_config(width_ms=float(ao_width_ms), rate_hz=AO_RATE_HZ, v_high=5.0, v_low=0.0)

    st.stop_event.clear()
    st.running = True

    def run() -> None:
        try:
            while not st.stop_event.is_set():
                for idx, (do_value, hold_s) in enumerate(do_sequence):
                    st.session.set_do(int(do_value))
                    st.last_step_id += 1
                    wait_s(float(hold_s))
                    if insert_index >= 0 and idx == insert_index:
                        st.session.pulse_ao_once()
                    if st.stop_event.is_set():
                        break
        finally:
            try:
                if st.session is not None:
                    st.session.set_do(ALL_OFF)
            except Exception:
                pass
            st.running = False

    st.seq_thread = threading.Thread(target=run, daemon=True)
    st.seq_thread.start()


def handle_command(st: WorkerState, msg: dict) -> dict:
    cmd = msg.get("cmd")

    if cmd == "ping":
        return {"ok": True}

    if cmd == "connect":
        device = str(msg.get("device", "")).strip()
        if not device:
            return {"ok": False, "error": "device is required"}
        if st.session is not None:
            return {"ok": True, **st.status()}
        st.session = DaqSession(device=device)
        return {"ok": True, **st.status()}

    if cmd == "disconnect":
        st.stop_event.set()
        if st.session is not None:
            st.session.close()
            st.session = None
        st.running = False
        return {"ok": True, **st.status()}

    if cmd == "set_do":
        if st.session is None:
            return {"ok": False, "error": "Not connected"}
        value = int(msg.get("value"))
        st.session.set_do(value)
        return {"ok": True, **st.status()}

    if cmd == "start_sequence":
        if st.session is None:
            return {"ok": False, "error": "Not connected"}
        if st.running:
            return {"ok": False, "error": "Sequence already running"}
        do_sequence = msg.get("sequence")
        insert_index = int(msg.get("insert_index", -1))
        ao_width_ms = float(msg.get("ao_width_ms", 1.0))
        if not isinstance(do_sequence, list) or not do_sequence:
            return {"ok": False, "error": "sequence must be a non-empty list"}
        # validate tuples
        seq2: list[tuple[int, float]] = []
        for item in do_sequence:
            if not (isinstance(item, (list, tuple)) and len(item) == 2):
                return {"ok": False, "error": "sequence items must be (value, hold_s)"}
            seq2.append((int(item[0]), float(item[1])))
        _start_sequence_thread(st, do_sequence=seq2, insert_index=insert_index, ao_width_ms=ao_width_ms)
        return {"ok": True, **st.status()}

    if cmd == "stop_sequence":
        st.stop_event.set()
        st.running = False
        try:
            if st.session is not None:
                st.session.set_do(ALL_OFF)
        except Exception:
            pass
        return {"ok": True, **st.status()}

    if cmd == "status":
        return {"ok": True, **st.status()}

    if cmd == "shutdown":
        # Caller will close the connection; main loop should exit.
        st.stop_event.set()
        if st.session is not None:
            st.session.close()
            st.session = None
        st.running = False
        return {"ok": True, "shutdown": True}

    return {"ok": False, "error": f"Unknown cmd: {cmd!r}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = ap.parse_args()

    st = WorkerState()

    listener = Listener((args.host, args.port), authkey=AUTHKEY)
    try:
        # Accept and serve. Keep it simple: one client at a time.
        while True:
            conn = listener.accept()
            try:
                while True:
                    try:
                        msg = conn.recv()
                    except EOFError:
                        break
                    if not isinstance(msg, dict):
                        conn.send({"ok": False, "error": "Message must be a dict"})
                        continue
                    resp = handle_command(st, msg)
                    conn.send(resp)
                    if resp.get("shutdown"):
                        return
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
    finally:
        try:
            if st.session is not None:
                st.session.close()
        except Exception:
            pass
        try:
            listener.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
