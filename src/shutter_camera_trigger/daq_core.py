"""Headless NI-DAQmx helpers for shutter/trigger sequences.

This is a GUI-free extraction of the essential parts used in shutter_gui.
We keep it small and deterministic:
- DO set (port write)
- optional AO finite pulse (hardware-timed by sample clock)
- software-timed wait with reduced drift (perf_counter)

DAQ is explicitly *software-timed* for DO durations.
"""

from __future__ import annotations

import time
from typing import Any


ALL_OFF = 0b0000

# AO waveform: we add 1 LOW sample on both edges for clarity/safety.
AO_EDGE_LOW_SAMPLES = 1


def wait_s(duration_s: float) -> None:
    """Software-timed wait with reduced drift.

    Uses perf_counter() and waits until the target time.
    Jitter still exists on Windows, but this avoids accumulating drift.
    """
    end_t = time.perf_counter() + max(0.0, float(duration_s))
    while True:
        remaining = end_t - time.perf_counter()
        if remaining <= 0:
            return
        if remaining > 0.005:
            time.sleep(remaining - 0.002)
        else:
            time.sleep(0)


class DaqSession:
    def __init__(self, *, device: str) -> None:
        # Lazy import so that --daq-mode dry works even without NI-DAQmx installed.
        import nidaqmx

        self._nidaqmx = nidaqmx

        self.device = device
        self.port_range = f"{device}/port0/line4:7"
        self.ao_ch = f"{device}/ao0"

        self._do_task: Any = nidaqmx.Task()
        self._ao_task: Any | None = None
        self._write_port = None

        self.actual_width_ms: float | None = None
        self._ao_rate_hz: float | None = None
        self._ao_width_ms: float | None = None
        self._ao_v_high: float | None = None
        self._ao_v_low: float | None = None

        self._setup_do()
        self.set_do(ALL_OFF)

    def _setup_do(self) -> None:
        from nidaqmx.constants import LineGrouping
        from nidaqmx.stream_writers import DigitalSingleChannelWriter

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

        import numpy as np
        from nidaqmx.constants import AcquisitionType, RegenerationMode

        ao_task = self._nidaqmx.Task()

        n_high = max(1, int(round((width_ms / 1000.0) * rate_hz)))
        self.actual_width_ms = (n_high / rate_hz) * 1000.0

        edge_low_samples = AO_EDGE_LOW_SAMPLES
        ao_wave = np.concatenate(
            [
                np.full(edge_low_samples, v_low, dtype=np.float64),
                np.full(n_high, v_high, dtype=np.float64),
                np.full(edge_low_samples, v_low, dtype=np.float64),
            ]
        )

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


class DryDaqSession:
    """DAQ session stub for bring-up without NI-DAQ hardware.

    It simulates timing (wait_s) and tracks the last DO value.
    """

    def __init__(self, *, device: str = "(dry)") -> None:
        self.device = device
        self.actual_width_ms: float | None = None
        self._last_do: int = ALL_OFF
        self._ao_total_s: float = 0.0

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
        if width_ms <= 0:
            raise ValueError("AO width_ms must be > 0")
        if rate_hz <= 0:
            raise ValueError("AO rate_hz must be > 0")

        n_high = max(1, int(round((width_ms / 1000.0) * rate_hz)))
        self.actual_width_ms = (n_high / rate_hz) * 1000.0
        self._ao_total_s = (n_high + 2 * AO_EDGE_LOW_SAMPLES) / rate_hz

    def set_do(self, value: int) -> None:
        self._last_do = int(value)

    def pulse_ao_once(self) -> None:
        # Simulate a hardware-timed AO waveform duration.
        wait_s(float(self._ao_total_s))

    def close(self) -> None:
        self._last_do = ALL_OFF


def run_do_sequence_once(
    session: DaqSession,
    do_sequence: list[tuple[int, float]],
    *,
    insert_index: int,
    ao_rate_hz: float,
    ao_width_ms: float,
    ao_v_high: float = 5.0,
    ao_v_low: float = 0.0,
) -> None:
    """Run exactly one DO sequence, optionally inserting one AO pulse."""
    if insert_index >= 0:
        session.ensure_ao_config(width_ms=ao_width_ms, rate_hz=ao_rate_hz, v_high=ao_v_high, v_low=ao_v_low)

    for idx, (do_value, hold_s) in enumerate(do_sequence):
        session.set_do(int(do_value))
        wait_s(float(hold_s))
        if insert_index >= 0 and idx == int(insert_index):
            session.pulse_ao_once()

    session.set_do(ALL_OFF)
