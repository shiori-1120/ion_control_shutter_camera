"""DO→AO→DO sequence runner (USB-6002 friendly).

- DO: on-demand port write (same style as test_all_shutters_multi.py)
- AO: finite waveform pulse with hardware-timed sample clock (same style as test_ao_shutter.py)

This script is intentionally simple:
- DO states are managed as a list
- Durations are controlled by time.sleep() so you can tune numbers easily
- No attempt is made to hard-sync DO edges with AO edges
"""

import time

import numpy as np
import nidaqmx
from nidaqmx.constants import AcquisitionType, LineGrouping, RegenerationMode
from nidaqmx.stream_writers import DigitalSingleChannelWriter


# -------------------------
# Device / channels
# -------------------------
# Adjust to your device name (must be the SAME device for DO and AO)
DEVICE = "Dev3"

# DO: port0 line4..7 as a single 4-bit port
# bit0=line4, bit1=line5, bit2=line6, bit3=line7
PORT_RANGE = f"{DEVICE}/port0/line4:7"

# AO: analog output channel
AO_CH = f"{DEVICE}/ao0"


# -------------------------
# DO patterns (Active High)
# -------------------------
ALL_OFF = 0b0000
ALL_ON = 0b1111

NM_397 = 0b0001  # line4
NM_397_SIG = 0b0010  # line5
# bit2 (line6) is used as Camera Trigger (DO)
CAMERA_TRIGGER = 0b0100  # line6
NM_729 = CAMERA_TRIGGER  # backward-compatible alias (was 729 shutter)
NM_854 = 0b1000  # line7
CAMERA_TRIGGER_854 = 0b1100  # line6 & line7
NM_729_854 = CAMERA_TRIGGER_854  # backward-compatible alias


# -------------------------
# AO pulse settings
# -------------------------
RATE_HZ = 5000
WIDTH_MS = 1.0
V_HIGH = 5.0
V_LOW = 0.0

EDGE_LOW_SAMPLES = 1


def pulse_ao_once(task: nidaqmx.Task) -> None:
    """Output exactly one AO pulse (finite).

    AO_WAVE is expected to be pre-written to the task buffer.
    """
    task.start()
    task.wait_until_done(timeout=5.0)
    task.stop()


def main() -> None:
    # DO output sequence (value, sleep_seconds)
    # You can tune these numbers freely.
    do_sequence: list[tuple[int, float]] = [
        (ALL_OFF, 0.001),
        (NM_397, 0.001),
        (NM_397_SIG, 0.001),
        (CAMERA_TRIGGER_854, 0.001),
        (ALL_OFF, 0.001),
    ]

    # Where to insert AO pulse inside the DO sequence:
    # After writing do_sequence[AO_INSERT_AFTER_INDEX], do one AO pulse.
    AO_INSERT_AFTER_INDEX = 1

    do_task = nidaqmx.Task()
    ao_task = nidaqmx.Task()

    write_port = None
    try:
        # DO setup
        do_task.do_channels.add_do_chan(
            PORT_RANGE,
            line_grouping=LineGrouping.CHAN_FOR_ALL_LINES,
        )
        do_writer = DigitalSingleChannelWriter(do_task.out_stream)
        write_port = do_writer.write_one_sample_port_uint16

        # AO setup
        ao_task.ao_channels.add_ao_voltage_chan(AO_CH, min_val=0.0, max_val=5.0)

        # Build one finite AO pulse from WIDTH_MS (ms).
        # Note: actual width is quantized by 1/RATE_HZ seconds.
        if WIDTH_MS <= 0:
            raise ValueError("WIDTH_MS must be > 0")
        n_high = max(1, int(round((WIDTH_MS / 1000.0) * RATE_HZ)))
        actual_width_ms = (n_high / RATE_HZ) * 1000.0
        ao_wave = np.concatenate(
            [
                np.full(EDGE_LOW_SAMPLES, V_LOW, dtype=np.float64),
                np.full(n_high, V_HIGH, dtype=np.float64),
                np.full(EDGE_LOW_SAMPLES, V_LOW, dtype=np.float64),
            ]
        )
        ao_task.timing.cfg_samp_clk_timing(
            RATE_HZ,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=len(ao_wave),
        )
        # Write the AO waveform once; reuse it for each pulse (less overhead).
        ao_task.out_stream.regen_mode = RegenerationMode.ALLOW_REGENERATION
        ao_task.write(ao_wave, auto_start=False)

        print(
            f"AO pulse width: requested={WIDTH_MS} ms, actual={actual_width_ms:.4f} ms "
            f"(RATE_HZ={RATE_HZ}, N_HIGH={n_high})"
        )

        print("Initializing... ALL OFF")
        write_port(ALL_OFF)
        time.sleep(0.01)

        print("DO→AO→DO sequence start (Ctrl+C to stop)")
        while True:
            for idx, (do_value, hold_s) in enumerate(do_sequence):
                write_port(int(do_value))
                time.sleep(float(hold_s))

                if idx == AO_INSERT_AFTER_INDEX:
                    pulse_ao_once(ao_task)

    except KeyboardInterrupt:
        pass
    finally:
        try:
            if write_port is not None:
                write_port(ALL_OFF)
        except Exception:
            pass

        try:
            ao_task.close()
        except Exception:
            pass

        try:
            do_task.close()
        except Exception:
            pass

        print("終了")


if __name__ == "__main__":
    main()
