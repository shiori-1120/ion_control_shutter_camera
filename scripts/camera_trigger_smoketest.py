r"""Smoke test for camera external trigger bootstrap.

Goal:
- Start camera worker (real) and DAQ worker.
- While camera worker performs bootstrap (waits for FRAMEREADY), continuously
  send a simple DO pulse on camera trigger line via DAQ.

This avoids a common deadlock in runners that wait for cam_ready before sending
any triggers.

Usage examples (PowerShell):

    # INTERNAL trigger sanity check (no TTL needed)
        C:/Users/tanak/miniforge3/Scripts/conda.exe run -p C:/Users/tanak/miniforge3 --no-capture-output python scripts/camera_trigger_smoketest.py --camera-mode real --daq-mode dry --trigger-source INTERNAL

    # EXTERNAL trigger bootstrap test (TTL required)
        C:/Users/tanak/miniforge3/Scripts/conda.exe run -p C:/Users/tanak/miniforge3 --no-capture-output python scripts/camera_trigger_smoketest.py --camera-mode real --daq-mode real --device Dev1 --trigger-source EXTERNAL --trigger-connector BNC --trigger-polarity POSITIVE

Note:
- Trigger settings are passed via cam_cfg (env vars are legacy fallback); see src/camera/lib/ControlDevice.py
"""

from __future__ import annotations

import argparse
import queue
import threading
import time
import sys
from pathlib import Path
from multiprocessing import Process, Queue


ALL_OFF = 0b0000
CAMERA_TRIGGER = 0b0100  # port1/line2 (matches shutter_gui)


def main() -> None:
    # Allow running as a plain script from repo root.
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    ap = argparse.ArgumentParser()
    ap.add_argument("--daq-mode", choices=["dry", "real"], default="dry")
    ap.add_argument("--device", default="Dev1")
    ap.add_argument("--camera-mode", choices=["dry", "real"], default="real")
    ap.add_argument("--exposure-s", type=float, default=0.001)
    ap.add_argument("--frame-timeout-s", type=float, default=1.0)
    ap.add_argument("--bootstrap-n", type=int, default=10)
    ap.add_argument("--cam-ready-timeout-s", type=float, default=30.0)

    ap.add_argument("--trigger-source", default="EXTERNAL", choices=["EXTERNAL", "INTERNAL", "EXT", "INT", "1", "2"])
    ap.add_argument("--trigger-connector", default="BNC", choices=["BNC", "MULTI", "INTERFACE"])
    ap.add_argument("--trigger-polarity", default="POSITIVE", choices=["POSITIVE", "NEGATIVE", "POS", "NEG", "RISING", "FALLING"])
    ap.add_argument("--trigger-active", default="EDGE", choices=["EDGE", "LEVEL"])
    ap.add_argument("--trigger-mode", default="NORMAL", choices=["NORMAL", "START"])
    ap.add_argument("--trigger-delay-s", type=float)
    ap.add_argument("--camera-verbose", action="store_true")

    ap.add_argument("--pulse-s", type=float, default=0.002)
    ap.add_argument("--idle-s", type=float, default=0.002)
    ap.add_argument("--prime-period-s", type=float, default=0.010, help="How often to enqueue a trigger sequence")

    args = ap.parse_args()

    # Lazy imports so dry bring-up works on machines without NI-DAQmx.
    from src.camera.ion_state_worker import ion_state_worker_main

    if args.daq_mode == "dry":
        from src.shutter_camera_trigger.daq_worker_dry import daq_worker_dry_main as daq_worker_main
    else:
        from src.shutter_camera_trigger.daq_worker_mpq import daq_worker_mpq_main as daq_worker_main

    daq_cmd_q: Queue = Queue()
    daq_resp_q: Queue = Queue()
    cam_cmd_q: Queue = Queue()
    cam_resp_q: Queue = Queue()

    daq_p = Process(
        target=daq_worker_main,
        args=(daq_cmd_q, daq_resp_q, {"device": str(args.device), "mode": str(args.daq_mode)}),
        daemon=True,
    )
    cam_cfg = {
        "mode": str(args.camera_mode),
        "exposure_s": float(args.exposure_s),
        "frame_timeout_s": float(args.frame_timeout_s),
        "bootstrap_n": int(args.bootstrap_n),
        "trigger": {
            "source": str(args.trigger_source),
            "connector": str(args.trigger_connector),
            "polarity": str(args.trigger_polarity),
            "active": str(args.trigger_active),
            "mode": str(args.trigger_mode),
            **({"delay_s": float(args.trigger_delay_s)} if args.trigger_delay_s is not None else {}),
        },
        "verbose": bool(args.camera_verbose),
    }
    cam_p = Process(target=ion_state_worker_main, args=(cam_cmd_q, cam_resp_q, cam_cfg), daemon=True)

    daq_p.start()
    cam_p.start()

    # Wait DAQ ready first (so priming can start immediately).
    daq_ready = daq_resp_q.get(timeout=10)
    if not daq_ready.get("ok"):
        raise RuntimeError(f"DAQ worker failed: {daq_ready}")

    stop_evt = threading.Event()

    do_sequence = [
        (ALL_OFF, float(args.idle_s)),
        (CAMERA_TRIGGER, float(args.pulse_s)),
        (ALL_OFF, float(args.idle_s)),
    ]

    def prime_loop() -> None:
        while not stop_evt.is_set():
            try:
                daq_cmd_q.put(
                    {
                        "cmd": "run_sequence_once",
                        "do_sequence": do_sequence,
                        "insert_index": -1,
                        "ao_width_ms": 0.0,
                        "ao_rate_hz": 5000.0,
                        "ao_v_high": 5.0,
                        "ao_v_low": 0.0,
                    }
                )
            except Exception:
                pass

            # Drain responses so the queue does not grow unbounded.
            t_end = time.time() + float(args.prime_period_s)
            while time.time() < t_end and not stop_evt.is_set():
                try:
                    daq_resp_q.get(timeout=0.01)
                except queue.Empty:
                    pass
                except Exception:
                    break

    t = threading.Thread(target=prime_loop, daemon=True)
    t.start()

    try:
        cam_ready = cam_resp_q.get(timeout=float(args.cam_ready_timeout_s))
        print("cam_ready:", cam_ready)
        if not cam_ready.get("ok"):
            raise RuntimeError(f"Camera worker failed: {cam_ready}")
        print("OK: camera bootstrap succeeded")
    finally:
        stop_evt.set()
        try:
            daq_cmd_q.put({"cmd": "close"})
        except Exception:
            pass
        try:
            cam_cmd_q.put({"cmd": "close"})
        except Exception:
            pass
        time.sleep(0.2)
        for p in (daq_p, cam_p):
            try:
                if p.is_alive():
                    p.terminate()
            except Exception:
                pass


if __name__ == "__main__":
    main()
