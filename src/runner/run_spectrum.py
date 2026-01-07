"""Run a simple excitation spectrum acquisition (same PC, multi-process).

Architecture:
- Main process:
  - sets Rigol DG frequency (USB, pyVISA)
  - orchestrates steps and logging
- DAQ worker process:
  - software-timed DO sequence + optional AO trigger pulse
- Camera worker process:
  - waits for FRAMEREADY and classifies bright/dark

This runner is intentionally minimal to get experiments running quickly.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import threading
import time
from datetime import datetime
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Any



def _limit_blas_threads() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def _parse_sequence_text(raw: str) -> list[tuple[int, float]]:
    steps: list[tuple[int, float]] = []
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 2:
            raise ValueError(f"Invalid sequence line: {line!r}")
        key = parts[0]
        hold_s = float(parts[1])
        if hold_s < 0:
            raise ValueError(f"hold_s must be >= 0: {line!r}")

        if all(ch in "01" for ch in key):
            value = int(key, 2)
        else:
            value = int(key, 0)

        if not (0 <= value <= 0b1111):
            raise ValueError(f"DO value must be 0..15 (4-bit): {line!r}")

        steps.append((int(value), float(hold_s)))

    if not steps:
        raise ValueError("Sequence is empty")
    return steps


def _load_sequence_json(path: str) -> tuple[list[tuple[int, float]], int, float]:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    raw = str(data.get("sequence_text") or "")
    insert_index = int(data.get("ao_insert_index", -1))
    ao_width_ms = float(data.get("ao_width_ms", 1.0))
    return _parse_sequence_text(raw), insert_index, ao_width_ms


def _make_run_dir(base: str = "data/output/spectrum") -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(base) / ts
    out.mkdir(parents=True, exist_ok=True)
    return out


def _freq_list_from_args(args: argparse.Namespace) -> list[float]:
    if args.freq_list:
        p = Path(args.freq_list)
        freqs: list[float] = []
        for line in p.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            freqs.append(float(s))
        if not freqs:
            raise ValueError("freq_list is empty")
        return freqs

    if args.freq_start is None or args.freq_stop is None or args.freq_step is None:
        raise ValueError("Specify either --freq-list OR (--freq-start, --freq-stop, --freq-step)")

    start = float(args.freq_start)
    stop = float(args.freq_stop)
    step = float(args.freq_step)
    if step == 0:
        raise ValueError("freq_step must be non-zero")

    freqs = []
    f = start
    if step > 0:
        while f <= stop + 1e-12:
            freqs.append(float(f))
            f += step
    else:
        while f >= stop - 1e-12:
            freqs.append(float(f))
            f += step
    if not freqs:
        raise ValueError("No frequencies generated")
    return freqs


def main() -> None:
    _limit_blas_threads()

    # Import workers lazily to allow dry bring-up on machines without NI-DAQmx/pyvisa.
    from src.camera.ion_state_worker import ion_state_worker_main

    ap = argparse.ArgumentParser()
    ap.add_argument("--visa-resource", help="DG922 VISA resource string (USB...)")
    ap.add_argument("--no-fg", action="store_true", help="Do not control FG (dry bring-up)")
    ap.add_argument("--freq-start", type=float)
    ap.add_argument("--freq-stop", type=float)
    ap.add_argument("--freq-step", type=float)
    ap.add_argument("--freq-list", help="Text file of frequencies (one per line)")

    ap.add_argument("--n-target", type=int, default=300, help="Target processed shots per frequency")
    ap.add_argument("--max-attempt", type=int, default=600, help="Max attempts per frequency (includes timeouts)")
    ap.add_argument("--settle-s", type=float, default=0.02, help="Wait after setting frequency")

    ap.add_argument("--daq-mode", choices=["dry", "real"], default="real")
    ap.add_argument("--device", default="Dev1", help="NI-DAQ device name (e.g., Dev1)")
    ap.add_argument("--sequence-json", required=True, help="Sequence JSON saved from shutter_gui (Save)")

    ap.add_argument("--camera-mode", choices=["dry", "real"], default="dry")
    ap.add_argument("--exposure-s", type=float, default=0.001)
    ap.add_argument("--frame-timeout-s", type=float, default=1.0)
    ap.add_argument("--bootstrap-n", type=int, default=10)
    ap.add_argument("--roi", nargs=4, type=int, help="ROI as xw yw xs ys")
    ap.add_argument("--bg-roi", nargs=4, type=int, help="Background ROI as xw yw xs ys")

    ap.add_argument("--trigger-source", default="EXTERNAL", choices=["EXTERNAL", "INTERNAL", "EXT", "INT", "1", "2"])
    ap.add_argument("--trigger-connector", default="BNC", choices=["BNC", "MULTI", "INTERFACE"])
    ap.add_argument("--trigger-polarity", default="POSITIVE", choices=["POSITIVE", "NEGATIVE", "POS", "NEG", "RISING", "FALLING"])
    ap.add_argument("--trigger-active", default="EDGE", choices=["EDGE", "LEVEL"])
    ap.add_argument("--trigger-mode", default="NORMAL", choices=["NORMAL", "START"])
    ap.add_argument("--trigger-delay-s", type=float)
    ap.add_argument("--camera-verbose", action="store_true")

    args = ap.parse_args()

    if not args.no_fg and not args.visa_resource:
        raise SystemExit("Specify --visa-resource, or use --no-fg for dry bring-up")

    freqs = _freq_list_from_args(args)
    do_sequence, insert_index, ao_width_ms = _load_sequence_json(args.sequence_json)

    out_dir = _make_run_dir()
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "visa_resource": args.visa_resource,
                "freqs": freqs,
                "n_target": args.n_target,
                "max_attempt": args.max_attempt,
                "settle_s": args.settle_s,
                "daq_mode": args.daq_mode,
                "device": args.device,
                "sequence_json": args.sequence_json,
                "insert_index": insert_index,
                "ao_width_ms": ao_width_ms,
                "camera_mode": args.camera_mode,
                "exposure_s": args.exposure_s,
                "frame_timeout_s": args.frame_timeout_s,
                "bootstrap_n": args.bootstrap_n,
                "roi": args.roi,
                "bg_roi": args.bg_roi,
                "trigger": {
                    "source": args.trigger_source,
                    "connector": args.trigger_connector,
                    "polarity": args.trigger_polarity,
                    "active": args.trigger_active,
                    "mode": args.trigger_mode,
                    "delay_s": args.trigger_delay_s,
                },
                "camera_verbose": bool(args.camera_verbose),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    shots_path = out_dir / "shots.csv"
    spec_path = out_dir / "spectrum.csv"

    daq_cmd_q: Queue = Queue()
    daq_resp_q: Queue = Queue()
    cam_cmd_q: Queue = Queue()
    cam_resp_q: Queue = Queue()

    if args.daq_mode == "dry":
        from src.shutter_camera_trigger.daq_worker_dry import daq_worker_dry_main as daq_worker_main
    else:
        from src.shutter_camera_trigger.daq_worker_mpq import daq_worker_mpq_main as daq_worker_main

    daq_p = Process(
        target=daq_worker_main,
        args=(daq_cmd_q, daq_resp_q, {"device": args.device, "mode": args.daq_mode}),
        daemon=True,
    )
    cam_cfg: dict[str, Any] = {
        "mode": args.camera_mode,
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
    if args.roi:
        cam_cfg["roi"] = list(map(int, args.roi))
    if args.bg_roi:
        cam_cfg["bg_roi"] = list(map(int, args.bg_roi))

    cam_p = Process(target=ion_state_worker_main, args=(cam_cmd_q, cam_resp_q, cam_cfg), daemon=True)

    daq_p.start()
    cam_p.start()

    # If camera bootstrap waits for external triggers, waiting for cam_ready
    # before sending any TTL can deadlock. Prime the trigger line while waiting.
    stop_prime = threading.Event()

    ALL_OFF = 0b0000
    CAMERA_TRIGGER = 0b0100  # port1/line2 (matches shutter_gui)
    prime_sequence = [(ALL_OFF, 0.002), (CAMERA_TRIGGER, 0.002), (ALL_OFF, 0.002)]

    def _prime_loop() -> None:
        while not stop_prime.is_set():
            try:
                daq_cmd_q.put(
                    {
                        "cmd": "run_sequence_once",
                        "do_sequence": prime_sequence,
                        "insert_index": -1,
                        "ao_width_ms": 0.0,
                        "ao_rate_hz": 5000.0,
                        "ao_v_high": 5.0,
                        "ao_v_low": 0.0,
                    }
                )
            except Exception:
                pass

            # Drain any responses so the queue doesn't grow.
            t_end = time.time() + 0.01
            while time.time() < t_end and not stop_prime.is_set():
                try:
                    daq_resp_q.get(timeout=0.01)
                except queue.Empty:
                    break
                except Exception:
                    break

    prime_thread = None
    if args.camera_mode == "real":
        prime_thread = threading.Thread(target=_prime_loop, daemon=True)
        prime_thread.start()

    # wait ready
    daq_ready = daq_resp_q.get(timeout=10)
    cam_ready = cam_resp_q.get(timeout=30)

    stop_prime.set()
    if not daq_ready.get("ok"):
        raise RuntimeError(f"DAQ worker failed: {daq_ready}")
    if not cam_ready.get("ok"):
        raise RuntimeError(f"Camera worker failed: {cam_ready}")

    rig = None
    if not args.no_fg:
        # Rigol DG control
        from src.lib.instruments.rigol_dg import RigolDG, RigolDgConfig

        rig = RigolDG(RigolDgConfig(visa_resource=str(args.visa_resource), channel=1, timeout_ms=5000))
        rig.open()
        try:
            idn = rig.idn()
        except Exception:
            idn = "(IDN failed)"

        (out_dir / "idn.txt").write_text(str(idn) + "\n", encoding="utf-8")
        rig.output(True)

    with shots_path.open("w", newline="", encoding="utf-8") as f_shots, spec_path.open(
        "w", newline="", encoding="utf-8"
    ) as f_spec:
        shots_writer = csv.DictWriter(
            f_shots,
            fieldnames=[
                "t_iso",
                "step_idx",
                "freq_hz",
                "attempt_idx",
                "processed_idx",
                "bright",
                "S_norm",
                "tau_on",
                "tau_off",
                "cam_event",
            ],
        )
        shots_writer.writeheader()

        spec_writer = csv.DictWriter(f_spec, fieldnames=["step_idx", "freq_hz", "n_processed", "n_bright", "p_bright"])
        spec_writer.writeheader()

        for step_idx, freq in enumerate(freqs):
            if rig is not None:
                rig.set_frequency_hz(freq)
                time.sleep(max(0.0, float(args.settle_s)))

            processed = 0
            n_bright = 0

            for attempt_idx in range(int(args.max_attempt)):
                if processed >= int(args.n_target):
                    break

                # Arm camera first (waits for next frame), then trigger via DAQ.
                cam_cmd_q.put({"cmd": "get_state", "timeout_s": float(args.frame_timeout_s)})

                daq_cmd_q.put(
                    {
                        "cmd": "run_sequence_once",
                        "do_sequence": do_sequence,
                        "insert_index": int(insert_index),
                        "ao_width_ms": float(ao_width_ms),
                        "ao_rate_hz": 5000.0,
                        "ao_v_high": 5.0,
                        "ao_v_low": 0.0,
                    }
                )

                daq_resp = daq_resp_q.get(timeout=10)
                if not daq_resp.get("ok"):
                    # DAQ failure: stop early; continuing would desync triggers.
                    raise RuntimeError(f"DAQ error: {daq_resp}")
                cam_resp = cam_resp_q.get(timeout=max(2.0, float(args.frame_timeout_s) + 1.0))

                if not cam_resp.get("ok"):
                    # timeout or error -> drop
                    continue

                bright = bool(cam_resp.get("bright"))
                s_norm = cam_resp.get("S_norm")
                tau_on = cam_resp.get("tau_on")
                tau_off = cam_resp.get("tau_off")

                processed += 1
                if bright:
                    n_bright += 1

                shots_writer.writerow(
                    {
                        "t_iso": datetime.now().isoformat(timespec="milliseconds"),
                        "step_idx": step_idx,
                        "freq_hz": float(freq),
                        "attempt_idx": attempt_idx,
                        "processed_idx": processed,
                        "bright": int(bright),
                        "S_norm": "" if s_norm is None else float(s_norm),
                        "tau_on": "" if tau_on is None else float(tau_on),
                        "tau_off": "" if tau_off is None else float(tau_off),
                        "cam_event": str(cam_resp.get("event")),
                    }
                )

            p_bright = (n_bright / processed) if processed > 0 else 0.0
            spec_writer.writerow(
                {
                    "step_idx": step_idx,
                    "freq_hz": float(freq),
                    "n_processed": processed,
                    "n_bright": n_bright,
                    "p_bright": float(p_bright),
                }
            )

    # Cleanup
    if rig is not None:
        try:
            rig.output(False)
        except Exception:
            pass
        rig.close()

    daq_cmd_q.put({"cmd": "close"})
    cam_cmd_q.put({"cmd": "close"})

    # Give workers time to exit
    time.sleep(0.2)


if __name__ == "__main__":
    main()
