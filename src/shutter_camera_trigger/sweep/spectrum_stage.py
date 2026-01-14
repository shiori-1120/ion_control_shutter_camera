from __future__ import annotations

import csv
import queue
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from ..hardware import CameraQueueDevice, DaqQueueDevice, DaqSequenceCommand


@dataclass(frozen=True)
class SpectrumRunResult:
    results: list[tuple[float, int, int]]  # (freq_hz, n_processed, n_bright)
    shots_csv: Path
    spectrum_csv: Path


def run_spectrum_stage(
    *,
    freqs: list[float],
    do_sequence: list[tuple[int, float]],
    insert_index: int,
    ao_width_ms: float,
    seq_cmd: DaqSequenceCommand | None,
    camera_commands: list[Any] | None,
    n_target: int,
    max_attempt: int,
    settle_s: float,
    update_interval_s: float,
    daq_cmd_q: Any,
    daq_resp_q: Any,
    cam_cmd_q: Any,
    cam_resp_q: Any,
    ao_rate_hz: float,
    mpq_get_with_ui: Callable[[Any, float, str], dict[str, Any]],
    should_stop: Callable[[], bool],
    ui_pump: Callable[[], None] | None,
    status_cb: Callable[[str], None] | None,
    update_point_cb: Callable[[int, float, int, int], None] | None,
    out_dir: Path,
    rig: Any | None = None,
) -> SpectrumRunResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    shots_path = out_dir / "shots.csv"
    spec_path = out_dir / "spectrum.csv"

    next_update = time.time() + max(0.2, float(update_interval_s))
    results: list[tuple[float, int, int]] = []
    seq_cmd_local = seq_cmd or DaqSequenceCommand(
        do_sequence=list(do_sequence),
        ao_insert_index=int(insert_index),
        ao_width_ms=float(ao_width_ms),
        ao_rate_hz=float(ao_rate_hz),
        ao_v_high=5.0,
        ao_v_low=0.0,
    )

    def _status(msg: str) -> None:
        if status_cb is None:
            return
        try:
            status_cb(msg)
        except Exception:
            pass

    def _pump() -> None:
        if ui_pump is None:
            return
        try:
            ui_pump()
        except Exception:
            pass

    def _build_camera_schedule(
        commands: list[Any], *, default_timeout_s: float
    ) -> list[dict[str, Any]]:
        schedule: list[dict[str, Any]] = []
        for cmd in commands:
            try:
                kind = str(getattr(cmd, "kind", cmd.get("kind", ""))).lower()
            except Exception:
                kind = ""
            try:
                meta = dict(getattr(cmd, "meta", cmd.get("meta", {})) or {})
            except Exception:
                meta = {}
            try:
                t_s = float(meta.get("t_s") if "t_s" in meta else getattr(cmd, "meta").get("t_s"))
            except Exception:
                try:
                    t_s = float(getattr(cmd, "t_s", 0.0))
                except Exception:
                    t_s = 0.0
            try:
                timeout_s = float(getattr(cmd, "timeout_s", meta.get("timeout_s", default_timeout_s)))
            except Exception:
                timeout_s = float(default_timeout_s)
            payload = {
                "cmd": "get_frame" if kind == "capture" else "get_state",
                "timeout_s": float(timeout_s),
            }
            schedule.append(
                {
                    "t_s": float(t_s),
                    "payload": payload,
                    "timeout_s": float(timeout_s),
                }
            )
        return schedule

    def _run_timed_sequence(
        *,
        camera_schedule: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        est_s = 0.0
        try:
            est_s = float(sum(float(hold_s) for _, hold_s in seq_cmd_local.do_sequence))
        except Exception:
            est_s = 0.0
        max_cam_timeout = max(
            [float(cmd.get("timeout_s") or 0.0) for cmd in camera_schedule],
            default=0.0,
        )
        overall_timeout = max(5.0, est_s + max_cam_timeout + 2.0)

        pre_cmds = [c for c in camera_schedule if float(c.get("t_s", 0.0)) <= 0.0]
        post_cmds = [c for c in camera_schedule if float(c.get("t_s", 0.0)) > 0.0]
        post_cmds.sort(key=lambda c: float(c.get("t_s", 0.0)))

        for cmd in pre_cmds:
            cam_cmd_q.put(dict(cmd["payload"]))

        t0 = time.monotonic()
        daq_cmd_q.put(
            {
                "cmd": "run_sequence_once",
                "do_sequence": list(seq_cmd_local.do_sequence),
                "insert_index": int(seq_cmd_local.ao_insert_index),
                "ao_width_ms": float(seq_cmd_local.ao_width_ms),
                "ao_rate_hz": float(seq_cmd_local.ao_rate_hz),
                "ao_v_high": float(seq_cmd_local.ao_v_high),
                "ao_v_low": float(seq_cmd_local.ao_v_low),
            }
        )

        expected_cam = len(pre_cmds) + len(post_cmds)
        cam_responses: list[dict[str, Any]] = []
        daq_resp: dict[str, Any] | None = None
        post_idx = 0

        while True:
            now = time.monotonic()
            while post_idx < len(post_cmds):
                t_s = float(post_cmds[post_idx].get("t_s", 0.0))
                if now - t0 < t_s:
                    break
                cam_cmd_q.put(dict(post_cmds[post_idx]["payload"]))
                post_idx += 1

            if daq_resp is None:
                try:
                    daq_resp = daq_resp_q.get_nowait()
                except queue.Empty:
                    pass

            while len(cam_responses) < expected_cam:
                try:
                    cam_responses.append(cam_resp_q.get_nowait())
                except queue.Empty:
                    break

            if daq_resp is not None and len(cam_responses) >= expected_cam:
                return daq_resp, cam_responses

            if now - t0 > overall_timeout:
                raise RuntimeError("Timed sequence timeout")

            _pump()
            time.sleep(0.001)

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
                "label_bright",
                "S_norm",
                "tau_on",
                "tau_off",
                "cam_event",
                "cam_sample",
            ],
        )
        shots_writer.writeheader()

        spec_writer = csv.DictWriter(
            f_spec,
            fieldnames=["step_idx", "freq_hz", "n_processed", "n_bright", "p_bright"],
        )
        spec_writer.writeheader()

        cam_device = CameraQueueDevice(cmd_q=cam_cmd_q)
        camera_schedule = (
            _build_camera_schedule(camera_commands, default_timeout_s=5.0)
            if camera_commands
            else []
        )
        for step_idx, freq in enumerate(freqs):
            if should_stop():
                break

            processed = 0
            n_bright = 0

            if rig is not None:
                try:
                    rig.set_frequency_hz(float(freq))
                    time.sleep(max(0.0, float(settle_s)))
                except Exception:
                    pass

            for attempt_idx in range(int(max_attempt)):
                if should_stop():
                    break
                if processed >= int(n_target):
                    break

                if camera_schedule:
                    daq_resp, cam_responses = _run_timed_sequence(camera_schedule=camera_schedule)
                    if not daq_resp.get("ok"):
                        raise RuntimeError(f"DAQ error: {daq_resp}")
                    cam_resp = cam_responses[-1] if cam_responses else {"ok": False, "event": "timeout"}
                else:
                    cam_device.send_get_state(1.0)
                    try:
                        DaqQueueDevice(cmd_q=daq_cmd_q, resp_q=daq_resp_q).run_sequence_once(
                            seq_cmd_local
                        )
                    except Exception as e:
                        raise RuntimeError(f"DAQ error: {e}")
                    cam_resp = mpq_get_with_ui(cam_resp_q, 5, "Camera response")
                    if not cam_resp.get("ok"):
                        continue
                if not cam_resp.get("ok"):
                    continue

                bright = bool(cam_resp.get("bright"))
                label_bright = cam_resp.get("label_bright")
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
                        "label_bright": "" if label_bright is None else int(bool(label_bright)),
                        "S_norm": "" if s_norm is None else float(s_norm),
                        "tau_on": "" if tau_on is None else float(tau_on),
                        "tau_off": "" if tau_off is None else float(tau_off),
                        "cam_event": str(cam_resp.get("event")),
                        "cam_sample": str(cam_resp.get("sample")) if cam_resp.get("sample") is not None else "",
                    }
                )

                now = time.time()
                if now >= next_update:
                    next_update = now + max(0.2, float(update_interval_s))
                    if update_point_cb is not None:
                        try:
                            update_point_cb(step_idx, float(freq), int(processed), int(n_bright))
                        except Exception:
                            pass
                    _status(
                        f"Running: step {step_idx+1}/{len(freqs)} freq={float(freq):.3e} Hz proc={processed}/{int(n_target)}"
                    )
                    _pump()

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
            results.append((float(freq), int(processed), int(n_bright)))

            if update_point_cb is not None:
                try:
                    update_point_cb(step_idx, float(freq), int(processed), int(n_bright))
                except Exception:
                    pass
            _status(f"Done step {step_idx+1}/{len(freqs)}")
            _pump()

    return SpectrumRunResult(results=results, shots_csv=shots_path, spectrum_csv=spec_path)
