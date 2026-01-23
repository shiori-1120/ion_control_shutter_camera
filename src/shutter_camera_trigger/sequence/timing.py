from __future__ import annotations

import queue
import time
from typing import Any, Callable

from ..hardware import CameraCommand, DaqSequenceCommand


def build_camera_schedule(
    camera_commands: list[CameraCommand | dict[str, Any]],
    *,
    default_timeout_s: float,
) -> list[dict[str, Any]]:
    schedule: list[dict[str, Any]] = []
    for idx, cmd in enumerate(camera_commands):
        kind = ""
        meta: dict[str, Any] = {}
        timeout_s = default_timeout_s
        try:
            if isinstance(cmd, dict):
                kind = str(cmd.get("kind", ""))
                meta = dict(cmd.get("meta") or {})
                timeout_s = float(cmd.get("timeout_s", meta.get("timeout_s", default_timeout_s)))
            else:
                kind = str(getattr(cmd, "kind", ""))
                meta = dict(getattr(cmd, "meta", {}) or {})
                timeout_s = float(getattr(cmd, "timeout_s", meta.get("timeout_s", default_timeout_s)))
        except Exception:
            kind = ""
            meta = {}
            timeout_s = default_timeout_s

        try:
            t_s = float(meta.get("t_s", 0.0))
        except Exception:
            t_s = 0.0
        tag = meta.get("tag")
        if tag is None or str(tag) == "":
            base = kind.lower().strip() or "action"
            tag = f"{base}@{t_s:.6f}"

        payload = {
            "cmd": "get_frame" if kind.lower() == "capture" else "get_state",
            "timeout_s": float(timeout_s),
            "tag": tag,
        }
        schedule.append(
            {
                "t_s": float(t_s),
                "payload": payload,
                "timeout_s": float(timeout_s),
            }
        )
    return schedule


def run_timed_sequence(
    *,
    seq_cmd: DaqSequenceCommand,
    daq_cmd_q: Any,
    daq_resp_q: Any,
    cam_cmd_q: Any,
    cam_resp_q: Any,
    camera_schedule: list[dict[str, Any]],
    ui_pump: Callable[[], None] | None = None,
    on_cam_resp: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    est_s = 0.0
    try:
        est_s = float(sum(float(hold_s) for _, hold_s in seq_cmd.do_sequence))
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
            "do_sequence": list(seq_cmd.do_sequence),
            "insert_index": int(seq_cmd.ao_insert_index),
            "ao_width_ms": float(seq_cmd.ao_width_ms),
            "ao_rate_hz": float(seq_cmd.ao_rate_hz),
            "ao_v_high": float(seq_cmd.ao_v_high),
            "ao_v_low": float(seq_cmd.ao_v_low),
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
                resp = cam_resp_q.get_nowait()
                cam_responses.append(resp)
                if on_cam_resp is not None and isinstance(resp, dict):
                    try:
                        on_cam_resp(resp)
                    except Exception:
                        pass
            except queue.Empty:
                break

        if daq_resp is not None and len(cam_responses) >= expected_cam:
            return daq_resp, cam_responses

        if now - t0 > overall_timeout:
            raise RuntimeError("Timed sequence timeout")

        if ui_pump is not None:
            try:
                ui_pump()
            except Exception:
                pass
        time.sleep(0.001)


def select_last_success_response(responses: list[dict[str, Any]]) -> dict[str, Any]:
    for resp in reversed(responses):
        if resp.get("ok"):
            return resp
    return responses[-1] if responses else {"ok": False, "event": "timeout"}
