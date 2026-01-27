from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path
from typing import Any
from tkinter import messagebox
import tkinter as tk

import numpy as np

from ..daq.guards import require_connected

from ..gui_support.diagnostics import resolve_log_path, set_last_error
from ..gui_support.camera_worker_manager import build_cam_cfg, ensure_camera_worker
from ..gui_tabs.camera_tab import update_camera_plot
from ..config.device_registry import resolve_output_root
from ..hardware import DaqClientDevice
from ..gui_support.output_state import set_output_state
from ..sequence.spec import build_sequence_spec, compile_sequence_spec
from ..sweep.session_parse import read_sequence_json_params


def _load_sequence_params(seq_path: Path):
    return read_sequence_json_params(seq_path=seq_path)


def start_sequence(
    app: Any,
    *,
    seq_path: Path,
    ao_rate_hz: float,
    nm_397: int,
) -> None:
    try:
        if not seq_path or not Path(seq_path).exists():
            raise FileNotFoundError(f"Sequence JSON not found: {seq_path}")
        require_connected(app)
        params = _load_sequence_params(Path(seq_path))
        ao_insert_index = int(params.ao_insert_index)
        ao_width_ms = float(params.ao_width_ms)
        try:
            raw = str(getattr(app, "insert_index_var", None).get() or "").strip()
        except Exception:
            raw = ""
        if raw:
            try:
                ao_insert_index = int(float(raw))
            except Exception as e:
                raise ValueError(f"Invalid AO insert index: {raw!r}") from e
            if ao_insert_index < -1 or ao_insert_index >= len(params.do_sequence):
                raise ValueError(
                    f"AO insert index must be -1..{len(params.do_sequence) - 1} (got {ao_insert_index})"
                )
        try:
            raw_width = str(getattr(app, "width_var", None).get() or "").strip()
        except Exception:
            raw_width = ""
        if raw_width:
            try:
                ao_width_ms = float(raw_width)
            except Exception as e:
                raise ValueError(f"Invalid AO width (ms): {raw_width!r}") from e
            if ao_width_ms < 0:
                raise ValueError("AO width (ms) must be >= 0")
        seq_spec = build_sequence_spec(
            do_sequence=params.do_sequence,
            ao_insert_index=int(ao_insert_index),
            ao_width_ms=float(ao_width_ms),
            ao_rate_hz=float(ao_rate_hz),
            ao_v_high=5.0,
            ao_v_low=0.0,
            camera_actions=params.camera_actions,
            sync_markers=params.sync_markers,
        )
        seq_cmd, _ = compile_sequence_spec(seq_spec)
        capture_enabled, capture_show_n = _resolve_seq_capture_settings(app)
        capture_cfg = None
        if capture_enabled:
            cam_cfg = build_cam_cfg(app)
            ensure_camera_worker(app, cam_cfg=cam_cfg, ready_timeout_s=90.0)
            cmd_q = getattr(app, "_cam_worker_cmd_q", None)
            resp_q = getattr(app, "_cam_worker_resp_q", None)
            if cmd_q is None or resp_q is None:
                raise RuntimeError("Camera worker queues missing")
            frame_timeout_s = float(cam_cfg.get("frame_timeout_s") or 5.0)
            out_dir = _resolve_sequence_capture_dir(app)
            out_dir.mkdir(parents=True, exist_ok=True)
            app._seq_capture_count = 0
            app._seq_capture_shown = 0
            capture_cfg = {
                "cmd_q": cmd_q,
                "resp_q": resp_q,
                "frame_timeout_s": frame_timeout_s,
                "out_dir": out_dir,
                "show_max": int(capture_show_n),
            }
    except Exception as e:
        messagebox.showerror("Sequence", str(e))
        set_last_error(
            app,
            label="Sequence",
            message=str(e),
            log_path=resolve_log_path(app, filename="app.log"),
        )
        return

    app._seq_running = True
    app._seq_thread = threading.Thread(
        target=sequence_loop,
        args=(app, seq_cmd, int(nm_397), capture_cfg),
        daemon=True,
    )
    app._seq_thread.start()

    app.status_var.set(f"Connected: {app._daq_device} ({app._daq_mode}) | Sequence running")
    app.start_btn.configure(state=tk.DISABLED)
    app.stop_btn.configure(state=tk.NORMAL)


def sequence_stopped_ui(app: Any, *, nm_397: int) -> None:
    app.start_btn.configure(state=tk.NORMAL)
    app.stop_btn.configure(state=tk.DISABLED)
    if app._daq.connected:
        try:
            value = int(nm_397)
            DaqClientDevice(app._daq).set_do(value)
            set_output_state(app, value)
        except Exception:
            pass
        app.status_var.set(f"Connected: {app._daq_device} ({app._daq_mode})")


def stop_sequence(app: Any, *, nm_397: int) -> None:
    app._seq_running = False

    if app._seq_thread is None:
        sequence_stopped_ui(app, nm_397=nm_397)
        return

    try:
        alive = app._seq_thread.is_alive()
    except Exception:
        alive = False

    if not alive:
        sequence_stopped_ui(app, nm_397=nm_397)
        return

    try:
        app.start_btn.configure(state=tk.DISABLED)
        app.stop_btn.configure(state=tk.DISABLED)
    except Exception:
        pass

    if not app._seq_stop_polling:
        app._seq_stop_polling = True
        app.after(100, lambda: poll_sequence_stop(app, nm_397=nm_397))


def poll_sequence_stop(app: Any, *, nm_397: int) -> None:
    try:
        t = app._seq_thread
        alive = bool(t and t.is_alive())
    except Exception:
        alive = False

    if alive:
        app.after(100, lambda: poll_sequence_stop(app, nm_397=nm_397))
        return

    app._seq_stop_polling = False
    sequence_stopped_ui(app, nm_397=nm_397)


def sequence_loop(
    app: Any,
    seq_cmd: Any,
    nm_397: int,
    capture_cfg: dict[str, Any] | None = None,
) -> None:
    try:
        est_s = 0.0
        try:
            est_s = float(sum(float(hold_s) for _, hold_s in seq_cmd.do_sequence))
        except Exception:
            est_s = 0.0
        req_timeout = max(5.0, est_s + 2.0)
        capture_enabled = bool(capture_cfg)
        cmd_q = capture_cfg.get("cmd_q") if capture_cfg else None
        resp_q = capture_cfg.get("resp_q") if capture_cfg else None
        frame_timeout_s = float(capture_cfg.get("frame_timeout_s")) if capture_cfg else 0.0
        out_dir = capture_cfg.get("out_dir") if capture_cfg else None
        show_max = int(capture_cfg.get("show_max")) if capture_cfg else 0
        resp_timeout_s = max(2.0, req_timeout + frame_timeout_s + 1.0)
        seq_idx = 0

        while app._seq_running:
            if capture_enabled and cmd_q is not None:
                try:
                    cmd_q.put(
                        {
                            "cmd": "get_frame",
                            "timeout_s": float(frame_timeout_s),
                            "tag": f"seq-{seq_idx}",
                        }
                    )
                except Exception:
                    pass
            DaqClientDevice(app._daq).run_sequence_once(
                seq_cmd
            )
            if capture_enabled and resp_q is not None:
                try:
                    resp = resp_q.get(timeout=float(resp_timeout_s))
                except Exception:
                    resp = None
                if isinstance(resp, dict) and resp.get("ok") and resp.get("frame") is not None:
                    try:
                        arr = np.asarray(resp.get("frame"))
                    except Exception:
                        arr = None
                    if arr is not None:
                        try:
                            if out_dir is not None:
                                idx = int(getattr(app, "_seq_capture_count", 0))
                                np.save(Path(out_dir) / f"seq_{idx:05d}.npy", arr)
                                app._seq_capture_count = idx + 1
                        except Exception:
                            pass
                        try:
                            shown = int(getattr(app, "_seq_capture_shown", 0))
                            if show_max > 0 and shown < show_max:
                                app._seq_capture_shown = shown + 1
                                app.after(
                                    0,
                                    lambda img=arr, i=seq_idx + 1: update_camera_plot(
                                        app, img, title=f"Sequence frame {i}"
                                    ),
                                )
                        except Exception:
                            pass
                        try:
                            app.after(
                                0,
                                lambda i=seq_idx + 1, shape=arr.shape: app._cam_status.set(
                                    f"Seq capture {i} shape={shape}"
                                ),
                            )
                        except Exception:
                            pass
            seq_idx += 1
    except Exception as e:
        err = str(e)
        app.after(0, lambda msg=err: messagebox.showerror("Sequence", msg))
        app.after(
            0,
            lambda msg=err: set_last_error(
                app,
                label="Sequence",
                message=msg,
                log_path=resolve_log_path(app, filename="app.log"),
            ),
        )
    finally:
        app._seq_running = False
        app.after(0, lambda: sequence_stopped_ui(app, nm_397=nm_397))


def _resolve_seq_capture_settings(app: Any) -> tuple[bool, int]:
    try:
        enabled = bool(getattr(app, "seq_capture_enable_var", None) and app.seq_capture_enable_var.get())
    except Exception:
        enabled = False
    try:
        raw = str(getattr(app, "seq_capture_show_n_var", None).get() or "").strip()
    except Exception:
        raw = ""
    try:
        show_n = int(float(raw)) if raw else 0
    except Exception:
        show_n = 0
    if show_n < 0:
        show_n = 0
    return enabled, show_n


def _resolve_sequence_capture_dir(app: Any) -> Path:
    try:
        root = getattr(app, "output_root", None)
        if root:
            base = Path(root)
        else:
            base = resolve_output_root()
    except Exception:
        base = resolve_output_root()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base / "sequence_capture" / ts
