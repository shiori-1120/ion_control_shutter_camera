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
from ..hardware import DaqClientDevice
from ..gui_support.output_state import set_output_state
from ..sequence.spec import build_sequence_spec, compile_sequence_spec
from ..sequence.timing import build_camera_schedule, run_timed_sequence, select_last_success_response
from ..sweep.session_parse import read_sequence_json_params
from ..gui_support.camera_worker_manager import build_cam_cfg, ensure_camera_worker
from ..gui_tabs.camera_tab import _build_prime_cb, _is_external_trigger, _require_external_trigger, plot_camera_frame
from ..config.device_registry import resolve_output_root


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
        args=(app, seq_cmd, int(nm_397)),
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
) -> None:
    try:
        est_s = 0.0
        try:
            est_s = float(sum(float(hold_s) for _, hold_s in seq_cmd.do_sequence))
        except Exception:
            est_s = 0.0
        req_timeout = max(5.0, est_s + 2.0)

        while app._seq_running:
            DaqClientDevice(app._daq).run_sequence_once(
                seq_cmd
            )
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


def start_sequence_preview(
    app: Any,
    *,
    seq_path: Path,
    ao_rate_hz: float,
    nm_397: int,
    n_runs: int,
    camera_trigger: int,
    roi_pulse_s: float,
    roi_idle_s: float,
) -> None:
    try:
        if app._seq_running:
            raise RuntimeError("Sequence is running; stop it before preview.")
        if getattr(app, "_seq_preview_running", False):
            raise RuntimeError("Sequence preview already running.")
        if int(n_runs) <= 0:
            raise ValueError("Preview N must be > 0")
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
            ao_insert_index = int(float(raw))
        try:
            raw_width = str(getattr(app, "width_var", None).get() or "").strip()
        except Exception:
            raw_width = ""
        if raw_width:
            ao_width_ms = float(raw_width)

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
        seq_cmd, cam_cmds = compile_sequence_spec(seq_spec)
        camera_schedule = build_camera_schedule(cam_cmds, default_timeout_s=5.0)
        if not camera_schedule:
            raise RuntimeError("camera_actions is empty; add capture actions to the sequence JSON.")

        cam_cfg = build_cam_cfg(app)
        trig_cfg = cam_cfg.get("trigger") or {}
        _require_external_trigger(trig_cfg)
        ready_timeout_s = 30.0
        prime_cb = None
        if cam_cfg.get("mode") == "real" and _is_external_trigger(trig_cfg):
            ready_timeout_s = 180.0
            prime_cb = _build_prime_cb(
                app,
                nm_397=nm_397,
                camera_trigger=camera_trigger,
                roi_pulse_s=roi_pulse_s,
                roi_idle_s=roi_idle_s,
                ao_rate_hz=ao_rate_hz,
            )
        ensure_camera_worker(app, cam_cfg=cam_cfg, ready_timeout_s=ready_timeout_s, prime_cb=prime_cb)

        daq_cmd_q = getattr(app._daq, "_cmd_q", None)
        daq_resp_q = getattr(app._daq, "_resp_q", None)
        cam_cmd_q = getattr(app, "_cam_worker_cmd_q", None)
        cam_resp_q = getattr(app, "_cam_worker_resp_q", None)
        if not (daq_cmd_q and daq_resp_q and cam_cmd_q and cam_resp_q):
            raise RuntimeError("Worker queues not ready")

    except Exception as e:
        messagebox.showerror("Sequence preview", str(e))
        set_last_error(
            app,
            label="Sequence preview",
            message=str(e),
            log_path=resolve_log_path(app, filename="app.log"),
        )
        return

    app._seq_preview_running = True
    _set_sequence_preview_ui(app, running=True)

    out_dir = resolve_output_root() / "sequence_preview" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    def _worker() -> None:
        err: str | None = None
        try:
            total = int(n_runs)
            for run_idx in range(1, total + 1):
                if not app._seq_preview_running:
                    break
                app.after(0, lambda i=run_idx: app._cam_status.set(f"Preview {i}/{total}"))
                run_dir = out_dir / f"run_{run_idx:04d}"
                run_dir.mkdir(parents=True, exist_ok=True)
                last_frame: np.ndarray | None = None
                last_tag: str | None = None
                last_roi: list[int] | None = None
                last_bg_roi: list[int] | None = None

                def _on_cam_resp(resp: dict[str, Any]) -> None:
                    nonlocal last_frame, last_tag, last_roi, last_bg_roi
                    if not resp.get("ok"):
                        return
                    if resp.get("event") != "frame":
                        return
                    try:
                        last_frame = np.asarray(resp.get("frame"))
                        last_tag = str(resp.get("tag") or "")
                        last_roi = resp.get("roi")
                        last_bg_roi = resp.get("bg_roi")
                        title = f"Preview {run_idx}/{total}"
                        if last_tag:
                            title += f" | {last_tag}"
                        app.after(
                            0,
                            lambda f=last_frame, t=title, r=last_roi, b=last_bg_roi: plot_camera_frame(
                                app,
                                f,
                                title=t,
                                roi=r,
                                bg_roi=b,
                            ),
                        )
                    except Exception:
                        pass

                daq_resp, cam_responses = run_timed_sequence(
                    seq_cmd=seq_cmd,
                    daq_cmd_q=daq_cmd_q,
                    daq_resp_q=daq_resp_q,
                    cam_cmd_q=cam_cmd_q,
                    cam_resp_q=cam_resp_q,
                    camera_schedule=camera_schedule,
                    ui_pump=None,
                    on_cam_resp=_on_cam_resp,
                )
                if not daq_resp.get("ok"):
                    raise RuntimeError(f"DAQ error: {daq_resp}")

                if last_frame is None:
                    last_ok = select_last_success_response(cam_responses)
                    if last_ok.get("ok") and last_ok.get("event") == "frame":
                        last_frame = np.asarray(last_ok.get("frame"))
                        last_tag = str(last_ok.get("tag") or "")
                        last_roi = last_ok.get("roi")
                        last_bg_roi = last_ok.get("bg_roi")
                        title = f"Preview {run_idx}/{total}"
                        if last_tag:
                            title += f" | {last_tag}"
                        app.after(
                            0,
                            lambda f=last_frame, t=title, r=last_roi, b=last_bg_roi: plot_camera_frame(
                                app,
                                f,
                                title=t,
                                roi=r,
                                bg_roi=b,
                            ),
                        )

                if last_frame is not None:
                    np.save(run_dir / "frame.npy", last_frame)
                    if last_tag:
                        (run_dir / "tag.txt").write_text(str(last_tag), encoding="utf-8")
                else:
                    (run_dir / "no_frame.txt").write_text("no frame captured", encoding="utf-8")

        except Exception as e:
            err = str(e)
        finally:
            app._seq_preview_running = False
            app.after(0, lambda: _set_sequence_preview_ui(app, running=False))
            if err:
                app.after(0, lambda msg=err: messagebox.showerror("Sequence preview", msg))
                app.after(
                    0,
                    lambda msg=err: set_last_error(
                        app,
                        label="Sequence preview",
                        message=msg,
                        log_path=resolve_log_path(app, filename="app.log"),
                    ),
                )
            else:
                app.after(0, lambda: app._cam_status.set("Preview done"))

    t = threading.Thread(target=_worker, daemon=True)
    app._seq_preview_thread = t
    t.start()


def stop_sequence_preview(app: Any) -> None:
    app._seq_preview_running = False
    _set_sequence_preview_ui(app, running=False)


def _set_sequence_preview_ui(app: Any, *, running: bool) -> None:
    try:
        if getattr(app, "seq_preview_btn", None) is not None:
            app.seq_preview_btn.configure(state=("disabled" if running else "normal"))
        if getattr(app, "seq_preview_stop_btn", None) is not None:
            app.seq_preview_stop_btn.configure(state=("normal" if running else "disabled"))
    except Exception:
        pass
