from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
import os

from ..hardware import DaqQueueDevice, DaqSequenceCommand
from ..sequence.spec import build_sequence_spec, compile_sequence_spec
from .session_config import build_daq_sequence_command


@dataclass(frozen=True)
class SweepSessionReady:
    ok: bool
    session: dict[str, Any] | None
    workers: Any | None
    error: str | None = None


def prepare_sweep_session(
    *,
    freqs,
    do_sequence,
    insert_index,
    ao_width_ms,
    n_target,
    max_attempt,
    settle_s,
    update_interval,
    daq_mode,
    cam_mode,
    device,
    visa_res,
    no_fg,
    fg_amp_vpp,
    trig_cfg,
    cam_exposure_s,
    seq_path,
    camera_actions,
    sync_markers,
    camera_verbose,
    subarray_cb: Callable[[dict], None],
    write_sweep_config_json,
    SweepPersistedConfig,
    create_sweep_workers,
    bootstrap_workers_for_sweep,
    build_sweep_session_dict,
    run_roi_bootstrap_stage,
    AO_RATE_HZ,
    NM_397,
    CAMERA_TRIGGER,
    ROI_PULSE_S,
    ROI_IDLE_S,
    ROI_MAX_ATTEMPT,
    log_dir,
    run_id,
    out_dir: Path,
    mpq_get_with_ui,
    ui_pump,
    status_cb,
    show_error_cb,
    stop_sweep_cb,
    write_last_worker_pids_cb,
    format_worker_failure,
) -> SweepSessionReady:
    """
    外部依存を引数で受け取り、sweep準備の本体を実行する。
    成功時は (True, session_dict)、失敗時は (False, None) を返す。
    """
    try:
        write_sweep_config_json(
            out_dir=out_dir,
            cfg=SweepPersistedConfig(
                freqs=freqs,
                n_target=n_target,
                max_attempt=max_attempt,
                settle_s=settle_s,
                update_interval=update_interval,
                daq_mode=daq_mode,
                device=device,
                sequence_json=str(seq_path),
                insert_index=insert_index,
                ao_width_ms=ao_width_ms,
                camera_actions=list(camera_actions),
                sync_markers=list(sync_markers),
                camera_mode=cam_mode,
                camera_exposure_s=float(cam_exposure_s),
                fg_amp_mvpp=float(fg_amp_vpp) * 1000.0,
                roi_bootstrap={
                    "pulse_s": ROI_PULSE_S,
                    "idle_s": ROI_IDLE_S,
                    "max_attempt": ROI_MAX_ATTEMPT,
                },
            )
        )
        cam_cfg: dict[str, Any] = {
            "mode": cam_mode,
            "exposure_s": float(cam_exposure_s),
            "frame_timeout_s": max(1.0, float(cam_exposure_s) * 4.0 + 0.5),
            "bootstrap_n": 10,
            "trigger": dict(trig_cfg),
            "verbose": bool(camera_verbose),
        }
        subarray_cb(cam_cfg)
        if log_dir is not None:
            cam_cfg["log_path"] = str(Path(log_dir) / "camera_worker.log")
        else:
            cam_cfg["log_path"] = str(out_dir / "camera_worker.log")
        if run_id:
            cam_cfg["run_id"] = str(run_id)
        daq_log_path = str(Path(log_dir) / "daq_worker.log") if log_dir is not None else None
        workers = create_sweep_workers(
            device=device,
            daq_mode=daq_mode,
            cam_cfg=cam_cfg,
            daq_log_path=daq_log_path,
            run_id=str(run_id) if run_id else None,
        )
        daq_p = workers.daq_proc
        cam_p = workers.cam_proc
        daq_cmd_q = workers.daq_cmd_q
        daq_resp_q = workers.daq_resp_q
        cam_cmd_q = workers.cam_cmd_q
        cam_resp_q = workers.cam_resp_q
        daq_p.start()
        trig_src = str(trig_cfg.get("source") or "EXTERNAL").strip().upper() or "EXTERNAL"
        prime_seq_one = [
            (NM_397, ROI_IDLE_S),
            (NM_397 | CAMERA_TRIGGER, ROI_PULSE_S),
            (NM_397, ROI_IDLE_S),
        ]
        cam_log_path = str(Path(log_dir) / "camera_worker.log") if log_dir is not None else str(out_dir / "camera_worker.log")
        daq_device = DaqQueueDevice(cmd_q=daq_cmd_q, resp_q=daq_resp_q)
        prime_cmd = DaqSequenceCommand(
            do_sequence=prime_seq_one,
            ao_insert_index=-1,
            ao_width_ms=0.0,
            ao_rate_hz=AO_RATE_HZ,
            ao_v_high=5.0,
            ao_v_low=0.0,
        )
        trig_src = str(trig_cfg.get("source") or "EXTERNAL").strip().upper() or "EXTERNAL"
        def _timeout_from_env(name: str, default: float) -> float:
            try:
                raw = os.environ.get(name, "")
                if raw is None:
                    return float(default)
                raw = str(raw).strip()
                if not raw:
                    return float(default)
                val = float(raw)
                return float(default) if val <= 0 else float(val)
            except Exception:
                return float(default)

        base_ready_timeout = 180.0 if trig_src in ("EXTERNAL", "EXT", "2", "") else 30.0
        cam_ready_timeout_s = _timeout_from_env("ION_CONTROL_CAMERA_READY_TIMEOUT_S", base_ready_timeout)
        prime_deadline_s = _timeout_from_env("ION_CONTROL_CAMERA_PRIME_DEADLINE_S", cam_ready_timeout_s)

        _ = bootstrap_workers_for_sweep(
            daq_resp_q=daq_resp_q,
            cam_proc=cam_p,
            cam_resp_q=cam_resp_q,
            mpq_get_with_ui=mpq_get_with_ui,
            format_worker_failure=format_worker_failure,
            cam_log_path=cam_log_path,
            cam_mode=cam_mode,
            trig_src=trig_src,
            prime_cmd=prime_cmd,
            daq_send=daq_cmd_q.put,
            daq_recv=lambda timeout, label: mpq_get_with_ui(daq_resp_q, timeout=timeout, label=label),
            daq_device=daq_device,
            ui_pump=ui_pump,
            status_cb=status_cb,
            daq_ready_timeout_s=5.0,
            cam_ready_timeout_s=cam_ready_timeout_s,
            prime_deadline_s=prime_deadline_s,
        )
        write_last_worker_pids_cb({
            "t_iso": datetime.now().isoformat(timespec="seconds"),
            "daq_pid": int(getattr(daq_p, "pid", 0) or 0),
            "cam_pid": int(getattr(cam_p, "pid", 0) or 0),
        })
        roi_ok = run_roi_bootstrap_stage(
            daq_cmd_q=daq_cmd_q,
            daq_resp_q=daq_resp_q,
            cam_cmd_q=cam_cmd_q,
            cam_resp_q=cam_resp_q,
            nm_397=NM_397,
            camera_trigger=CAMERA_TRIGGER,
            roi_pulse_s=ROI_PULSE_S,
            roi_idle_s=ROI_IDLE_S,
            max_attempt=ROI_MAX_ATTEMPT,
            status_cb=status_cb,
            ui_pump=ui_pump,
        )
        if not roi_ok:
            error_msg = "ROI bootstrap failed"
            try:
                show_error_cb("Sweep", error_msg)
            except Exception:
                pass
            stop_sweep_cb(clean_only=True)
            return SweepSessionReady(ok=False, session=None, workers=None, error=error_msg)
        session_dict = build_sweep_session_dict(
            freqs=freqs,
            do_sequence=do_sequence,
            insert_index=insert_index,
            ao_width_ms=ao_width_ms,
            seq_cmd=build_daq_sequence_command(
                do_sequence=do_sequence,
                insert_index=insert_index,
                ao_width_ms=ao_width_ms,
                ao_rate_hz=AO_RATE_HZ,
                ao_v_high=5.0,
                ao_v_low=0.0,
            ),
            camera_commands=compile_sequence_spec(
                build_sequence_spec(
                    do_sequence=do_sequence,
                    ao_insert_index=insert_index,
                    ao_width_ms=ao_width_ms,
                    ao_rate_hz=AO_RATE_HZ,
                    ao_v_high=5.0,
                    ao_v_low=0.0,
                    camera_actions=camera_actions,
                    sync_markers=sync_markers,
                )
            )[1],
            camera_actions=camera_actions,
            sync_markers=sync_markers,
            n_target=n_target,
            max_attempt=max_attempt,
            settle_s=settle_s,
            update_interval=update_interval,
            daq_mode=daq_mode,
            cam_mode=cam_mode,
            device=device,
            visa_res=visa_res,
            no_fg=no_fg,
            fg_amp_vpp=fg_amp_vpp,
            trig_cfg=dict(trig_cfg),
            cam_exposure_s=float(cam_exposure_s),
            seq_path=str(seq_path),
        )
        return SweepSessionReady(ok=True, session=session_dict, workers=workers, error=None)
    except Exception as e:
        error_msg = f"Worker init failed ({type(e).__name__}): {e}"
        try:
            show_error_cb("Sweep", error_msg)
        except Exception:
            pass
        stop_sweep_cb(clean_only=True)
        return SweepSessionReady(ok=False, session=None, workers=None, error=error_msg)
