from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from ..hardware import FgDevice, RigolFgDevice
from .spectrum_stage import SpectrumRunResult, run_spectrum_stage


def _setup_fg_for_sweep(
    *,
    fg_connected: bool,
    fg_handle: Any,
    visa_res: str,
    fg_amp_vpp: float,
    no_fg: bool,
    warn_cb: Callable[[str], None] | None,
) -> tuple[FgDevice | Any | None, bool]:
    if no_fg:
        return None, False

    if fg_connected and fg_handle is not None:
        rig = fg_handle
        try:
            if hasattr(rig, "apply"):
                rig.apply({"amp_vpp": fg_amp_vpp})
            elif hasattr(rig, "set_amplitude_vpp"):
                rig.set_amplitude_vpp(fg_amp_vpp)
            if hasattr(rig, "output"):
                rig.output(True)
        except Exception:
            pass
        return rig, False

    visa_res = str(visa_res or "").strip()
    if not visa_res:
        return None, False

    try:
        rig = RigolFgDevice(channel=1, timeout_ms=5000)
        rig.open(visa_res)
        try:
            rig.apply({"amp_vpp": fg_amp_vpp})
        except Exception:
            pass
        rig.output(True)
        return rig, True
    except Exception as e:
        if warn_cb is not None:
            try:
                warn_cb(f"FG init failed, continuing without FG: {e}")
            except Exception:
                pass
        return None, False


def _teardown_fg_for_sweep(rig: FgDevice | Any, rig_owned: bool) -> None:
    if rig is None:
        return
    try:
        if hasattr(rig, "output"):
            rig.output(False)
    except Exception:
        pass
    if rig_owned:
        try:
            rig.close()
        except Exception:
            pass


def run_spectrum_flow(
    *,
    freqs: list[float],
    do_sequence: list[tuple[int, float]],
    insert_index: int,
    ao_width_ms: float,
    seq_cmd: Any | None,
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
    fg_connected: bool,
    fg_handle: Any,
    fg_amp_vpp: float,
    visa_res: str,
    no_fg: bool,
    warn_cb: Callable[[str], None] | None,
) -> SpectrumRunResult:
    rig, rig_owned = _setup_fg_for_sweep(
        fg_connected=fg_connected,
        fg_handle=fg_handle,
        visa_res=visa_res,
        fg_amp_vpp=fg_amp_vpp,
        no_fg=no_fg,
        warn_cb=warn_cb,
    )
    try:
        return run_spectrum_stage(
            freqs=freqs,
            do_sequence=do_sequence,
            insert_index=int(insert_index),
            ao_width_ms=float(ao_width_ms),
            seq_cmd=seq_cmd,
            n_target=int(n_target),
            max_attempt=int(max_attempt),
            settle_s=float(settle_s),
            update_interval_s=float(update_interval_s),
            daq_cmd_q=daq_cmd_q,
            daq_resp_q=daq_resp_q,
            cam_cmd_q=cam_cmd_q,
            cam_resp_q=cam_resp_q,
            ao_rate_hz=float(ao_rate_hz),
            mpq_get_with_ui=mpq_get_with_ui,
            should_stop=should_stop,
            ui_pump=ui_pump,
            status_cb=status_cb,
            update_point_cb=update_point_cb,
            out_dir=out_dir,
            rig=rig,
        )
    finally:
        _teardown_fg_for_sweep(rig, rig_owned)
