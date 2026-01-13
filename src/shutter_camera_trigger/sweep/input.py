from __future__ import annotations

from pathlib import Path
from typing import Any

from ..gui_support.validators import parse_camera_trigger_cfg, parse_exposure_s_safe, parse_fg_amp_vpp_safe
from ..gui_support.diagnostics import resolve_log_path, set_last_error
from .controller import SweepInput
from .session_parse import parse_freqs_from_expressions, read_sequence_json_params
from ..sequence.parser import parse_sequence_text as parse_sequence_text_raw

SEQUENCE_BITS = 4


def collect_sweep_input(app: Any, *, default_daq_device: str) -> SweepInput | None:
    trig_cfg = parse_camera_trigger_cfg(app)
    try:
        freqs = parse_freqs_from_expressions(
            start_expr=app.sw_freq_start.get(),
            stop_expr=app.sw_freq_stop.get(),
            step_expr=app.sw_freq_step.get(),
        )
        seq_path = Path(app.sw_seq_path.get())
        seq_params = read_sequence_json_params(seq_path=seq_path)
        do_sequence = parse_sequence_text_raw(seq_params.sequence_text, bits=SEQUENCE_BITS)
        insert_index = int(seq_params.ao_insert_index)
        ao_width_ms = float(seq_params.ao_width_ms)
        n_target = int(app.sw_n_target.get())
        max_attempt = int(app.sw_max_attempt.get())
        settle_s = float(app.sw_settle_s.get())
        update_interval = max(0.2, float(app.sw_update_interval.get()))
        daq_mode = app.sw_daq_mode.get()
        cam_mode = app.sw_cam_mode.get()
        cam_exposure_s = parse_exposure_s_safe(app)
        device = app.sw_device.get().strip() or default_daq_device
        visa_res = app.sw_visa.get().strip()
        no_fg = bool(app.sw_no_fg.get())
        fg_amp_vpp = parse_fg_amp_vpp_safe(app, max_mvpp=810.0, default_vpp=0.790)
        dry_image_dir = app.dry_image_dir_var.get().strip()
    except Exception as e:
        from tkinter import messagebox

        messagebox.showerror("Sweep", str(e))
        set_last_error(
            app,
            label="Sweep input",
            message=str(e),
            log_path=resolve_log_path(app, filename="sweep.log"),
        )
        return None
    return SweepInput(
        freqs=freqs,
        do_sequence=do_sequence,
        insert_index=insert_index,
        ao_width_ms=ao_width_ms,
        n_target=n_target,
        max_attempt=max_attempt,
        settle_s=settle_s,
        update_interval=update_interval,
        daq_mode=daq_mode,
        cam_mode=cam_mode,
        cam_exposure_s=cam_exposure_s,
        device=device,
        visa_res=visa_res,
        no_fg=no_fg,
        fg_amp_vpp=fg_amp_vpp,
        dry_image_dir=dry_image_dir,
        trig_cfg=trig_cfg,
        seq_path=seq_path,
        camera_verbose=app.camera_verbose_var.get(),
    )
