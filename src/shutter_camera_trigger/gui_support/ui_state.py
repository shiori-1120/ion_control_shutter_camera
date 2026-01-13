from __future__ import annotations

import tkinter as tk
from typing import Any


def init_ui_state(
    app: Any,
    *,
    default_daq_device: str,
    default_fg_resource: str,
    default_fg_amp_mvpp: str,
    default_seq_path: str,
) -> None:
    app.fg_resource_var = tk.StringVar(value=default_fg_resource)
    app.fg_amp_mvpp_var = tk.StringVar(value=default_fg_amp_mvpp)
    app.sw_no_fg = tk.BooleanVar(value=True)
    app.camera_mode_top_var = tk.StringVar(value="dry")
    app.camera_exposure_ms_var = tk.StringVar(value="100.0")
    app.dry_image_dir_var = tk.StringVar(value="")
    app.sw_seq_path = tk.StringVar(value=default_seq_path)

    app.camera_trigger_source_var = tk.StringVar(value="EXTERNAL")
    app.camera_trigger_connector_var = tk.StringVar(value="BNC")
    app.camera_trigger_polarity_var = tk.StringVar(value="POSITIVE")
    app.camera_trigger_active_var = tk.StringVar(value="EDGE")
    app.camera_trigger_mode_var = tk.StringVar(value="NORMAL")
    app.camera_trigger_delay_s_var = tk.StringVar(value="")
    app.camera_verbose_var = tk.BooleanVar(value=False)
    app.camera_verbose_additional_only = True
    app.show_debug_fields = True

    app.camera_subarray_enable_var = tk.BooleanVar(value=False)
    app.camera_sub_x_var = tk.StringVar(value="0")
    app.camera_sub_y_var = tk.StringVar(value="0")
    app.camera_sub_w_var = tk.StringVar(value="")
    app.camera_sub_h_var = tk.StringVar(value="")

    app.device_var = tk.StringVar(value=default_daq_device)
    app.device_mode_var = tk.StringVar(value="dry")
    app.width_var = tk.StringVar(value="15.0")

    app.sw_n_target = tk.StringVar(value="50")
    app.sw_max_attempt = tk.StringVar(value="100")
    app.sw_settle_s = tk.StringVar(value="0.02")
    app.sw_update_interval = tk.StringVar(value="1.0")
