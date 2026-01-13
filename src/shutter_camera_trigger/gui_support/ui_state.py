from __future__ import annotations

import tkinter as tk
from typing import Any


def init_ui_state(app: Any, *, default_daq_device: str, default_fg_resource: str, default_fg_amp_mvpp: str) -> None:
    app.fg_resource_var = tk.StringVar(value=default_fg_resource)
    app.fg_amp_mvpp_var = tk.StringVar(value=default_fg_amp_mvpp)
    app.camera_mode_top_var = tk.StringVar(value="dry")
    app.camera_exposure_ms_var = tk.StringVar(value="100.0")
    app.dry_image_dir_var = tk.StringVar(value="")

    app.camera_trigger_source_var = tk.StringVar(value="EXTERNAL")
    app.camera_trigger_connector_var = tk.StringVar(value="BNC")
    app.camera_trigger_polarity_var = tk.StringVar(value="POSITIVE")
    app.camera_trigger_active_var = tk.StringVar(value="EDGE")
    app.camera_trigger_mode_var = tk.StringVar(value="NORMAL")
    app.camera_trigger_delay_s_var = tk.StringVar(value="")
    app.camera_verbose_var = tk.BooleanVar(value=False)

    app.camera_subarray_enable_var = tk.BooleanVar(value=False)
    app.camera_sub_x_var = tk.StringVar(value="0")
    app.camera_sub_y_var = tk.StringVar(value="0")
    app.camera_sub_w_var = tk.StringVar(value="")
    app.camera_sub_h_var = tk.StringVar(value="")

    app.device_var = tk.StringVar(value=default_daq_device)
    app.device_mode_var = tk.StringVar(value="real")
    app.width_var = tk.StringVar(value="15.0")