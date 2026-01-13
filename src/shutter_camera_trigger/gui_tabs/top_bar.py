from __future__ import annotations

import tkinter as tk
from typing import Any, Callable
from tkinter import ttk


def build_top_bar(
    app: Any,
    *,
    connect_cb: Callable[[], None],
    disconnect_cb: Callable[[], None],
    fg_connect_cb: Callable[[], None],
    fg_disconnect_cb: Callable[[], None],
    cam_check_cb: Callable[[], None],
    browse_dry_images_cb: Callable[[], None],
) -> None:
    top = ttk.Frame(app, padding=10)
    top.pack(side=tk.TOP, fill=tk.X)

    ttk.Label(top, text="Device").grid(row=0, column=0, sticky=tk.W)
    ttk.Entry(top, textvariable=app.device_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=5)

    ttk.Label(top, text="DAQ mode").grid(row=0, column=2, sticky=tk.W)
    ttk.Combobox(top, textvariable=app.device_mode_var, values=["real", "dry"], width=6, state="readonly").grid(
        row=0, column=3, sticky=tk.W, padx=5
    )

    ttk.Label(top, text="AO width (ms)").grid(row=0, column=4, sticky=tk.W)
    ttk.Entry(top, textvariable=app.width_var, width=10).grid(row=0, column=5, sticky=tk.W, padx=5)

    app.connect_btn = ttk.Button(top, text="Connect", command=connect_cb)
    app.connect_btn.grid(row=0, column=6, padx=5)
    app.disconnect_btn = ttk.Button(top, text="Disconnect", command=disconnect_cb, state=tk.DISABLED)
    app.disconnect_btn.grid(row=0, column=7)

    ttk.Label(top, text="FG VISA").grid(row=1, column=0, sticky=tk.W, pady=(6, 0))
    ttk.Entry(top, textvariable=app.fg_resource_var, width=32).grid(
        row=1, column=1, columnspan=3, sticky=tk.W, padx=5, pady=(6, 0)
    )
    app.fg_connect_btn = ttk.Button(top, text="FG Connect", command=fg_connect_cb)
    app.fg_connect_btn.grid(row=1, column=4, padx=5, pady=(6, 0))
    app.fg_disconnect_btn = ttk.Button(top, text="FG Disconnect", command=fg_disconnect_cb, state=tk.DISABLED)
    app.fg_disconnect_btn.grid(row=1, column=5, padx=5, pady=(6, 0))

    ttk.Label(top, text="FG amp (mVpp)").grid(row=1, column=6, sticky=tk.W, pady=(6, 0))
    ttk.Entry(top, textvariable=app.fg_amp_mvpp_var, width=10).grid(row=1, column=7, sticky=tk.W, padx=5, pady=(6, 0))

    ttk.Label(top, text="Camera mode").grid(row=2, column=0, sticky=tk.W, pady=(6, 0))
    ttk.Combobox(top, textvariable=app.camera_mode_top_var, values=["dry", "real"], width=6, state="readonly").grid(
        row=2, column=1, sticky=tk.W, padx=5, pady=(6, 0)
    )

    ttk.Label(top, text="Exposure (ms)").grid(row=2, column=2, sticky=tk.W, pady=(6, 0))
    ttk.Entry(top, textvariable=app.camera_exposure_ms_var, width=10).grid(row=2, column=3, sticky=tk.W, padx=5, pady=(6, 0))

    app.cam_check_btn = ttk.Button(top, text="Camera check", command=cam_check_cb)
    app.cam_check_btn.grid(row=2, column=4, padx=5, pady=(6, 0))

    ttk.Label(top, text="Dry images (dry cam)").grid(row=2, column=5, sticky=tk.W, pady=(6, 0))
    ttk.Entry(top, textvariable=app.dry_image_dir_var, width=30).grid(
        row=2, column=6, columnspan=2, sticky=tk.W, padx=5, pady=(6, 0)
    )
    ttk.Button(top, text="...", width=3, command=browse_dry_images_cb).grid(row=2, column=8, pady=(6, 0))

    ttk.Label(top, text="Cam trig").grid(row=3, column=0, sticky=tk.W, pady=(6, 0))
    ttk.Combobox(
        top,
        textvariable=app.camera_trigger_source_var,
        values=["EXTERNAL", "INTERNAL"],
        width=9,
        state="readonly",
    ).grid(row=3, column=1, sticky=tk.W, padx=5, pady=(6, 0))
    ttk.Label(top, text="Conn").grid(row=3, column=2, sticky=tk.W, pady=(6, 0))
    ttk.Combobox(
        top,
        textvariable=app.camera_trigger_connector_var,
        values=["BNC", "MULTI", "INTERFACE"],
        width=9,
        state="readonly",
    ).grid(row=3, column=3, sticky=tk.W, padx=5, pady=(6, 0))
    ttk.Label(top, text="Pol").grid(row=3, column=4, sticky=tk.W, pady=(6, 0))
    ttk.Combobox(
        top,
        textvariable=app.camera_trigger_polarity_var,
        values=["POSITIVE", "NEGATIVE"],
        width=9,
        state="readonly",
    ).grid(row=3, column=5, sticky=tk.W, padx=5, pady=(6, 0))
    ttk.Label(top, text="Act").grid(row=3, column=6, sticky=tk.W, pady=(6, 0))
    ttk.Combobox(
        top,
        textvariable=app.camera_trigger_active_var,
        values=["EDGE", "LEVEL"],
        width=7,
        state="readonly",
    ).grid(row=3, column=7, sticky=tk.W, padx=5, pady=(6, 0))

    ttk.Label(top, text="Mode").grid(row=4, column=0, sticky=tk.W, pady=(6, 0))
    ttk.Combobox(
        top,
        textvariable=app.camera_trigger_mode_var,
        values=["NORMAL", "START"],
        width=9,
        state="readonly",
    ).grid(row=4, column=1, sticky=tk.W, padx=5, pady=(6, 0))
    ttk.Label(top, text="Delay (s)").grid(row=4, column=2, sticky=tk.W, pady=(6, 0))
    ttk.Entry(top, textvariable=app.camera_trigger_delay_s_var, width=10).grid(
        row=4, column=3, sticky=tk.W, padx=5, pady=(6, 0)
    )
    ttk.Checkbutton(top, text="Cam verbose", variable=app.camera_verbose_var).grid(
        row=4, column=4, sticky=tk.W, padx=5, pady=(6, 0)
    )

    sub = ttk.LabelFrame(top, text="Subarray")
    sub.grid(row=5, column=0, columnspan=9, sticky=tk.W + tk.E, pady=(8, 0))

    ttk.Checkbutton(sub, text="Enable", variable=app.camera_subarray_enable_var).grid(
        row=0, column=0, sticky=tk.W, padx=6, pady=4
    )
    ttk.Label(sub, text="X").grid(row=0, column=1, sticky=tk.W)
    ttk.Entry(sub, textvariable=app.camera_sub_x_var, width=8).grid(row=0, column=2, sticky=tk.W, padx=(2, 10))
    ttk.Label(sub, text="Y").grid(row=0, column=3, sticky=tk.W)
    ttk.Entry(sub, textvariable=app.camera_sub_y_var, width=8).grid(row=0, column=4, sticky=tk.W, padx=(2, 10))
    ttk.Label(sub, text="W").grid(row=0, column=5, sticky=tk.W)
    ttk.Entry(sub, textvariable=app.camera_sub_w_var, width=8).grid(row=0, column=6, sticky=tk.W, padx=(2, 10))
    ttk.Label(sub, text="H").grid(row=0, column=7, sticky=tk.W)
    ttk.Entry(sub, textvariable=app.camera_sub_h_var, width=8).grid(row=0, column=8, sticky=tk.W, padx=(2, 10))

    try:
        sub.grid_columnconfigure(9, weight=1)
    except Exception:
        pass

    app.status_var = tk.StringVar(value="Disconnected")
    ttk.Label(top, textvariable=app.status_var).grid(row=6, column=0, columnspan=9, sticky=tk.W, pady=(8, 0))

    top.grid_columnconfigure(9, weight=1)
